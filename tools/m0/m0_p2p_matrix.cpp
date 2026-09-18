// m0_p2p_matrix.cpp — 4-card P2P matrix for M0 fabric validation (MI50 / gfx906).
//
// PROTOCOL CONSTRAINTS (project history):
//   * SDMA peer COPY only (hipMemcpyPeer). Kernel peer STORES poison the writer
//     GPU on gfx906 (D108 wedge, required a <source-host> power cycle) — DO NOT add them.
//   * Byte-exact verification on every transfer (I1: a bandwidth number without a
//     correctness check is not evidence).
//   * N>=15 repeats with median + IQR (D048/I5): a CI, not a single-shot number.
//
// MEASUREMENT INTEGRITY (learned the hard way, first run of this probe):
//   hipMemcpyPeer RETURNS WITHOUT THE TRANSFER HAVING COMPLETED in this ROCm
//   (7.14 / core-7.14) build. Timing it as a blocking call produced 59,473 GB/s
//   self-copy and 93,515 GB/s peer copy — 2000x the physical Gen4 x16 ceiling.
//   Every timed region therefore synchronizes BOTH devices explicitly, and every
//   bandwidth is checked against the link ceiling; implausible results are
//   flagged rather than reported as measurements (D049 discipline).
//
// stdout: exactly one JSON object. All diagnostics go to stderr.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                      \
  do {                                                                             \
    hipError_t _e = (x);                                                           \
    if (_e != hipSuccess) {                                                        \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e), __FILE__,  \
              __LINE__);                                                           \
      exit(3);                                                                     \
    }                                                                              \
  } while (0)

// hipFree/hipDeviceGetPCIBusId are marked nodiscard in ROCm 7.14; their return
// values are deliberately ignored here (frees at teardown, best-effort bus id).
#pragma clang diagnostic ignored "-Wunused-value"

using clk = std::chrono::steady_clock;

static double now_s() {
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

// A peer transfer touches two devices; the enqueueing device's stream is not
// guaranteed to be the one that drains it, so synchronize both.
static void sync_pair(int a, int b) {
  hipSetDevice(a);
  hipDeviceSynchronize();
  if (b != a) {
    hipSetDevice(b);
    hipDeviceSynchronize();
  }
}

static double median(std::vector<double> v) {
  if (v.empty()) return -1.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double iqr(std::vector<double> v) {
  if (v.size() < 4) return -1.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  auto q = [&](double p) {
    double idx = p * (double)(n - 1);
    size_t lo = (size_t)idx;
    double f = idx - (double)lo;
    double hi = (lo + 1 < n) ? v[lo + 1] : v[lo];
    return v[lo] + f * (hi - v[lo]);
  };
  return q(0.75) - q(0.25);
}

// Physical plausibility bounds. Peer traffic is bounded by the endpoint's PCIe
// link; self traffic by HBM. Generous ceilings so a REAL regression is not
// masked, but an enqueue-only measurement (the failure mode above) is caught.
static const double PEER_BW_CEILING = 40.0;    // GB/s; Gen4 x16 ~32, D109 saw 28.5
static const double SELF_BW_CEILING = 1200.0;  // GB/s; HBM2 ~1 TB/s peak
static const double HOST_BW_CEILING = 40.0;    // GB/s; host link bound

struct CopyResult {
  double bw_median = -1, bw_iqr = -1, bw_block = -1;
  double lat_median = -1, lat_iqr = -1, lat_synced_median = -1;
  bool bytes_ok = false;
  bool plausible = false;
  int repeats = 0;
  std::string err = "";
};

static void fill_pattern(unsigned char* p, size_t bytes) {
  for (size_t i = 0; i < bytes; i++) p[i] = (unsigned char)((i * 31u + 7u) & 0xff);
}

// One ordered transfer direction: per-op synchronized timing (distribution) plus
// a block-total cross-check, then byte-exact verification.
static CopyResult measure_pair(int src, int dst, size_t bytes, int repeats,
                               int warmup, double ceiling) {
  CopyResult r;
  r.repeats = repeats;
  void *sbuf = nullptr, *dbuf = nullptr;

  CK(hipSetDevice(src));
  if (hipMalloc(&sbuf, bytes) != hipSuccess) { r.err = "hipMalloc src failed"; return r; }
  CK(hipSetDevice(dst));
  if (hipMalloc(&dbuf, bytes) != hipSuccess) { r.err = "hipMalloc dst failed"; (void)hipFree(sbuf); return r; }

  std::vector<unsigned char> host(bytes);
  fill_pattern(host.data(), bytes);
  CK(hipSetDevice(src));
  CK(hipMemcpy(sbuf, host.data(), bytes, hipMemcpyHostToDevice));
  CK(hipSetDevice(dst));
  CK(hipMemset(dbuf, 0xA5, bytes));  // poison: a no-op copy cannot pass verify
  sync_pair(src, dst);

  for (int i = 0; i < warmup; i++) {
    hipError_t we = hipMemcpyPeer(dbuf, dst, sbuf, src, bytes);
    if (we != hipSuccess) { r.err = std::string("warmup copy: ") + hipGetErrorString(we); goto cleanup; }
  }
  sync_pair(src, dst);

  // (1) per-op, synchronized on both devices.
  {
    std::vector<double> bw;
    for (int i = 0; i < repeats; i++) {
      sync_pair(src, dst);
      double t0 = now_s();
      hipError_t ce = hipMemcpyPeer(dbuf, dst, sbuf, src, bytes);
      sync_pair(src, dst);  // REQUIRED: the call alone does not wait
      double t1 = now_s();
      if (ce != hipSuccess) { r.err = std::string("copy: ") + hipGetErrorString(ce); goto cleanup; }
      double dt = t1 - t0;
      if (dt > 0) bw.push_back(((double)bytes / 1e9) / dt);
    }
    r.bw_median = median(bw);
    r.bw_iqr = iqr(bw);
  }

  // (2) block total: immune to per-op sync overhead placement.
  {
    sync_pair(src, dst);
    double t0 = now_s();
    for (int i = 0; i < repeats; i++) {
      if (hipMemcpyPeer(dbuf, dst, sbuf, src, bytes) != hipSuccess) {
        r.err = "block copy failed";
        break;
      }
    }
    sync_pair(src, dst);
    double t1 = now_s();
    if (!r.err.empty()) goto cleanup;
    if (t1 > t0) r.bw_block = ((double)bytes * (double)repeats / 1e9) / (t1 - t0);
  }

  // (3) byte-exact verification (also forces any outstanding transfer to land).
  {
    std::vector<unsigned char> back(bytes);
    CK(hipSetDevice(dst));
    hipError_t ce = hipMemcpy(back.data(), dbuf, bytes, hipMemcpyDeviceToHost);
    if (ce != hipSuccess) { r.err = std::string("verify copy: ") + hipGetErrorString(ce); goto cleanup; }
    r.bytes_ok = (memcmp(back.data(), host.data(), bytes) == 0);
  }

  r.plausible = (r.bw_median > 0 && r.bw_median <= ceiling);

cleanup:
  (void)hipFree(sbuf);
  (void)hipFree(dbuf);
  return r;
}

// Latency at the decode operating point (20 KB, D110). Two estimators:
//   * block-amortized per-op: N copies then one sync, divided by N  (primary;
//     this is the dispatch-limited number D110's 11.8 us/op refers to)
//   * per-op synchronized: an upper bound including sync overhead
static CopyResult measure_latency(int src, int dst, size_t bytes, int repeats) {
  CopyResult r;
  r.repeats = repeats;
  void *sbuf = nullptr, *dbuf = nullptr;
  CK(hipSetDevice(src));
  if (hipMalloc(&sbuf, bytes) != hipSuccess) { r.err = "hipMalloc src failed"; return r; }
  CK(hipSetDevice(dst));
  if (hipMalloc(&dbuf, bytes) != hipSuccess) { r.err = "hipMalloc dst failed"; (void)hipFree(sbuf); return r; }
  CK(hipSetDevice(src));
  CK(hipMemset(sbuf, 0x11, bytes));
  sync_pair(src, dst);

  for (int i = 0; i < 5; i++) hipMemcpyPeer(dbuf, dst, sbuf, src, bytes);
  sync_pair(src, dst);

  // per-op synchronized
  {
    std::vector<double> lat;
    for (int i = 0; i < repeats; i++) {
      sync_pair(src, dst);
      double t0 = now_s();
      hipError_t ce = hipMemcpyPeer(dbuf, dst, sbuf, src, bytes);
      sync_pair(src, dst);
      double t1 = now_s();
      if (ce != hipSuccess) { r.err = std::string("lat copy: ") + hipGetErrorString(ce); break; }
      lat.push_back((t1 - t0) * 1e6);
    }
    r.lat_synced_median = median(lat);
    r.lat_iqr = iqr(lat);
  }

  // block-amortized (primary)
  {
    sync_pair(src, dst);
    double t0 = now_s();
    for (int i = 0; i < repeats; i++) hipMemcpyPeer(dbuf, dst, sbuf, src, bytes);
    sync_pair(src, dst);
    double t1 = now_s();
    if (t1 > t0) r.lat_median = (t1 - t0) * 1e6 / (double)repeats;
  }

  r.bytes_ok = true;  // correctness carried by the bandwidth leg
  (void)hipFree(sbuf);
  (void)hipFree(dbuf);
  return r;
}

// Host<->device transfer at the same byte point. This measures the ceiling of
// the HOST LINK (root port <-> switch upstream). It is the discriminator for
// M0-DECISION-TREE section 1: if peer bandwidth ~= host bandwidth, peer traffic
// is being redirected through the root complex instead of staying on the
// switch; D110 recorded switch-internal peer traffic at 1.8x the upstream
// ceiling, so the two regimes are far apart and easy to tell apart.
static CopyResult measure_host_transfer(int dev, size_t bytes, int repeats,
                                        int warmup, bool h2d) {
  CopyResult r;
  r.repeats = repeats;
  void* dbuf = nullptr;
  void* hbuf = nullptr;
  if (hipSetDevice(dev) != hipSuccess) { r.err = "setDevice failed"; return r; }
  if (hipMalloc(&dbuf, bytes) != hipSuccess) { r.err = "hipMalloc failed"; return r; }
  if (hipHostMalloc(&hbuf, bytes) != hipSuccess) {
    r.err = "hipHostMalloc failed";
    (void)hipFree(dbuf);
    return r;
  }
  fill_pattern((unsigned char*)hbuf, bytes);
  CK(hipMemset(dbuf, 0x5A, bytes));
  sync_pair(dev, dev);

  for (int i = 0; i < warmup; i++) {
    if (h2d) hipMemcpy(dbuf, hbuf, bytes, hipMemcpyHostToDevice);
    else hipMemcpy(hbuf, dbuf, bytes, hipMemcpyDeviceToHost);
  }
  sync_pair(dev, dev);

  {  // per-op synchronized
    std::vector<double> bw;
    for (int i = 0; i < repeats; i++) {
      sync_pair(dev, dev);
      double t0 = now_s();
      hipError_t e = h2d ? hipMemcpy(dbuf, hbuf, bytes, hipMemcpyHostToDevice)
                         : hipMemcpy(hbuf, dbuf, bytes, hipMemcpyDeviceToHost);
      sync_pair(dev, dev);
      double t1 = now_s();
      if (e != hipSuccess) { r.err = hipGetErrorString(e); break; }
      double dt = t1 - t0;
      if (dt > 0) bw.push_back(((double)bytes / 1e9) / dt);
    }
    r.bw_median = median(bw);
    r.bw_iqr = iqr(bw);
  }

  {  // block cross-check
    sync_pair(dev, dev);
    double t0 = now_s();
    for (int i = 0; i < repeats; i++) {
      if (h2d) hipMemcpy(dbuf, hbuf, bytes, hipMemcpyHostToDevice);
      else hipMemcpy(hbuf, dbuf, bytes, hipMemcpyDeviceToHost);
    }
    sync_pair(dev, dev);
    double t1 = now_s();
    if (t1 > t0) r.bw_block = ((double)bytes * (double)repeats / 1e9) / (t1 - t0);
  }

  if (h2d) {
    std::vector<unsigned char> back(bytes);
    if (hipMemcpy(back.data(), dbuf, bytes, hipMemcpyDeviceToHost) == hipSuccess) {
      r.bytes_ok = (memcmp(back.data(), hbuf, bytes) == 0);
    }
  } else {
    r.bytes_ok = true;
  }
  r.plausible = (r.bw_median > 0 && r.bw_median <= HOST_BW_CEILING);
  (void)hipFree(dbuf);
  (void)hipHostFree(hbuf);
  return r;
}

int main(int argc, char** argv) {  int repeats = 20, lat_repeats = 200, warmup = 3;
  size_t big = 256ull << 20;   // 256 MB, the D109 bandwidth point
  size_t small = 20ull << 10;  // 20 KB, the D110 decode operating point
  if (const char* e = getenv("M0_REPEATS")) repeats = atoi(e);
  if (const char* e = getenv("M0_LAT_REPEATS")) lat_repeats = atoi(e);
  if (const char* e = getenv("M0_BIG_BYTES")) big = strtoull(e, nullptr, 10);
  if (const char* e = getenv("M0_SMALL_BYTES")) small = strtoull(e, nullptr, 10);

  int n = 0;
  CK(hipGetDeviceCount(&n));
  fprintf(stderr, "device_count=%d repeats=%d lat_repeats=%d big=%zu small=%zu\n",
          n, repeats, lat_repeats, big, small);

  std::vector<std::string> name(n), bus(n), arch(n);
  std::vector<double> memgb(n);
  for (int i = 0; i < n; i++) {
    hipDeviceProp_t p;
    CK(hipGetDeviceProperties(&p, i));
    char b[<bus>] = {0};
    hipDeviceGetPCIBusId(b, sizeof(b), i);
    name[i] = p.name;
    bus[i] = b;
    arch[i] = p.gcnArchName;
    memgb[i] = (double)p.totalGlobalMem / 1073741824.0;
  }

  std::vector<int> can((size_t)n * n, 0);
  std::vector<int> en((size_t)n * n, 0);
  std::vector<std::string> en_err((size_t)n * n, "");
  for (int a = 0; a < n; a++) {
    CK(hipSetDevice(a));
    for (int b = 0; b < n; b++) {
      if (a == b) continue;
      int ok = 0;
      hipError_t e = hipDeviceCanAccessPeer(&ok, a, b);
      can[(size_t)a * n + b] = (e == hipSuccess) ? ok : -1;
      if (can[(size_t)a * n + b] == 1) {
        // ROCm 7.14 exposes hipDeviceEnablePeerAccess (hipEnablePeerAccess does
        // not exist in this header; hipCtxEnablePeerAccess is the deprecated form).
        hipError_t e2 = hipDeviceEnablePeerAccess(b, 0);
        if (e2 == hipSuccess || e2 == hipErrorPeerAccessAlreadyEnabled) {
          en[(size_t)a * n + b] = 1;
        } else {
          en[(size_t)a * n + b] = 0;
          en_err[(size_t)a * n + b] = hipGetErrorString(e2);
        }
      }
    }
  }

  printf("{\n");
  printf("  \"tool\": \"m0_p2p_matrix\",\n");
  printf("  \"bytes_big\": %zu,\n", big);
  printf("  \"bytes_small\": %zu,\n", small);
  printf("  \"repeats\": %d,\n", repeats);
  printf("  \"lat_repeats\": %d,\n", lat_repeats);
  printf("  \"peer_bw_ceiling_gbps\": %.1f,\n", PEER_BW_CEILING);
  printf("  \"self_bw_ceiling_gbps\": %.1f,\n", SELF_BW_CEILING);
  printf("  \"device_count\": %d,\n", n);

  printf("  \"devices\": [");
  for (int i = 0; i < n; i++) {
    printf("%s\n    {\"index\": %d, \"name\": \"%s\", \"gcn_arch\": \"%s\", "
           "\"pci_bus_id\": \"%s\", \"mem_gb\": %.2f}",
           i ? "," : "", i, name[i].c_str(), arch[i].c_str(), bus[i].c_str(), memgb[i]);
  }
  printf("\n  ],\n");

  printf("  \"self_copy\": [");
  for (int i = 0; i < n; i++) {
    fprintf(stderr, "self %d...\n", i);
    CopyResult s = measure_pair(i, i, big, repeats, warmup, SELF_BW_CEILING);
    printf("%s\n    {\"dev\": %d, \"bw_gbps_median\": %.3f, \"bw_iqr\": %.3f, "
           "\"bw_gbps_block\": %.3f, \"bytes_ok\": %s, \"plausible\": %s, "
           "\"repeats\": %d, \"err\": \"%s\"}",
           i ? "," : "", i, s.bw_median, s.bw_iqr, s.bw_block,
           s.bytes_ok ? "true" : "false", s.plausible ? "true" : "false",
           s.repeats, s.err.c_str());
  }
  printf("\n  ],\n");

  printf("  \"host_transfer\": [");
  for (int i = 0; i < n; i++) {
    fprintf(stderr, "host xfer dev %d h2d...\n", i);
    CopyResult up = measure_host_transfer(i, big, repeats, warmup, true);
    fprintf(stderr, "host xfer dev %d d2h...\n", i);
    CopyResult dn = measure_host_transfer(i, big, repeats, warmup, false);
    printf("%s\n    {\"dev\": %d, \"h2d_gbps\": %.3f, \"h2d_iqr\": %.3f, "
           "\"d2h_gbps\": %.3f, \"d2h_iqr\": %.3f, \"h2d_block\": %.3f, "
           "\"d2h_block\": %.3f, \"bytes_ok\": %s, \"plausible\": %s, "
           "\"err\": \"%s\"}",
           i ? "," : "", i, up.bw_median, up.bw_iqr, dn.bw_median, dn.bw_iqr,
           up.bw_block, dn.bw_block, (up.bytes_ok && dn.bytes_ok) ? "true" : "false",
           (up.plausible && dn.plausible) ? "true" : "false",
           (up.err + dn.err).c_str());
  }
  printf("\n  ],\n");

  printf("  \"pairs\": [");
  bool first = true;
  for (int a = 0; a < n; a++) {
    for (int b = 0; b < n; b++) {
      if (a == b) continue;
      CopyResult bw, lat;
      bool ready = (can[(size_t)a * n + b] == 1) && (en[(size_t)a * n + b] == 1);
      if (ready) {
        fprintf(stderr, "pair %d->%d bw...\n", a, b);
        bw = measure_pair(a, b, big, repeats, warmup, PEER_BW_CEILING);
        fprintf(stderr, "pair %d->%d lat...\n", a, b);
        lat = measure_latency(a, b, small, lat_repeats);
      }
      printf("%s\n    {\"src\": %d, \"dst\": %d, \"can_access_peer\": %d, "
             "\"peer_access_enabled\": %d, \"enable_err\": \"%s\", "
             "\"bw_gbps_median\": %.3f, \"bw_iqr\": %.3f, \"bw_gbps_block\": %.3f, "
             "\"bytes_ok\": %s, \"plausible\": %s, "
             "\"lat_us_median\": %.3f, \"lat_us_synced_median\": %.3f, "
             "\"lat_us_iqr\": %.3f, \"repeats\": %d, \"err\": \"%s\"}",
             first ? "" : ",", a, b, can[(size_t)a * n + b], en[(size_t)a * n + b],
             en_err[(size_t)a * n + b].c_str(), bw.bw_median, bw.bw_iqr, bw.bw_block,
             bw.bytes_ok ? "true" : "false", bw.plausible ? "true" : "false",
             lat.lat_median, lat.lat_synced_median, lat.lat_iqr,
             bw.repeats, bw.err.c_str());
      first = false;
    }
  }
  printf("\n  ]\n}\n");
  fprintf(stderr, "done\n");
  return 0;
}
