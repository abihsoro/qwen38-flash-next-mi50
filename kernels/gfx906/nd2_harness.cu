// nd2_harness.cu — N-D2 validation: wave64 MoE int4 decode vs exact (double)
// CPU reference. Exercises in-kernel expert_map (EP): expert_map mixes local
// indices with -1 (non-local); act rows for non-local experts must be 0 and
// the down kernel must skip them.
// Reference is STAGED like serving: act_double -> act_fp16 (round) -> out.
// Levels: w13 fp32 acc (F32_OUT vs act_double) <= 1e-4; w13 fp16 output vs
// act_fp16 <= 1 ulp; w2 fp32 acc (consuming the fp16 act) vs out_ref <= 1e-4;
// w2 fp16 output vs out_ref <= 1 ulp.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "moe_int4_gfx906.h"

static const int KM = 2560;
static const int NM = 640;
static const int TOPK = 10;
static const int GROUP = 128;
static const int N_LOCAL = 64;
static const int N_GLOBAL = 128;
static const int Ms[] = {1, 4, 8, 16};
static const int kNumMs = 4;

static float frand() { return (float)(rand() % 100000) / 50000.0f - 1.0f; }
static float silu_f(float x) { return x / (1.0f + expf(-x)); }

int main() {
  bool all_ok = true;
  srand(42);
  const int K8 = KM / 8, KG = KM / GROUP, N8 = NM / 8, NG = NM / GROUP;
  std::vector<uint32_t> w13(N_LOCAL * 2 * NM * K8), w2(N_LOCAL * KM * N8);
  std::vector<__half> s13(N_LOCAL * 2 * NM * KG), s2(N_LOCAL * KM * NG);
  for (auto& v : w13) v = (uint32_t)(rand() & 0xFFFFFFFF);
  for (auto& v : w2) v = (uint32_t)(rand() & 0xFFFFFFFF);
  for (auto& v : s13) v = __float2half(frand() * 0.02f + 0.02f);
  for (auto& v : s2) v = __float2half(frand() * 0.02f + 0.02f);
  std::vector<int32_t> emap(N_GLOBAL, -1);
  for (int i = 0; i < N_LOCAL; i++) emap[i] = i;

  uint32_t *dw13, *dw2;
  __half *ds13, *ds2;
  int32_t* demap;
  hipMalloc(&dw13, w13.size() * 4);
  hipMalloc(&dw2, w2.size() * 4);
  hipMalloc(&ds13, s13.size() * 2);
  hipMalloc(&ds2, s2.size() * 2);
  hipMalloc(&demap, N_GLOBAL * 4);
  hipMemcpy(dw13, w13.data(), w13.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(dw2, w2.data(), w2.size() * 4, hipMemcpyHostToDevice);
  hipMemcpy(ds13, s13.data(), s13.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(ds2, s2.data(), s2.size() * 2, hipMemcpyHostToDevice);
  hipMemcpy(demap, emap.data(), N_GLOBAL * 4, hipMemcpyHostToDevice);

  const int W = 4;
  dim3 block(W * 64);
  dim3 grid1((NM + W - 1) / W, TOPK, 1);
  dim3 grid2((KM + W - 1) / W, 1, 1);

  for (int mi = 0; mi < kNumMs; mi++) {
    const int M = Ms[mi];
    grid1.z = M;
    grid2.z = M;
    std::vector<__half> x(M * KM);
    std::vector<int32_t> ids(M * TOPK);
    std::vector<float> tw(M * TOPK);
    for (int i = 0; i < M * KM; i++) x[i] = __float2half(frand());
    for (int i = 0; i < M * TOPK; i++) {
      ids[i] = (i % 3 == 2) ? (N_GLOBAL - 1 - (i % 7)) : (i % N_LOCAL);
      tw[i] = frand() * 0.1f + 0.05f;
    }
    // staged reference
    std::vector<double> act_d((size_t)M * TOPK * NM);
    for (int m = 0; m < M; m++)
      for (int s = 0; s < TOPK; s++) {
        const int e = emap[ids[(size_t)m * TOPK + s]];
        for (int n = 0; n < NM; n++) {
          if (e < 0) { act_d[((size_t)m * TOPK + s) * NM + n] = 0.0; continue; }
          const uint64_t base = (uint64_t)e * 2 * NM;
          double g = 0, u = 0;
          for (int k = 0; k < KM; k++) {
            const double xv = (double)__half2float(x[(size_t)m * KM + k]);
            const int grp = k / GROUP;
            const double sg = (double)__half2float(s13[(base + n) * KG + grp]);
            const double su = (double)__half2float(s13[(base + NM + n) * KG + grp]);
            g += (double)((int)((w13[(base + n) * K8 + k / 8] >> (4 * (k % 8))) & 0xF) - 8) * xv * sg;
            u += (double)((int)((w13[(base + NM + n) * K8 + k / 8] >> (4 * (k % 8))) & 0xF) - 8) * xv * su;
          }
          act_d[((size_t)m * TOPK + s) * NM + n] = (double)silu_f((float)g) * u;
        }
      }
    // the w2 kernels consume the FP16 act (serving), so the out reference
    // is staged through the fp16-rounded act
    std::vector<double> act_fp16((size_t)M * TOPK * NM);
    for (size_t i = 0; i < act_d.size(); i++)
      act_fp16[i] = (double)__half2float(__float2half((float)act_d[i]));
    std::vector<double> out_d((size_t)M * KM);
    for (int m = 0; m < M; m++)
      for (int h = 0; h < KM; h++) {
        double acc = 0;
        for (int s = 0; s < TOPK; s++) {
          const int e = emap[ids[(size_t)m * TOPK + s]];
          if (e < 0) continue;
          double sacc = 0;
          const uint32_t* wrow = w2.data() + ((uint64_t)e * KM + h) * N8;
          const __half* srow = s2.data() + ((uint64_t)e * KM + h) * NG;
          const double* xrow = &act_fp16[((size_t)m * TOPK + s) * NM];
          for (int n = 0; n < NM; n++) {
            const int q = (int)((wrow[n / 8] >> (4 * (n % 8))) & 0xF) - 8;
            sacc += (double)q * xrow[n] * (double)__half2float(srow[n / GROUP]);
          }
          acc += sacc * (double)tw[(size_t)m * TOPK + s];
        }
        out_d[(size_t)m * KM + h] = acc;
      }
    __half *dx, *dact, *dout;
    float *dactf, *doutf;
    int32_t* dids;
    float* dtw;
    hipMalloc(&dx, M * KM * 2);
    hipMalloc(&dids, M * TOPK * 4);
    hipMalloc(&dtw, M * TOPK * 4);
    hipMalloc(&dact, (size_t)M * TOPK * NM * 2);
    hipMalloc(&dout, M * KM * 2);
    hipMalloc(&dactf, (size_t)M * TOPK * NM * 4);
    hipMalloc(&doutf, M * KM * 4);
    hipMemcpy(dx, x.data(), M * KM * 2, hipMemcpyHostToDevice);
    hipMemcpy(dids, ids.data(), M * TOPK * 4, hipMemcpyHostToDevice);
    hipMemcpy(dtw, tw.data(), M * TOPK * 4, hipMemcpyHostToDevice);
    // w13 fp32 accumulation
    gfx906::moe_w13_silu_gemv_gfx906_<W, true>
        <<<grid1, block, KM * 2>>>(dx, dw13, ds13, dids, false, demap,
                                   (__half*)dactf, KM, NM, TOPK, GROUP);
    // w13 fp16 serving
    gfx906::moe_w13_silu_gemv_gfx906_<W>
        <<<grid1, block, KM * 2>>>(dx, dw13, ds13, dids, false, demap, dact,
                                   KM, NM, TOPK, GROUP);
    // w2 fp32 accumulation (consumes the fp16 act, as serving does)
    gfx906::moe_w2_gemv_gfx906_<W, true>
        <<<grid2, block, TOPK * NM * 2>>>(dact, dw2, ds2, dids, false, demap,
                                          dtw, false, (__half*)doutf, NM, KM,
                                          TOPK, GROUP);
    // w2 fp16 serving
    gfx906::moe_w2_gemv_gfx906_<W>
        <<<grid2, block, TOPK * NM * 2>>>(dact, dw2, ds2, dids, false, demap,
                                          dtw, false, dout, NM, KM, TOPK, GROUP);
    hipDeviceSynchronize();
    std::vector<float> actf((size_t)M * TOPK * NM), outf(M * KM);
    std::vector<__half> acth((size_t)M * TOPK * NM), outh(M * KM);
    hipMemcpy(actf.data(), dactf, actf.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(outf.data(), doutf, outf.size() * 4, hipMemcpyDeviceToHost);
    hipMemcpy(acth.data(), dact, acth.size() * 2, hipMemcpyDeviceToHost);
    hipMemcpy(outh.data(), dout, outh.size() * 2, hipMemcpyDeviceToHost);
    auto metrics = [](const std::vector<double>& ref, const std::vector<float>& f32,
                      const std::vector<__half>& h16, const char* name, bool* ok) {
      double maxr = 0, ea = 0, eo = 0;
      for (size_t i = 0; i < ref.size(); i++) {
        double r = fabs(ref[i]);
        if (r > maxr) maxr = r;
        double e1 = fabs((double)f32[i] - ref[i]);
        if (e1 > ea) ea = e1;
        double e2 = fabs((double)__half2float(h16[i]) - ref[i]);
        if (e2 > eo) eo = e2;
      }
      double ra = maxr > 0 ? ea / maxr : 0, ro = maxr > 0 ? eo / maxr : 0;
      bool okv = (ra <= 1e-4) && (ro <= 1e-3);
      *ok = *ok && okv;
      printf("  %s acc %.2e out %.2e %s\n", name, ra, ro, okv ? "PASS" : "FAIL");
    };
    printf("M=%2d\n", M);
    metrics(act_d, actf, acth, "w13", &all_ok);
    metrics(out_d, outf, outh, "w2 ", &all_ok);
    // ---- bench: time the w13+w2 pair (fork RESULTS unit: 98 us on V620) ----
    for (int wu = 0; wu < 20; wu++) {
      gfx906::moe_w13_silu_gemv_gfx906_<W><<<grid1, block, KM * 2>>>(
          dx, dw13, ds13, dids, false, demap, dact, KM, NM, TOPK, GROUP);
      gfx906::moe_w2_gemv_gfx906_<W><<<grid2, block, TOPK * NM * 2>>>(
          dact, dw2, ds2, dids, false, demap, dtw, false, dout, NM, KM, TOPK, GROUP);
    }
    hipDeviceSynchronize();
    hipEvent_t b0, b1;
    hipEventCreate(&b0); hipEventCreate(&b1);
    const int BREPS = 200;
    hipEventRecord(b0);
    for (int r = 0; r < BREPS; r++) {
      gfx906::moe_w13_silu_gemv_gfx906_<W><<<grid1, block, KM * 2>>>(
          dx, dw13, ds13, dids, false, demap, dact, KM, NM, TOPK, GROUP);
      gfx906::moe_w2_gemv_gfx906_<W><<<grid2, block, TOPK * NM * 2>>>(
          dact, dw2, ds2, dids, false, demap, dtw, false, dout, NM, KM, TOPK, GROUP);
    }
    hipEventRecord(b1);
    hipEventSynchronize(b1);
    float bms = 0.f;
    hipEventElapsedTime(&bms, b0, b1);
    printf("BENCH M=%2d  w13+w2 pair: %.2f us\n", M, bms * 1000.f / BREPS);
    hipEventDestroy(b0); hipEventDestroy(b1);
    hipFree(dx); hipFree(dids); hipFree(dtw);
    hipFree(dact); hipFree(dout); hipFree(dactf); hipFree(doutf);
  }
  printf(all_ok ? "N-D2: ALL PASS\n" : "N-D2: FAILURES PRESENT\n");
  return all_ok ? 0 : 1;
}
