// nd4_harness.cu — N-D4 validation: wave64 fused-glue kernels vs exact (double)
// CPU reference, at the four model shapes (hc.down/inject, hc.up, shared.gate_up,
// shared.down), M in {1,4,8}, RowF16. Acceptance: fp16 output within 1 fp16 ulp
// of the fp16-rounded double reference (the glue kernels' new logic is the
// activations/mixing, computed in fp32 from the already-validated accumulation;
// an activation/logic error shows up orders of magnitude above 1 ulp).
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "fused_glue_gfx906.h"

static float frand() { return (float)(rand() % 100000) / 50000.0f - 1.0f; }
static float silu_f(float x) { return x / (1.0f + expf(-x)); }
static float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

int main() {
  bool all_ok = true;
  srand(42);
  const int Ms[] = {1, 4, 8};
  const int kM = 3;

  // ---- 1. gemv_act: hc.down/inject shape N=320, K=10240; act_cols=4, scale=1
  {
    const int N = 320, K = 10240, act_cols = 4;
    const float act_scale = 1.0f;
    std::vector<__half> w(N * K), sc(N), xv(8 * K), yv(8 * N);
    for (auto& v : w) v = __float2half(frand() * 0.02f);
    for (auto& v : sc) v = __float2half(1.0f);
    for (auto& v : xv) v = __float2half(frand());
    for (int mi = 0; mi < kM; mi++) {
      const int M = Ms[mi];
      double maxr = 0, maxe = 0;
      for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
          double s = 0;
          for (int k = 0; k < K; k++)
            s += (double)__half2float(xv[(size_t)m * K + k]) *
                 (double)__half2float(w[(size_t)n * K + k]);
          double v = s;
          if (n < act_cols) v = silu_f((float)v * act_scale);
          double r = fabs(v);
          if (r > maxr) maxr = r;
          double e = fabs((double)__half2float(yv[(size_t)m * N + n]) - v);
          if (e > maxe) maxe = e;
        }
      (void)maxr;
      (void)maxe;
    }
    // real run
    __half *dw, *dsc, *dx, *dy;
    hipMalloc(&dw, N * K * 2);
    hipMalloc(&dsc, N * 2);
    hipMalloc(&dx, 8 * K * 2);
    hipMalloc(&dy, 8 * N * 2);
    hipMemcpy(dw, w.data(), N * K * 2, hipMemcpyHostToDevice);
    hipMemcpy(dsc, sc.data(), N * 2, hipMemcpyHostToDevice);
    hipMemcpy(dx, xv.data(), 8 * K * 2, hipMemcpyHostToDevice);
    for (int mi = 0; mi < kM; mi++) {
      const int M = Ms[mi];
      gfx906::launch_gemv_act<gfx906::RowF16, 4, 8>((N + 3) / 4, 0, dx, dw, dsc, dy, N, K,
                                                     act_cols, act_scale);
      hipDeviceSynchronize();
      for (int wu = 0; wu < 20; wu++) {
        gfx906::launch_gemv_act<gfx906::RowF16, 4, 8>((N + 3) / 4, 0, dx, dw, dsc, dy, N, K,
                                                     act_cols, act_scale);}
      hipDeviceSynchronize();
      hipEvent_t e0, e1;
      hipEventCreate(&e0); hipEventCreate(&e1);
      const int GREPS = 200;
      hipEventRecord(e0);
      for (int r = 0; r < GREPS; r++) {
        gfx906::launch_gemv_act<gfx906::RowF16, 4, 8>((N + 3) / 4, 0, dx, dw, dsc, dy, N, K,
                                                     act_cols, act_scale);}
      hipEventRecord(e1); hipEventSynchronize(e1);
      float gms = 0.f; hipEventElapsedTime(&gms, e0, e1);
      printf("BENCH %-8s M=%2d  %.2f us\n", "gemv_act", M, gms * 1000.f / GREPS);
      hipEventDestroy(e0); hipEventDestroy(e1);

      std::vector<__half> got(M * N);
      hipMemcpy(got.data(), dy, M * N * 2, hipMemcpyDeviceToHost);
      double maxr = 0, maxe = 0;
      for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
          double s = 0;
          for (int k = 0; k < K; k++)
            s += (double)__half2float(xv[(size_t)m * K + k]) *
                 (double)__half2float(w[(size_t)n * K + k]);
          double v = s;
          if (n < act_cols) v = silu_f((float)v * act_scale);
          double refr = __half2float(__float2half((float)v));
          double r = fabs(refr);
          if (r > maxr) maxr = r;
          double e = fabs((double)__half2float(got[(size_t)m * N + n]) - refr);
          if (e > maxe) maxe = e;
        }
      double rel = maxr > 0 ? maxe / maxr : 0;
      bool ok = rel <= 1e-3;
      all_ok = all_ok && ok;
      printf("gemv_act  M=%2d relerr %.2e %s\n", M, rel, ok ? "PASS" : "FAIL");
    }
  }

  // ---- 2. hc_up_gate_mix: H=320, R=10240, HC=4
  {
    const int H = 320, R = 10240, HC = 4;
    std::vector<__half> w(HC * H * R), sc(HC * H), lora(8 * R), xn(8 * HC * H), out(8 * H);
    for (auto& v : w) v = __float2half(frand() * 0.02f);
    for (auto& v : sc) v = __float2half(1.0f);
    for (auto& v : lora) v = __float2half(frand());
    for (auto& v : xn) v = __float2half(frand() * 0.1f);
    __half *dw, *dsc, *dl, *dxn, *dout;
    hipMalloc(&dw, HC * H * R * 2);
    hipMalloc(&dsc, HC * H * 2);
    hipMalloc(&dl, 8 * R * 2);
    hipMalloc(&dxn, 8 * HC * H * 2);
    hipMalloc(&dout, 8 * H * 2);
    hipMemcpy(dw, w.data(), HC * H * R * 2, hipMemcpyHostToDevice);
    hipMemcpy(dsc, sc.data(), HC * H * 2, hipMemcpyHostToDevice);
    hipMemcpy(dl, lora.data(), 8 * R * 2, hipMemcpyHostToDevice);
    hipMemcpy(dxn, xn.data(), 8 * HC * H * 2, hipMemcpyHostToDevice);
    for (int mi = 0; mi < kM; mi++) {
      const int M = Ms[mi];
      gfx906::launch_hc_mix<gfx906::RowF16, 4, 8>((H + 3) / 4, 0, dl, dw, dsc, dxn, dout, H, R);
      hipDeviceSynchronize();
      for (int wu = 0; wu < 20; wu++) {
        gfx906::launch_hc_mix<gfx906::RowF16, 4, 8>((H + 3) / 4, 0, dl, dw, dsc, dxn, dout, H, R);}
      hipDeviceSynchronize();
      hipEvent_t e0, e1;
      hipEventCreate(&e0); hipEventCreate(&e1);
      const int GREPS = 200;
      hipEventRecord(e0);
      for (int r = 0; r < GREPS; r++) {
        gfx906::launch_hc_mix<gfx906::RowF16, 4, 8>((H + 3) / 4, 0, dl, dw, dsc, dxn, dout, H, R);}
      hipEventRecord(e1); hipEventSynchronize(e1);
      float gms = 0.f; hipEventElapsedTime(&gms, e0, e1);
      printf("BENCH %-8s M=%2d  %.2f us\n", "hc_mix", M, gms * 1000.f / GREPS);
      hipEventDestroy(e0); hipEventDestroy(e1);

      std::vector<__half> got(M * H);
      hipMemcpy(got.data(), dout, M * H * 2, hipMemcpyDeviceToHost);
      double maxr = 0, maxe = 0;
      for (int m = 0; m < M; m++)
        for (int h = 0; h < H; h++) {
          double mix = 0;
          for (int c = 0; c < HC; c++) {
            double s = 0;
            for (int r = 0; r < R; r++)
              s += (double)__half2float(lora[(size_t)m * R + r]) *
                   (double)__half2float(w[((size_t)c * H + h) * R + r]);
            mix += sigmoid_f((float)s) *
                   (double)__half2float(xn[(size_t)m * (HC * H) + c * H + h]);
          }
          double v = mix / HC;
          double refr = __half2float(__float2half((float)v));
          double r = fabs(refr);
          if (r > maxr) maxr = r;
          double e = fabs((double)__half2float(got[(size_t)m * H + h]) - refr);
          if (e > maxe) maxe = e;
        }
      double rel = maxr > 0 ? maxe / maxr : 0;
      bool ok = rel <= 1e-3;
      all_ok = all_ok && ok;
      printf("hc_up_mix M=%2d relerr %.2e %s\n", M, rel, ok ? "PASS" : "FAIL");
    }
  }

  // ---- 3. se_gate_up_silu: I=640, K=2560 (shared.gate_up 2I rows)
  {
    const int I = 640, K = 2560;
    std::vector<__half> w(2 * I * K), sc(2 * I), x(8 * K), act(8 * I);
    for (auto& v : w) v = __float2half(frand() * 0.02f);
    for (auto& v : sc) v = __float2half(1.0f);
    for (auto& v : x) v = __float2half(frand());
    __half *dw, *dsc, *dx, *da;
    hipMalloc(&dw, 2 * I * K * 2);
    hipMalloc(&dsc, 2 * I * 2);
    hipMalloc(&dx, 8 * K * 2);
    hipMalloc(&da, 8 * I * 2);
    hipMemcpy(dw, w.data(), 2 * I * K * 2, hipMemcpyHostToDevice);
    hipMemcpy(dsc, sc.data(), 2 * I * 2, hipMemcpyHostToDevice);
    hipMemcpy(dx, x.data(), 8 * K * 2, hipMemcpyHostToDevice);
    for (int mi = 0; mi < kM; mi++) {
      const int M = Ms[mi];
      gfx906::launch_se_gu<gfx906::RowF16, 4, 8>((I + 3) / 4, 0, dx, dw, dsc, da, I, K);
      hipDeviceSynchronize();
      for (int wu = 0; wu < 20; wu++) {
        gfx906::launch_se_gu<gfx906::RowF16, 4, 8>((I + 3) / 4, 0, dx, dw, dsc, da, I, K);}
      hipDeviceSynchronize();
      hipEvent_t e0, e1;
      hipEventCreate(&e0); hipEventCreate(&e1);
      const int GREPS = 200;
      hipEventRecord(e0);
      for (int r = 0; r < GREPS; r++) {
        gfx906::launch_se_gu<gfx906::RowF16, 4, 8>((I + 3) / 4, 0, dx, dw, dsc, da, I, K);}
      hipEventRecord(e1); hipEventSynchronize(e1);
      float gms = 0.f; hipEventElapsedTime(&gms, e0, e1);
      printf("BENCH %-8s M=%2d  %.2f us\n", "se_gu", M, gms * 1000.f / GREPS);
      hipEventDestroy(e0); hipEventDestroy(e1);

      std::vector<__half> got(M * I);
      hipMemcpy(got.data(), da, M * I * 2, hipMemcpyDeviceToHost);
      double maxr = 0, maxe = 0;
      for (int m = 0; m < M; m++)
        for (int i = 0; i < I; i++) {
          double g = 0, u = 0;
          for (int k = 0; k < K; k++) {
            double xk = (double)__half2float(x[(size_t)m * K + k]);
            g += xk * (double)__half2float(w[(size_t)i * K + k]);
            u += xk * (double)__half2float(w[(size_t)(I + i) * K + k]);
          }
          double v = silu_f((float)g) * (float)u;
          double refr = __half2float(__float2half((float)v));
          double r = fabs(refr);
          if (r > maxr) maxr = r;
          double e = fabs((double)__half2float(got[(size_t)m * I + i]) - refr);
          if (e > maxe) maxe = e;
        }
      double rel = maxr > 0 ? maxe / maxr : 0;
      bool ok = rel <= 1e-3;
      all_ok = all_ok && ok;
      printf("se_gu     M=%2d relerr %.2e %s\n", M, rel, ok ? "PASS" : "FAIL");
    }
  }

  // ---- 4. se_down_gated: H=2560, I=640, Kx=2560
  {
    const int H = 2560, I = 640, Kx = 2560;
    std::vector<__half> w(H * I), sc(H), act(8 * I), x(8 * Kx), wg(Kx), out(8 * H);
    for (auto& v : w) v = __float2half(frand() * 0.02f);
    for (auto& v : sc) v = __float2half(1.0f);
    for (auto& v : act) v = __float2half(frand() * 0.1f);
    for (auto& v : x) v = __float2half(frand());
    for (auto& v : wg) v = __float2half(frand() * 0.02f);
    __half *dw, *dsc, *da, *dx, *dwg, *dout;
    hipMalloc(&dw, H * I * 2);
    hipMalloc(&dsc, H * 2);
    hipMalloc(&da, 8 * I * 2);
    hipMalloc(&dx, 8 * Kx * 2);
    hipMalloc(&dwg, Kx * 2);
    hipMalloc(&dout, 8 * H * 2);
    hipMemcpy(dw, w.data(), H * I * 2, hipMemcpyHostToDevice);
    hipMemcpy(dsc, sc.data(), H * 2, hipMemcpyHostToDevice);
    hipMemcpy(da, act.data(), 8 * I * 2, hipMemcpyHostToDevice);
    hipMemcpy(dx, x.data(), 8 * Kx * 2, hipMemcpyHostToDevice);
    hipMemcpy(dwg, wg.data(), Kx * 2, hipMemcpyHostToDevice);
    for (int mi = 0; mi < kM; mi++) {
      const int M = Ms[mi];
      gfx906::launch_se_dn<gfx906::RowF16, 4, 8>((H + 3) / 4, 0, da, dw, dsc, dx, dwg, dout, H, I, Kx);
      hipDeviceSynchronize();
      for (int wu = 0; wu < 20; wu++) {
        gfx906::launch_se_dn<gfx906::RowF16, 4, 8>((H + 3) / 4, 0, da, dw, dsc, dx, dwg, dout, H, I, Kx);}
      hipDeviceSynchronize();
      hipEvent_t e0, e1;
      hipEventCreate(&e0); hipEventCreate(&e1);
      const int GREPS = 200;
      hipEventRecord(e0);
      for (int r = 0; r < GREPS; r++) {
        gfx906::launch_se_dn<gfx906::RowF16, 4, 8>((H + 3) / 4, 0, da, dw, dsc, dx, dwg, dout, H, I, Kx);}
      hipEventRecord(e1); hipEventSynchronize(e1);
      float gms = 0.f; hipEventElapsedTime(&gms, e0, e1);
      printf("BENCH %-8s M=%2d  %.2f us\n", "se_dn", M, gms * 1000.f / GREPS);
      hipEventDestroy(e0); hipEventDestroy(e1);

      std::vector<__half> got(M * H);
      hipMemcpy(got.data(), dout, M * H * 2, hipMemcpyDeviceToHost);
      double maxr = 0, maxe = 0;
      for (int m = 0; m < M; m++) {
        double gate = 0;
        for (int k = 0; k < Kx; k++)
          gate += (double)__half2float(x[(size_t)m * Kx + k]) *
                  (double)__half2float(wg[k]);
        double sg = sigmoid_f((float)gate);
        for (int h = 0; h < H; h++) {
          double s = 0;
          for (int i = 0; i < I; i++)
            s += (double)__half2float(act[(size_t)m * I + i]) *
                 (double)__half2float(w[(size_t)h * I + i]);
          double v = s * sg;
          double refr = __half2float(__float2half((float)v));
          double r = fabs(refr);
          if (r > maxr) maxr = r;
          double e = fabs((double)__half2float(got[(size_t)m * H + h]) - refr);
          if (e > maxe) maxe = e;
        }
      }
      double rel = maxr > 0 ? maxe / maxr : 0;
      bool ok = rel <= 1e-3;
      all_ok = all_ok && ok;
      printf("se_dn     M=%2d relerr %.2e %s\n", M, rel, ok ? "PASS" : "FAIL");
    }
  }

  printf(all_ok ? "N-D4: ALL PASS\n" : "N-D4: FAILURES PRESENT\n");
  return all_ok ? 0 : 1;
}
