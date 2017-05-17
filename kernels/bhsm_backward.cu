extern "C"

__global__ void bhsm_backward(
  const float *wxy,
  const float *x,
  const float *w,
  const int   *ts,
  const int   *paths,
  const float *codes,
  const int   *begins,
  const float *gLoss,
  const int   n_in,
  const int   max_len,
  const int   n_ex,
  float *gx,
  float *gW
) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n_ex * max_len) {
    int idx = i / max_len;
    int offset = i - idx * max_len;
    int t = ts[idx];

    int begin = begins[t];
    int length = begins[t+1] - begin;

    if (offset < length) {
      int p = begin + offset;
      int node = paths[p];
      float g = -gLoss[0] * codes[p] / (1.0f + exp(wxy[i]));

      int w_start = n_in * node;
      int x_start = n_in * idx;
      for (int j = 0; j < n_in; ++j) {
        int w_i = w_start + j;
        int x_i = x_start + j;
        atomicAdd(gx + x_i, g * w[w_i]);
        atomicAdd(gW + w_i, g * x[x_i]);
      }
    }
  }
}
