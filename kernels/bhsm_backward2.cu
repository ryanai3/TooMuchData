extern "C"

__global__ void bhsm_backward2(
  const float *wxy,
  const float *x,
  const float *w,
  const int   *ts,
  const int   *paths,
  const float *codes,
  const int   *begins,
  const int   *lens,
  const float *gLoss,
  const int   n_in,
  const int   max_len,
  const int   n_ex,
  float *gx,
  float *gW
) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < n_ex * max_len) {
    int idx = i / max_len;
    int offset = i - idx * max_len;
    int t = ts[idx];

    int begin = begins[t];
    int length = lens[t];

    if (offset < length) {
      int p = begin + offset;
      int node = paths[p];

      if (y < n_in) {
        float g = -gLoss[0] * codes[p] / (1.0f + exp(wxy[i]));
        int w_i = (n_in * node) + y;
        int x_i = (n_in * idx)  + y;
        atomicAdd(gx + x_i, g * w[w_i]);
        atomicAdd(gW + w_i, g * x[x_i]);
      }
    }
  }
}
