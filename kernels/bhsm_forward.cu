extern "C"

__global__ void bhsm_forward(
  const float *x,
  const float *w,
  const int   *ts,
  const int   *paths,
  const float *codes,
  const int   *begins,
//  const int   *lens,
  const int   n_in,
  const int   max_len,
  const int   n_ex,
  float *ls,
  float *wxy
) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n_ex * max_len) {
    int idx = i / max_len;
    int offset = i - idx * max_len;
    int t = ts[idx];

    int begin = begins[t];
    int length = begins[t+1] - begin;
//    int length = lens[t];

    if (offset < length) {
      int p = begin + offset;
      int node = paths[p];

      float wx = 0;
      int w_start = n_in * node;
      int x_start = n_in * idx;
      for (int j = 0; j < n_in; ++j) {
        wx += (w[w_start + j] * x[x_start + j]);
      }
      wx *= codes[p];
      wxy[i] = wx;
      ls[i] = log(1 + exp(-wx));
    }
  }
}
