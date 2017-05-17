extern "C"

__global__ void bhsm_forward2(
  const float *x,
  const float *w,
  const int   *ts,
  const int   *paths,
  const float *codes,
  const int   *begins,
  const int   *lens,
  const int   n_in,
  const int   max_len,
  const int   n_ex,
  float *ls,
  float *wxy
) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_ex) {
    int t = ts[i];
    int begin = begins[t];
    int length = lens[t];

    for (int offset = 0; offset < length; ++offset) {
        int p = begin + offset;
        int node = paths[p];
        float wx = 0;
        int w_start = n_in * node;
        int x_start = n_in * i;
        for (int j = 0; j < n_in; ++j) {
          wx += (w[w_start + j] * x[x_start + j]);
        }
        wx *= codes[p];
        int out_i = (i * max_len) + offset;
        wxy[out_i] = wx;
        ls[out_i] = log(1 + exp(-wx));
    }
  }
}
