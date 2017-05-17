extern "C"

__global__ void cast_select_forward(
  const float * actors,
  const int   * begins,
  const int   * offsets,
  const float * out
) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  int offset = offsets[i];
  if (offset != -1) {
    int variable_idx = begins[i] + offset[i];

  }
  

}
