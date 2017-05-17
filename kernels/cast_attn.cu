extern "C"

__global__ void attend_forward(
  const int   n_actors,
  const float *actor_pool,
  const int   *actor_idxs,
  const int   batch_size,
  const int   *cast_sizes,
  const int   vec_size,
  const float *query,
  float       *scores
) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n_actors) {
    int idx = actor_idxs[i]
  }


  if (batch_i < n_actors) {
    if (cast_i < cast_sizes[batch_i]) {
      for (int i = 0; i < vec_size; ++i) {
        int idx = begins[batch_i] + cast_i;
        scores[idx] += actors[(idx * vec_size) + i] * query[batch_i + i];
      }   
    }
  }
}
