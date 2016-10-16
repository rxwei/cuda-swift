__global__
void gIncr(float *d, size_t ind, float delta) {
  d[ind] += delta;
}

__global__
void gSum(float *d, size_t size, float *total) {
  total = 0;
  for (size_t i = 0; i < size; ++i) {
    *total += d[i];
  }
}
