#ifndef RNG_H
#define RNG_H

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ 
void setup_rng(curandState* states, unsigned long seed, int image_width) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * image_width;

	curand_init(seed, offset, 0, &states[offset]);
}

__device__ 
float random_float(curandState* states, int idx) {
	curandState localState = states[idx];
	float random = 1.0 - curand_uniform(&localState);
	states[idx] = localState;
	return random;
}

__device__ 
float random_float(curandState* states, int idx, float min, float max) {
	float base_rng = random_float(states, idx);
	return base_rng * (max - min) + min;
}

__host__ 
float random_float() {
	return rand() / (RAND_MAX + 1.0);
}

__host__ 
inline float random_float(float min, float max) {
	// Returns a random real in [min,max).
	return min + (max - min) * random_float();
}

__host__ 
int random_int(int max) {
	return rand() % max;
}

#endif // ! RNG_H
