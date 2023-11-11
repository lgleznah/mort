#ifndef PERLIN_CUH
#define PERLIN_CUH

#include "rng.cuh"
#include "vec3.cuh"

#define POINT_COUNT 256

struct perlin {
	public:
		perlin() {}
		perlin(char dummy) {
			for (int i = 0; i < POINT_COUNT; ++i) {
				ranfloat[i] = random_float();
			}

			perlin_generate_perm(perm_x);
			perlin_generate_perm(perm_y);
			perlin_generate_perm(perm_z);
		}

		__device__
		float noise(const point3& p) const {
			auto i = (int)(4 * p.x()) & 255;
			auto j = (int)(4 * p.y()) & 255;
			auto k = (int)(4 * p.z()) & 255;

			return ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
		}


	private:
		float ranfloat[POINT_COUNT];
		int perm_x[POINT_COUNT];
		int perm_y[POINT_COUNT];
		int perm_z[POINT_COUNT];

		static int* perlin_generate_perm(int* arr) {
			for (int i = 0; i < POINT_COUNT; i++) {
				arr[i] = i;
			}
			permute(arr, POINT_COUNT);
		}

		static void permute(int* arr, int n) {
			for (int i = n - 1; i > 0; i--) {
				int target = (int)random_float(0.0, i);
				int tmp = arr[i];
				arr[i] = arr[target];
				arr[target] = tmp;
			}
		}
};

#endif