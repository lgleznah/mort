#ifndef AABB_H
#define AABB_H

#include "vec3.cuh"
#include "interval.cuh"
#include "ray.cuh"

#include <cuda_runtime.h>

class aabb {
	public:
		interval x, y, z;

		__host__ __device__ aabb() {}

		__host__ __device__ aabb(const interval& ix, const interval& iy, const interval& iz): x(ix), y(iy), z(iz) {}

		__host__ __device__ aabb(const aabb& box0, const aabb& box1) {
			x = interval(box0.x, box1.x);
			y = interval(box0.y, box1.y);
			z = interval(box0.z, box1.z);
		}
		
		__host__ __device__ aabb(const point3& a, const point3& b) {
			x = interval(min(a[0], b[0]), max(a[0], b[0]));
			y = interval(min(a[1], b[1]), max(a[1], b[1]));
			z = interval(min(a[2], b[2]), max(a[2], b[2]));
		}

		__host__ __device__ const interval& axis(int n) const {
			if (n == 1) return y;
			if (n == 2) return z;
			return x;
		}

		__host__ __device__ bool hit(const ray& r, interval ray_t) const {
			for (int a = 0; a < 3; a++) {
				auto invD = 1 / r.direction()[a];
				auto orig = r.origin()[a];

				auto t0 = (axis(a).imin - orig) * invD;
				auto t1 = (axis(a).imax - orig) * invD;

				if (invD < 0)
					std::swap(t0, t1);

				if (t0 > ray_t.imin) ray_t.imin = t0;
				if (t1 < ray_t.imax) ray_t.imax = t1;

				if (ray_t.imax <= ray_t.imin)
					return false;
			}
			return true;
		}
};

#endif