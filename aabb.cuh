#ifndef AABB_CUH
#define AABB_CUH

#include "interval.cuh"
#include "ray.cuh"

struct aabb {
	interval x, y, z;

	__host__
	aabb() {}

	__host__
	aabb(const interval& ix, const interval& iy, const interval& iz): x(ix), y(iy), z(iz) {}

	__host__
	aabb(const point3& a, const point3& b) {
		x = interval(fmin(a[0], b[0]), fmax(a[0], b[0]));
		y = interval(fmin(a[1], b[1]), fmax(a[1], b[1]));
		z = interval(fmin(a[2], b[2]), fmax(a[2], b[2]));
	}

	__host__
	aabb(const aabb& box0, const aabb& box1) {
		x = interval(box0.x, box1.x);
		y = interval(box0.y, box1.y);
		z = interval(box0.z, box1.z);
	}

	__host__ __device__
	const interval& axis(int n) const {
		if (n == 1) return y;
		if (n == 2) return z;
		return x;
	}

	__device__
	bool hit(const ray& r, float t_min, float t_max) const {
		for (int a = 0; a < 3; a++) {
			auto invD = 1.0 / r.direction()[a];
			auto orig = r.origin()[a];

			auto t0 = (axis(a).imin - orig) * invD;
			auto t1 = (axis(a).imax - orig) * invD;

			if (invD < 0) {
				auto aux = t0;
				t0 = t1;
				t1 = aux;
			}

			if (t0 > t_min) t_min = t0;
			if (t1 < t_max) t_max = t1;

			if (t_max <= t_min)
				return false;
		}
		return true;
	}
};

__host__
aabb operator+(const aabb& bbox, const vec3& offset) {
	return aabb(bbox.x + offset.x(), bbox.y + offset.y(), bbox.z + offset.z());
}

__host__
aabb operator+(const vec3& offset, const aabb& bbox) {
	return bbox + offset;
}

#endif