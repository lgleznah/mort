#ifndef INTERVAL_H
#define INTERVAL_H

#include <cuda_runtime.h>

class interval {
	public:
		float imin, imax;

		__host__ __device__ interval() {}
		__host__ __device__ interval(float _min, float _max): imin(_min), imax(_max) {}
		__host__ __device__ interval(const interval& a, const interval& b) {
			imin = min(a.imin, b.imin);
			imax = max(a.imax, b.imax);
		}

		__host__ __device__ float size() const {
			return imax - imin;
		}

		__host__ __device__ interval expand(float delta) const {
			auto padding = delta / 2.0;
			return interval(imin - padding, imax + padding);
		}

		__host__ __device__ bool contains(float x) const {
			return imin <= x && x <= imax;
		}

		__host__ __device__ bool surrounds(float x) const {
			return imin < x && x < imax;
		}

		__host__ __device__ float clamp(float x) const {
			if (x < imin) return imin;
			if (x > imax) return imax;
			return x;
		}
};


interval operator+(const interval& ival, double displacement) {
	return interval(ival.imin + displacement, ival.imax + displacement);
}

interval operator+(double displacement, const interval& ival) {
	return ival + displacement;
}

#endif