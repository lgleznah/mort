#ifndef ONB_CUH
#define ONB_CUH

#include "vec3.cuh"

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

class onb {
	public:
		__host__ __device__ 
		onb() {}

		__host__ __device__
		vec3 operator[](int i) const { return axis[i]; }

		__host__ __device__
		vec3& operator[](int i) { return axis[i]; }

		__host__ __device__
		vec3 u() const { return axis[0]; }

		__host__ __device__
		vec3 v() const { return axis[1]; }

		__host__ __device__
		vec3 w() const { return axis[2]; }

		__host__ __device__
		vec3 local(float a, float b, float c) const {
			return a*u() + b*v() + c*w();
		}

		__host__ __device__
		vec3 local(const vec3& a) const {
			return a.x()*u() + a.y()*v() + a.z()*w();
		}

		__host__ __device__
		void build_from_w(const vec3& w) {
			vec3 unit_w = unit_vector(w);
			vec3 a = (fabsf(unit_w.x()) > 0.9) ? vec3(0, 1, 0) : vec3(1, 0, 0);
			vec3 v = unit_vector(cross(unit_w, a));
			vec3 u = cross(unit_w, v);

			axis[0] = u;
			axis[1] = v;
			axis[2] = unit_w;
		}

	public:
		vec3 axis[3];
};

#endif