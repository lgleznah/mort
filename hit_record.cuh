#ifndef HIT_RECORD_H
#define HIT_RECORD_H

#include <cuda_runtime.h>

#include "ray.cuh"
//#include "materials.cuh"

class material;
struct hit_record {
	point3 p;
	vec3 normal;
	int mat_idx;
	int mat_type;
	float t;
	float u, v;
	bool front_face;

	__device__ inline void set_face_normal(const ray& r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

#endif // !HIT_RECORD_H
