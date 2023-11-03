#ifndef SPHERE_H
#define SPHERE_H

#include <math_constants.h>

#include "hit_record.cuh"
#include "vec3.cuh"
#include "hittable.cuh"
#include "materials.cuh"


#define SOME_THREAD_ONLY(whatevs) {if ((threadIdx.x < 100) && (threadIdx.y < 100) && (blockIdx.x < 100) && (blockIdx.y < 100)) {whatevs;}}

struct sphere {

	public:
		__host__ __device__ 
		sphere() {}

		__host__ __device__ 
		sphere(point3 cen, float r, int material_type, int material_idx) : center1(cen), radius(r), mat_type(material_type), mat_idx(material_idx), moves(false) {
			auto rvec = vec3(radius, radius, radius);
		}

		__host__ __device__ 
		sphere(point3 cen1, point3 cen2, float r, int material_type, int material_idx) : center1(cen1), radius(r), mat_type(material_type), mat_idx(material_idx) {
			moves = true;
			center_vec = cen2 - cen1;
			auto rvec = vec3(radius, radius, radius);
		}

		point3 center1;
		float radius;

		bool moves;
		vec3 center_vec;

		int mat_type;
		int mat_idx;

		__device__ 
		bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
			const vec3 oc = r.origin() - center(r.time());
			const auto a = r.direction().length_squared();
			const auto half_b = dot(oc, r.direction());
			const auto c = oc.length_squared() - radius * radius;

			const auto discriminant = half_b * half_b - a * c;
			if (discriminant < 0) return false;
			const auto sqrtd = sqrt(discriminant);

			// Find the nearest root that lies in the acceptable range.
			auto root = (-half_b - sqrtd) / a;
			if (root < t_min || t_max < root) {
				root = (-half_b + sqrtd) / a;
				if (root < t_min || t_max < root)
					return false;
			}

			rec.t = root;
			rec.p = r.at(rec.t);
			const vec3 outward_normal = (rec.p - center(r.time())) / radius;
			rec.set_face_normal(r, outward_normal);
			/*if (outward_normal.x() < -0.9 && outward_normal.y() > -0.1 && outward_normal.z() > -0.1) {
				float u, v;
				compute_uv(outward_normal, u, v);
				printf("(%f, %f, %f) -> (%f, %f)\n", outward_normal.x(), outward_normal.y(), outward_normal.z(), u, v);
			}*/
			compute_uv(outward_normal, rec.u, rec.v);
			rec.mat_type = mat_type;
			rec.mat_idx = mat_idx;

			return true;
		}
	
		__device__ 
		point3 center(float time) const {
			if (!moves) {
				return center1;
			}

			return center1 + time * center_vec;
		}

		// TODO: using CUDART_PI_F somehow causes u and v to get the SAME value. WAAAAAAAAAAT
		// Apparently, using the "f" suffix at the end of pi somehow causes this behaviour
		__device__
		void compute_uv(const point3& p, float& u, float& v) const {
			float theta = acos(-p.y());
			float phi = atan2(-p.z(), p.x()) + 3.141592565;

			u = phi / (2.0 * 3.141592565);
			v = theta / 3.141592565;
		}
};

#endif