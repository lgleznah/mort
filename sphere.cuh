#ifndef SPHERE_H
#define SPHERE_H

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
			rec.mat_type = mat_type;
			rec.mat_idx = mat_idx;

			return true;
		}
	
		__device__ point3 center(float time) const {
			if (!moves) {
				return center1;
			}

			return center1 + time * center_vec;
		}
};

#endif