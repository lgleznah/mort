#ifndef MATERIALS_H
#define MATERIALS_H

#include <curand.h>
#include <curand_kernel.h>

#include "vec3.cuh"
#include "hit_record.cuh"
#include "ray.cuh"
#include "textures.cuh"

#define MAT_LAMBERTIAN 1
#define MAT_METAL 2
#define MAT_DIELECTRIC 3
#define MAT_DIFFUSE_LIGHT 4

struct lambertian {
	public:
		int texType, texIdx;

		int idx;
		static int global_idx;

		__host__ lambertian() {}
		__host__ lambertian(int _texType, int _texIdx): texType(_texType), texIdx(_texIdx) { idx = global_idx++; }

		__device__ 
		bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* states, int idx) const {
			auto scatter_direction = rec.normal + random_in_unit_sphere(states, idx);
			scattered = ray(rec.p, scatter_direction, r_in.time());
			attenuation = valueDispatch(texType, texIdx, rec.u, rec.v, rec.p);
			return true;
		}

		__device__
		color emitted(float u, float v, const point3& p) const {
			return color(0.0, 0.0, 0.0);
		}

		int getType() const { return MAT_LAMBERTIAN; }
		int getIdx() const { return idx; }
};


struct metal {
	public:
		color albedo;
		float fuzz;

		int idx;
		static int global_idx;

		__host__ metal() {}
		__host__ metal(const color& a, float f): albedo(a), fuzz(f) { idx = global_idx++; }

		__device__ 
		bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* states, int idx) const {
			vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
			reflected += random_in_unit_sphere(states, idx) * fuzz;
			scattered = ray(rec.p, reflected, r_in.time());
			attenuation = albedo;
			return dot(scattered.direction(), rec.normal) > 0;
		}

		__device__
		color emitted(float u, float v, const point3& p) const {
			return color(0.0, 0.0, 0.0);
		}

		int getType() const { return MAT_METAL; }
		int getIdx() const { return idx; }
};

struct dielectric {
	public:
		float ior, inv_ior;
		color albedo;

		int idx;
		static int global_idx;

		__host__ dielectric() {}
		__host__ dielectric(float refraction_index, const color& a) : ior(refraction_index), inv_ior(1.0 / refraction_index), albedo(a) { idx = global_idx++; }
		__host__ dielectric(float refraction_index): ior(refraction_index), inv_ior(1.0 / refraction_index), albedo(color(1.0,1.0,1.0)) { idx = global_idx++; }

		__device__ 
		bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* states, int idx) const {
			attenuation = albedo;
			float refraction_ratio = rec.front_face ? inv_ior : ior;

			vec3 unit_direction = unit_vector(r_in.direction());
			float cos_theta = min(dot(-unit_direction, rec.normal), 1.0);
			float sin_theta = sqrt(1.0 - cos_theta * cos_theta);

			bool cant_refract = (refraction_ratio * sin_theta) > 1.0;
			vec3 direction;

			if (cant_refract || reflectance(cos_theta, refraction_ratio) > random_float(states, idx)) {
				direction = reflect(unit_direction, rec.normal);
			}
			else {
				direction = refract(unit_direction, rec.normal, refraction_ratio);
			}

			scattered = ray(rec.p, direction, r_in.time());
			return true;
		}

		__device__
		color emitted(float u, float v, const point3& p) const {
			return color(0.0, 0.0, 0.0);
		}

		int getType() const { return MAT_DIELECTRIC; }
		int getIdx() const { return idx; }
};

struct diffuse_light {
	public:
		int texType, texIdx;

		int idx;
		static int global_idx;

		__host__ diffuse_light() {}
		__host__ diffuse_light(int _texType, int _texIdx): texType(_texType), texIdx(_texIdx) { idx = global_idx++; }

		__device__
		bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* states, int idx) const {
			return false;
		}

		__device__
		color emitted(float u, float v, const point3& p) const {
			return valueDispatch(texType, texIdx, u, v, p);
		}

		int getType() const { return MAT_DIFFUSE_LIGHT; }
		int getIdx() const { return idx; }
};

int lambertian::global_idx = 0;
int metal::global_idx = 0;
int dielectric::global_idx = 0;
int diffuse_light::global_idx = 0;

#endif