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
#define MAT_ISOTROPIC 5

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

struct isotropic {
	public:
		int texType, texIdx;

		int idx;
		static int global_idx;

		__host__
		isotropic() {}

		__host__
		isotropic(int _texType, int _texIdx) : texType(_texType), texIdx(_texIdx) { idx = global_idx++; }

		__device__
		bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* states, int idx) const {
			scattered = ray(rec.p, random_in_unit_sphere(states, idx), r_in.time());
			attenuation = valueDispatch(texType, texIdx, rec.u, rec.v, rec.p);
			return true;
		}

		__device__
		color emitted(float u, float v, const point3& p) const {
			return color(0.0, 0.0, 0.0);
		}

		int getType() const { return MAT_ISOTROPIC; }
		int getIdx() const { return idx; }
};

int lambertian::global_idx = 0;
int metal::global_idx = 0;
int dielectric::global_idx = 0;
int diffuse_light::global_idx = 0;
int isotropic::global_idx = 0;

#define NUM_LAMBERTIANS 50
__constant__ lambertian dev_lambertian[NUM_LAMBERTIANS];

#define NUM_METALS 100
__constant__ metal dev_metal[NUM_METALS];

#define NUM_DIELECTRICS 100
__constant__ dielectric dev_dielectric[NUM_DIELECTRICS];

#define NUM_DIFFUSE_LIGHTS 100
__constant__ diffuse_light dev_diffuse_light[NUM_DIFFUSE_LIGHTS];

#define NUM_ISOTROPICS 100
__constant__ isotropic dev_isotropic[NUM_DIFFUSE_LIGHTS];

struct world_materials {
	lambertian* host_lambertian;
	int num_lambertians;

	metal* host_metal;
	int num_metals;

	dielectric* host_dielectric;
	int num_dielectrics;

	diffuse_light* host_diffuse_light;
	int num_diffuse_lights;

	isotropic* host_isotropic;
	int num_isotropics;

	void resetCounters() {
		num_lambertians = num_metals = num_dielectrics = num_diffuse_lights = num_isotropics = 0;
	}

	void resetMats() {
		resetCounters();
		free(host_lambertian);
		free(host_metal);
		free(host_dielectric);
		free(host_diffuse_light);
		free(host_isotropic);
	}

	void allocMats() {
		resetCounters();
		host_lambertian = (lambertian*)malloc(NUM_LAMBERTIANS * sizeof(lambertian));
		host_metal = (metal*)malloc(NUM_METALS * sizeof(metal));
		host_dielectric = (dielectric*)malloc(NUM_DIELECTRICS * sizeof(dielectric));
		host_diffuse_light = (diffuse_light*)malloc(NUM_DIFFUSE_LIGHTS * sizeof(diffuse_light));
		host_isotropic = (isotropic*)malloc(NUM_ISOTROPICS * sizeof(isotropic));
	}
};

void materialsToDevice(world_materials mats) {
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_lambertian, mats.host_lambertian, mats.num_lambertians * sizeof(lambertian), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_metal, mats.host_metal, mats.num_metals * sizeof(metal), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_dielectric, mats.host_dielectric, mats.num_dielectrics * sizeof(dielectric), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_diffuse_light, mats.host_diffuse_light, mats.num_diffuse_lights * sizeof(diffuse_light), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_isotropic, mats.host_isotropic, mats.num_isotropics * sizeof(isotropic), 0, cudaMemcpyHostToDevice));
}

__device__ 
bool scatterDispatch(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* states, int idx) {
	switch (rec.mat_type) {
	case MAT_LAMBERTIAN:
		return dev_lambertian[rec.mat_idx].scatter(r_in, rec, attenuation, scattered, states, idx);
		break;

	case MAT_METAL:
		return dev_metal[rec.mat_idx].scatter(r_in, rec, attenuation, scattered, states, idx);
		break;

	case MAT_DIELECTRIC:
		return dev_dielectric[rec.mat_idx].scatter(r_in, rec, attenuation, scattered, states, idx);
		break;

	case MAT_DIFFUSE_LIGHT:
		return dev_diffuse_light[rec.mat_idx].scatter(r_in, rec, attenuation, scattered, states, idx);
		break;

	case MAT_ISOTROPIC:
		return dev_isotropic[rec.mat_idx].scatter(r_in, rec, attenuation, scattered, states, idx);
	}

	return false;
}

__device__ 
color emitDispatch(int mat_type, int mat_idx, const float u, const float v, const point3& p) {
	switch (mat_type) {
	case MAT_LAMBERTIAN:
		return dev_lambertian[mat_idx].emitted(u, v, p);
		break;

	case MAT_METAL:
		return dev_metal[mat_idx].emitted(u, v, p);
		break;

	case MAT_DIELECTRIC:
		return dev_dielectric[mat_idx].emitted(u, v, p);
		break;

	case MAT_DIFFUSE_LIGHT:
		return dev_diffuse_light[mat_idx].emitted(u, v, p);
		break;

	case MAT_ISOTROPIC:
		return dev_isotropic[mat_idx].emitted(u, v, p);
	}

	return color(0.0, 0.0, 0.0);
}

#endif