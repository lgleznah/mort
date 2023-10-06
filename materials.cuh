#ifndef MATERIALS_H
#define MATERIALS_H

#include <curand.h>
#include <curand_kernel.h>

#include "vec3.cuh"
#include "hit_record.cuh"
#include "ray.cuh"

class material {
	public:
		virtual ~material() = default;
		virtual __device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* states, int idx) const = 0;
		virtual void toDevice() = 0;

		material* gpu_mat;
};


class lambertian : public material {
	public:
		color albedo;

		__host__ __device__ lambertian() { gpu_mat = nullptr; }
		__host__ __device__ lambertian(const color& a): albedo(a) { gpu_mat = nullptr; }

		__device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* states, int idx) const override {
			auto scatter_direction = rec.normal + random_in_unit_sphere(states, idx);
			scattered = ray(rec.p, scatter_direction, r_in.time());
			attenuation = albedo;
			return true;
		}

		void toDevice() override;
};

class metal : public material {
	public:
		color albedo;
		float fuzz;

		__host__ __device__ metal() { gpu_mat = nullptr; }
		__host__ __device__ metal(const color& a, float f): albedo(a), fuzz(f) { gpu_mat = nullptr; }

		__device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* states, int idx) const override {
			vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
			reflected += random_in_unit_sphere(states, idx) * fuzz;
			scattered = ray(rec.p, reflected, r_in.time());
			attenuation = albedo;
			return dot(scattered.direction(), rec.normal) > 0;
		}

		virtual void toDevice() override;
};

class dielectric : public material {
	public:
		float ior, inv_ior;
		color albedo;

		__host__ __device__ dielectric() { gpu_mat = nullptr; }
		__host__ __device__ dielectric(float refraction_index, const color& a) : ior(refraction_index), inv_ior(1.0 / refraction_index), albedo(a) { gpu_mat = nullptr; }
		__host__ __device__ dielectric(float refraction_index): ior(refraction_index), inv_ior(1.0 / refraction_index), albedo(color(1.0,1.0,1.0)) { gpu_mat = nullptr; }

		__device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* states, int idx) const override {
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

		virtual void toDevice() override;
};


__global__ void lambertianToDevice(color albedo, material** ptr) {
	lambertian* gpu_lambertian = new lambertian(albedo);
	*ptr = gpu_lambertian;
}

__global__ void metalToDevice(color albedo, float fuzz, material** ptr) {
	metal* gpu_metal = new metal(albedo, fuzz);
	*ptr = gpu_metal;
}

__global__ void dielectricToDevice(float ior, color albedo, material** ptr) {
	dielectric* gpu_dielectric = new dielectric(ior, albedo);
	*ptr = gpu_dielectric;
}

void lambertian::toDevice() {
	material** gpu_gpu_mat_ptr;
	HANDLE_ERROR(cudaMalloc((void**)&gpu_gpu_mat_ptr, sizeof(material*)));
	lambertianToDevice<<<1, 1 >>>(albedo, gpu_gpu_mat_ptr);
	HANDLE_ERROR(cudaMemcpy(&gpu_mat, gpu_gpu_mat_ptr, sizeof(material*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(gpu_gpu_mat_ptr));
}

void metal::toDevice() {
	material** gpu_gpu_mat_ptr;
	HANDLE_ERROR(cudaMalloc((void**)&gpu_gpu_mat_ptr, sizeof(material*)));
	metalToDevice<<<1, 1 >>>(albedo, fuzz, gpu_gpu_mat_ptr);
	HANDLE_ERROR(cudaMemcpy(&gpu_mat, gpu_gpu_mat_ptr, sizeof(material*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(gpu_gpu_mat_ptr));
}

void dielectric::toDevice() {
	material** gpu_gpu_mat_ptr;
	HANDLE_ERROR(cudaMalloc((void**)&gpu_gpu_mat_ptr, sizeof(material*)));
	dielectricToDevice<<<1, 1 >>>(ior, albedo, gpu_gpu_mat_ptr);
	HANDLE_ERROR(cudaMemcpy(&gpu_mat, gpu_gpu_mat_ptr, sizeof(material*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(gpu_gpu_mat_ptr));
}

#endif