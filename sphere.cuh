#ifndef SPHERE_H
#define SPHERE_H

#include "hit_record.cuh"
#include "vec3.cuh"
#include "hittable.cuh"
#include "materials.cuh"

#include "data_union.cuh"

#define SOME_THREAD_ONLY(whatevs) {if ((threadIdx.x < 100) && (threadIdx.y < 100) && (blockIdx.x < 100) && (blockIdx.y < 100)) {whatevs;}}

class sphere : public hittable {

	public:
		__host__ __device__ 
		sphere() {}

		__host__ __device__ 
		sphere(point3 cen, float r, material* m) : center1(cen), radius(r), mat(m), moves(false) {
			auto rvec = vec3(radius, radius, radius);
		}

		__host__ __device__ 
		sphere(point3 cen1, point3 cen2, float r, material* m) : center1(cen1), radius(r), mat(m) {
			moves = true;
			center_vec = cen2 - cen1;
			auto rvec = vec3(radius, radius, radius);
		}

		point3 center1;
		float radius;

		bool moves;
		vec3 center_vec;

		material* mat;
		material* gpu_mat;

		__device__ 
		bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
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
			rec.mat = gpu_mat;

			return true;
		}
	
		__device__ point3 center(float time) const {
			if (!moves) {
				return center1;
			}

			return center1 + time * center_vec;
		}

		int toDevice(hittable** list, int idx, int& mat_idx, data_union* ptr, mat_union* mat_ptr) override;

		void freeFromDevice() const override {
			HANDLE_ERROR(cudaFree(gpu_obj));
		}

};

__global__ void sphereToDevice(sphere cpu_sphere, int idx, data_union* arr_ptr);

int sphere::toDevice(hittable** list, int idx, int& mat_idx, data_union* ptr, mat_union* mat_ptr) {
	// Move material to GPU (if it wasn't already moved)
	if (mat->gpu_mat == nullptr) {
		mat->toDevice(mat_idx, mat_ptr);
		gpu_mat = mat->gpu_mat;
		mat_idx += 1;
	}
	hittable** gpu_gpu_sph_ptr;
	HANDLE_ERROR(cudaMalloc((void**)&gpu_gpu_sph_ptr, sizeof(hittable*)));
	sphereToDevice<<<1, 1 >>>(*this, idx, ptr);
	HANDLE_ERROR(cudaMemcpy(&gpu_obj, gpu_gpu_sph_ptr, sizeof(hittable*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(gpu_gpu_sph_ptr));
	return idx + 1;
}

#endif