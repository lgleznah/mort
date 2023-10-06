#ifndef SPHERE_H
#define SPHERE_H

#include "hit_record.cuh"
#include "vec3.cuh"
#include "hittable.cuh"
#include "materials.cuh"

#define SOME_THREAD_ONLY(whatevs) {if ((threadIdx.x < 100) && (threadIdx.y < 100) && (blockIdx.x < 100) && (blockIdx.y < 100)) {whatevs;}}


class sphere : public hittable {

	public:
		__host__ __device__ 
		sphere() {}

		__host__ __device__ 
		sphere(point3 cen, float r, material* m) : center1(cen), radius(r), mat(m), moves(false) {
			auto rvec = vec3(radius, radius, radius);
			bbox = aabb(center1 - rvec, center1 + rvec);
		}
		__host__ __device__ sphere(point3 cen1, point3 cen2, float r, material* m) : center1(cen1), radius(r), mat(m) {
			moves = true;
			center_vec = cen2 - cen1;
			auto rvec = vec3(radius, radius, radius);
			aabb box1(cen1 - rvec, cen2 + rvec);
			aabb box2(cen2 - rvec, cen2 + rvec);
			bbox = aabb(box1, box2);
		}

		point3 center1;
		float radius;

		bool moves;
		vec3 center_vec;

		aabb bbox;

		material* mat;
		material* gpu_mat;

		__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
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

		int toDevice(hittable** list, int idx) override;

		void freeFromDevice() const override {
			HANDLE_ERROR(cudaFree(gpu_obj));
		}

		__device__ aabb bounding_box() const override { return bbox; }
};

__global__ void sphereToDevice(sphere cpu_sphere, hittable** list, int idx, hittable** ptr) {
	sphere* gpu_sphere = new sphere();
	gpu_sphere->center1 = cpu_sphere.center1;
	gpu_sphere->radius = cpu_sphere.radius;
	gpu_sphere->moves = cpu_sphere.moves;
	gpu_sphere->center_vec = cpu_sphere.center_vec;
	gpu_sphere->gpu_mat = cpu_sphere.gpu_mat;
	gpu_sphere->gpu_obj = gpu_sphere;
	gpu_sphere->bbox = cpu_sphere.bbox;
	list[idx] = gpu_sphere;
	*ptr = gpu_sphere;
}

int sphere::toDevice(hittable** list, int idx) {
	// Move material to GPU (if it wasn't already moved)
	if (mat->gpu_mat == nullptr) {
		mat->toDevice();
		gpu_mat = mat->gpu_mat;
	}
	hittable** gpu_gpu_sph_ptr;
	HANDLE_ERROR(cudaMalloc((void**)&gpu_gpu_sph_ptr, sizeof(hittable*)));
	sphereToDevice<<<1, 1 >>>(*this, list, idx, gpu_gpu_sph_ptr);
	HANDLE_ERROR(cudaMemcpy(&gpu_obj, gpu_gpu_sph_ptr, sizeof(hittable*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(gpu_gpu_sph_ptr));
	return idx + 1;
}

#endif