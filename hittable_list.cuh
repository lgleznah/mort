#ifndef WORLD_H
#define WORLD_H

#include <vector>
#include <memory>
#include <cuda_runtime.h>

#include "sphere.cuh"
#include "hittable.cuh"

#define SOME_THREAD_ONLY(whatevs) {if ((threadIdx.x < 100) && (threadIdx.y < 100) && (blockIdx.x < 100) && (blockIdx.y < 100)) {whatevs;}}


using std::vector;

class hittable_list : public hittable {

	public:
		__host__ __device__ hittable_list() { num_obj = 0; objects = (hittable**)malloc(1024*sizeof(hittable*));}

		void add(hittable* object) { 
			objects[num_obj] = object; num_obj++;
			bbox = aabb(bbox, object->bounding_box());
		}

		void clear() { 
			freeFromDevice();
			num_obj = 0; 
		}

		void freeFromDevice() const override {
			for (int i = 0; i < num_obj; i++) {
				objects[i]->freeFromDevice();
			}
			HANDLE_ERROR(cudaFree(dev_objects));
		}

		int moveAllToDevice() {
			HANDLE_ERROR(cudaMalloc((void**)&dev_objects, num_obj * sizeof(hittable*)));
			int size = toDevice(dev_objects, 0);
			return size;
		}

		int toDevice(hittable** list, int idx) override;

		__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override {
			hit_record temp_rec;
			bool hit_anything = false;
			auto closest_so_far = t_max;

			for (uint16_t i = 0; i < num_obj; i++) {
				if (dev_objects[i]->hit(r, t_min, closest_so_far, temp_rec)) {
					hit_anything = true;
					closest_so_far = temp_rec.t;
					rec = temp_rec;
				}
			}

			return hit_anything;
		}

		__device__ __host__ aabb bounding_box() const override { return bbox; }

	public:
		hittable** objects;
		hittable** dev_objects;
		int num_obj;
		aabb bbox;

		hittable_list* gpu_hittable_list;
};

__global__ void hittableListToDevice(hittable_list cpu_list, hittable_list** ptr) {
	hittable_list* gpu_list = new hittable_list();
	gpu_list->num_obj = cpu_list.num_obj;
	gpu_list->dev_objects = cpu_list.dev_objects;
	gpu_list->bbox = cpu_list.bbox;
	*ptr = gpu_list;
}

int hittable_list::toDevice(hittable** list, int idx)  {
	int new_idx = idx;
	for (int i = 0; i < num_obj; i++) {
		new_idx = objects[new_idx]->toDevice(list, new_idx);
	}
	hittable_list** gpu_gpu_lst_ptr;
	HANDLE_ERROR(cudaMalloc((void**)&gpu_gpu_lst_ptr, sizeof(hittable_list*)));
	hittableListToDevice<<<1, 1 >>>(*this, gpu_gpu_lst_ptr);
	HANDLE_ERROR(cudaMemcpy(&gpu_hittable_list, gpu_gpu_lst_ptr, sizeof(hittable_list*), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaFree(gpu_gpu_lst_ptr));
	return new_idx;
}

#endif