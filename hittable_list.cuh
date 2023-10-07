#ifndef WORLD_H
#define WORLD_H

#include <vector>
#include <memory>
#include <cuda_runtime.h>

#include "hittable.cuh"
#include "data_union.cuh"
#include "mat_union.cuh"

#define SOME_THREAD_ONLY(whatevs) {if ((threadIdx.x < 100) && (threadIdx.y < 100) && (blockIdx.x < 100) && (blockIdx.y < 100)) {whatevs;}}

using std::vector;

__constant__ __device__ data_union data_arr[600];
__constant__ __device__ mat_union mat_arr[600];

class hittable_list : public hittable {

	public:
		__host__ __device__ hittable_list() { num_obj = 0; objects = (hittable**)malloc(1024*sizeof(hittable*));}

		void add(hittable* object) { 
			objects[num_obj] = object; num_obj++;
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
			data_union* ptr;
			mat_union* mat_ptr;
			HANDLE_ERROR(cudaGetSymbolAddress((void**)&ptr, data_arr));
			HANDLE_ERROR(cudaGetSymbolAddress((void**)&mat_ptr, mat_arr));
			int mat_idx = 0;
			int size = toDevice(dev_objects, 0, mat_idx, ptr, mat_ptr);
			return size;
		}

		int toDevice(hittable** list, int idx, int& mat_idx, data_union* ptr, mat_union* mat_ptr) override;

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

	public:
		hittable** objects;
		hittable** dev_objects;
		int num_obj;

		//hittable_list* gpu_hittable_list;
};

int hittable_list::toDevice(hittable** list, int idx, int& mat_idx, data_union* ptr, mat_union* mat_ptr)  {
	int new_idx = idx;
	for (int i = 0; i < num_obj; i++) {
		new_idx = objects[new_idx]->toDevice(list, new_idx, mat_idx, ptr, mat_ptr);
	}

	return new_idx;
}

#endif