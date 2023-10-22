/*#ifndef HITTABLE_H
#define HITTABLE_H

#include <cuda_runtime.h>

#include "ray.cuh"
#include "hit_record.cuh"

union data_union;
union mat_union;

class hittable {
	public:
		virtual ~hittable() = default;

		virtual __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const { return false; }
		virtual int toDevice(hittable** list, int data_idx, int& mat_idx, data_union* ptr, mat_union* mat_ptr) { return data_idx; }
		virtual void freeFromDevice() const {}

	public:
		hittable* gpu_obj;
};

#endif*/