#ifndef HITTABLE_H
#define HITTABLE_H

#include <cuda_runtime.h>

#include "ray.cuh"
#include "hit_record.cuh"
#include "aabb.cuh"

class hittable {
	public:
		virtual ~hittable() = default;

		virtual __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
		virtual int toDevice(hittable** list, int idx) = 0;
		virtual void freeFromDevice() const = 0;

		virtual __device__ aabb bounding_box() const = 0;

	public:
		hittable* gpu_obj;
};

#endif