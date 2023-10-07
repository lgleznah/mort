#ifndef DATA_UNION_CUH
#define DATA_UNION_CUH

#include "hittable.cuh"
#include "sphere.cuh"

union data_union {
	data_union() {}
	~data_union() {}

	hittable hit;
	sphere sph;
};


__global__ void sphereToDevice(sphere cpu_sphere, int idx, data_union* arr_ptr) {
	sphere* other_gpu_sphere = new (&arr_ptr[idx]) sphere();
	other_gpu_sphere->center1 = cpu_sphere.center1;
	other_gpu_sphere->radius = cpu_sphere.radius;
	other_gpu_sphere->moves = cpu_sphere.moves;
	other_gpu_sphere->center_vec = cpu_sphere.center_vec;
	other_gpu_sphere->gpu_mat = cpu_sphere.gpu_mat;
}


#endif