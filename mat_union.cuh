#ifndef MAT_UNION_CUH
#define MAT_UNION_CUH

#include "materials.cuh"

union mat_union {
	mat_union() {}
	~mat_union() {}

	material mat;
	lambertian lamb;
	metal met;
	dielectric diel;
};

__global__ void lambertianToDevice(color albedo, int mat_idx, material** ptr, mat_union* arr_ptr) {
	lambertian* gpu_lambertian = new (&arr_ptr[mat_idx]) lambertian(albedo);
	*ptr = gpu_lambertian;
}

__global__ void metalToDevice(color albedo, float fuzz, int mat_idx, material** ptr, mat_union* arr_ptr) {
	metal* gpu_metal = new (&arr_ptr[mat_idx]) metal(albedo, fuzz);
	*ptr = gpu_metal;
}

__global__ void dielectricToDevice(float ior, color albedo, int mat_idx, material** ptr, mat_union* arr_ptr) {
	dielectric* gpu_dielectric = new (&arr_ptr[mat_idx]) dielectric(ior, albedo);
	*ptr = gpu_dielectric;
}

#endif