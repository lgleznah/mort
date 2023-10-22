#ifndef WORLD_H
#define WORLD_H

#include <vector>
#include <memory>
#include <cuda_runtime.h>

#include "sphere.cuh"
#include "materials.cuh"

#define SOME_THREAD_ONLY(whatevs) {if ((threadIdx.x < 100) && (threadIdx.y < 100) && (blockIdx.x < 100) && (blockIdx.y < 100)) {whatevs;}}

using std::vector;

//__constant__  data_union data_arr[600];
//__constant__  mat_union mat_arr[600];

#define NUM_SPHERES 500
__constant__ sphere dev_spheres[500];

#define NUM_LAMBERTIANS 500
__constant__ lambertian dev_lambertians[NUM_LAMBERTIANS];

#define NUM_METALS 100
__constant__ metal dev_metals[NUM_METALS];

#define NUM_DIELECTRICS 100
__constant__ dielectric dev_dielectrics[NUM_DIELECTRICS];

struct hittable_list {

	public:
		__host__  hittable_list() { 
			num_spheres = num_lambertians = num_metals = num_dielectrics = 0;

			spheres = (sphere*) malloc(NUM_SPHERES * sizeof(sphere));
			lambertians = (lambertian*) malloc(NUM_LAMBERTIANS * sizeof(lambertian));
			metals = (metal*) malloc(NUM_METALS * sizeof(metal));
			dielectrics = (dielectric*) malloc(NUM_DIELECTRICS * sizeof(dielectric));
		}

		void add(sphere object) { 
			spheres[num_spheres++] = object;
		}

		void add(lambertian mat) {
			lambertians[num_lambertians++] = mat;
		}
		
		void add(metal mat) {
			metals[num_metals++] = mat;
		}

		void add(dielectric mat) {
			dielectrics[num_dielectrics++] = mat;
		}

		void clear() { 
			free(spheres);
			free(lambertians);
			free(metals);
			free(dielectrics);
			num_spheres = num_lambertians = num_metals = num_dielectrics = 0;
		}

		int toDevice() {
			HANDLE_ERROR(cudaMemcpyToSymbol(dev_spheres, spheres, num_spheres * sizeof(sphere), 0, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpyToSymbol(dev_lambertians, lambertians, num_lambertians * sizeof(lambertian), 0, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpyToSymbol(dev_metals, metals, num_metals * sizeof(metal), 0, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpyToSymbol(dev_dielectrics, dielectrics, num_dielectrics * sizeof(dielectric), 0, cudaMemcpyHostToDevice));
		}

		__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
			hit_record temp_rec;
			bool hit_anything = false;
			auto closest_so_far = t_max;

			for (uint16_t i = 0; i < num_spheres; i++) {
				if (dev_spheres[i].hit(r, t_min, closest_so_far, temp_rec)) {
					hit_anything = true;
					closest_so_far = temp_rec.t;
					rec = temp_rec;
				}
			}

			return hit_anything;
		}

		__device__ bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* states, int idx) const {
			switch (rec.mat_type) {
				case MAT_LAMBERTIAN:
					return dev_lambertians[rec.mat_idx].scatter(r_in, rec, attenuation, scattered, states, idx);
					break;

				case MAT_METAL:
					return dev_metals[rec.mat_idx].scatter(r_in, rec, attenuation, scattered, states, idx);
					break;

				case MAT_DIELECTRIC:
					return dev_dielectrics[rec.mat_idx].scatter(r_in, rec, attenuation, scattered, states, idx);
					break;
			}
		}

	public:
		sphere* spheres;
		int num_spheres;

		lambertian* lambertians;
		int num_lambertians;

		metal* metals;
		int num_metals;

		dielectric* dielectrics;
		int num_dielectrics;

		//hittable_list* gpu_hittable_list;
};

#endif