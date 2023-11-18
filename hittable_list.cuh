#ifndef WORLD_H
#define WORLD_H

#include <vector>
#include <memory>
#include <cuda_runtime.h>

#include "objects.cuh"
#include "materials.cuh"
#include "textures.cuh"

#define SOME_THREAD_ONLY(whatevs) {if ((threadIdx.x < 100) && (threadIdx.y < 100) && (blockIdx.x < 100) && (blockIdx.y < 100)) {whatevs;}}

using std::vector;

#define NUM_SPHERES 50
__constant__ sphere dev_spheres[NUM_SPHERES];

#define NUM_QUADS 50
__constant__ quad dev_quads[NUM_QUADS];

#define NUM_LAMBERTIANS 50
__constant__ lambertian dev_lambertians[NUM_LAMBERTIANS];

#define NUM_METALS 100
__constant__ metal dev_metals[NUM_METALS];

#define NUM_DIELECTRICS 100
__constant__ dielectric dev_dielectrics[NUM_DIELECTRICS];

struct hittable_list {

	public:
		__host__  hittable_list() { 
			num_spheres = 
				num_quads =
				num_lambertians = 
				num_metals = 
				num_dielectrics = 
				num_solid_colors = 
				num_checker_textures = 
				num_image_textures = 
				num_noise_textures = 0;

			spheres = (sphere*)malloc(NUM_SPHERES * sizeof(sphere));
			quads = (quad*) malloc(NUM_QUADS * sizeof(quad));
			lambertians = (lambertian*) malloc(NUM_LAMBERTIANS * sizeof(lambertian));
			metals = (metal*) malloc(NUM_METALS * sizeof(metal));
			dielectrics = (dielectric*) malloc(NUM_DIELECTRICS * sizeof(dielectric));
			solid_colors = (solid_color*)malloc(NUM_SOLIDS * sizeof(solid_color));
			checker_textures = (checker_texture*)malloc(NUM_CHECKERS * sizeof(checker_texture));
			image_textures = (image_texture*)malloc(NUM_IMAGES * sizeof(image_texture));
			noise_textures = (noise_texture*) malloc(NUM_NOISE * sizeof(noise_texture));
		}

		void add(sphere object) { 
			spheres[num_spheres++] = object;
		}

		void add(quad object) {
			quads[num_quads++] = object;
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

		void add(solid_color tex) {
			solid_colors[num_solid_colors++] = tex;
		}

		void add(checker_texture tex) {
			checker_textures[num_checker_textures++] = tex;
		}

		void add(image_texture tex) {
			image_textures[num_image_textures++] = tex;
		}

		void add(noise_texture tex) {
			noise_textures[num_noise_textures++] = tex;
		}

		void clear() { 
			free(spheres);
			free(quads);
			free(lambertians);
			free(metals);
			free(dielectrics);
			free(solid_colors);
			free(checker_textures);
			free(image_textures);
			free(noise_textures);

			num_spheres =
				num_quads =
				num_lambertians =
				num_metals =
				num_dielectrics =
				num_solid_colors =
				num_checker_textures =
				num_image_textures =
				num_noise_textures = 0;
		}

		void toDevice() {
			HANDLE_ERROR(cudaMemcpyToSymbol(dev_spheres, spheres, num_spheres * sizeof(sphere), 0, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpyToSymbol(dev_quads, quads, num_quads * sizeof(quad), 0, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpyToSymbol(dev_lambertians, lambertians, num_lambertians * sizeof(lambertian), 0, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpyToSymbol(dev_metals, metals, num_metals * sizeof(metal), 0, cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpyToSymbol(dev_dielectrics, dielectrics, num_dielectrics * sizeof(dielectric), 0, cudaMemcpyHostToDevice));
			texturesToDevice(solid_colors, checker_textures, image_textures, noise_textures);
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

			for (uint16_t i = 0; i < num_quads; i++) {
				if (dev_quads[i].hit(r, t_min, closest_so_far, temp_rec)) {
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

			return false;
		}

	public:
		sphere* spheres;
		int num_spheres;

		quad* quads;
		int num_quads;

		lambertian* lambertians;
		int num_lambertians;

		metal* metals;
		int num_metals;

		dielectric* dielectrics;
		int num_dielectrics;

		solid_color* solid_colors;
		int num_solid_colors;

		checker_texture* checker_textures;
		int num_checker_textures;

		image_texture* image_textures;
		int num_image_textures;

		noise_texture* noise_textures;
		int num_noise_textures;
};

#endif