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

struct world {

	public:
		__host__  world() { 
			objs.allocObjs();
			mats.allocMats();
			texs.allocTexs();
		}

		void add(sphere object) { 
			objs.host_sphere[objs.num_spheres++] = object;
		}

		void add(quad object) {
			objs.host_quad[objs.num_quads++] = object;
		}

		void add(translate object) {
			objs.host_translate[objs.num_translates++] = object;
		}

		void add(rotate_y object) {
			objs.host_rotate_y[objs.num_rotate_y++] = object;
		}

		void add(constant_medium object) {
			objs.host_constant_medium[objs.num_constant_medium++] = object;
		}

		void add(hittable_list object) {
			objs.host_hittable_list[objs.num_hittable_list++] = object;
		}

		void add(bvh object) {
			objs.host_bvh[objs.num_bvh++] = object;
		}

		void add(lambertian mat) {
			mats.host_lambertian[mats.num_lambertians++] = mat;
		}
		
		void add(metal mat) {
			mats.host_metal[mats.num_metals++] = mat;
		}

		void add(dielectric mat) {
			mats.host_dielectric[mats.num_dielectrics++] = mat;
		}

		void add(diffuse_light mat) {
			mats.host_diffuse_light[mats.num_diffuse_lights++] = mat;
		}

		void add(isotropic mat) {
			mats.host_isotropic[mats.num_isotropics++] = mat;
		}

		void add(solid_color tex) {
			texs.host_solid_color[texs.num_solid_colors++] = tex;
		}

		void add(checker_texture tex) {
			texs.host_checker_texture[texs.num_checker_textures++] = tex;
		}

		void add(image_texture tex) {
			texs.host_image_texture[texs.num_image_textures++] = tex;
		}

		void add(noise_texture tex) {
			texs.host_noise_texture[texs.num_noise_textures++] = tex;
		}

		void clear() { 
			objs.resetObjs();
			mats.resetMats();
			texs.resetTexs();
		}

		void toDevice() {
			materialsToDevice(mats);
			objectsToDevice(objs);
			texturesToDevice(texs);
		}

		__device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* states, int idx) const {
			hit_record temp_rec;
			bool hit_anything = false;
			auto closest_so_far = t_max;

			for (uint16_t i = 0; i < objs.num_spheres; i++) {
				if (!dev_sphere[i].skip && dev_sphere[i].hit(r, t_min, closest_so_far, temp_rec)) {
					hit_anything = true;
					closest_so_far = temp_rec.t;
					rec = temp_rec;
				}
			}

			for (uint16_t i = 0; i < objs.num_quads; i++) {
				if (!dev_quad[i].skip && dev_quad[i].hit(r, t_min, closest_so_far, temp_rec)) {
					hit_anything = true;
					closest_so_far = temp_rec.t;
					rec = temp_rec;
				}
			}

			for (uint16_t i = 0; i < objs.num_translates; i++) {
				if (!dev_translate[i].skip && dev_translate[i].hit(r, t_min, closest_so_far, temp_rec, states, idx)) {
					hit_anything = true;
					closest_so_far = temp_rec.t;
					rec = temp_rec;
				}
			}

			for (uint16_t i = 0; i < objs.num_rotate_y; i++) {
				if (!dev_rotate_y[i].skip && dev_rotate_y[i].hit(r, t_min, closest_so_far, temp_rec, states, idx)) {
					hit_anything = true;
					closest_so_far = temp_rec.t;
					rec = temp_rec;
				}
			}

			for (uint16_t i = 0; i < objs.num_constant_medium; i++) {
				if (!dev_constant_medium[i].skip && dev_constant_medium[i].hit(r, t_min, closest_so_far, temp_rec, states, idx)) {
					hit_anything = true;
					closest_so_far = temp_rec.t;
					rec = temp_rec;
				}
			}

			for (uint16_t i = 0; i < objs.num_hittable_list; i++) {
				if (!dev_hittable_list[i].skip && dev_hittable_list[i].hit(r, t_min, closest_so_far, temp_rec, states, idx)) {
					hit_anything = true;
					closest_so_far = temp_rec.t;
					rec = temp_rec;
				}
			}

			for (uint16_t i = 0; i < objs.num_bvh; i++) {
				if (!dev_bvh[i].skip && dev_bvh[i].hit(r, t_min, closest_so_far, temp_rec, states, idx)) {
					hit_anything = true;
					closest_so_far = temp_rec.t;
					rec = temp_rec;
				}
			}

			return hit_anything;
		}

	public:
		world_objects objs;
		world_materials mats;
		world_textures texs;
};

#endif