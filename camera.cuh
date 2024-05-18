#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>
#include <curand.h>

#include "vec3.cuh"
#include "utils.h"
#include "rng.cuh"

struct Camera {
    float aspect_ratio = 1.0;
	int image_width = 1500;
	int image_height;
	int samples_per_pixel = 50;
	float pixel_samples_scale;
	int sqrt_spp;
	float recip_sqrt_spp;
	int bounce_limit = 10;
	int vfov = 90;
	color background = color(0.70, 0.80, 1.00);
	color* recursionAttenuation;
	color* recursionEmission;

	point3 center;
	point3 pixel00_loc;
	vec3 pixel_delta_u;
	vec3 pixel_delta_v;

	point3 lookfrom = point3(0, 0, 1);
	point3 lookat = point3(0, 0, 0);
	vec3 vup = vec3(0, 1, 0);
	vec3 v, u, w;

	float defocus_angle = 0;
	float focus_dist = 10;
	vec3 defocus_disk_u;
	vec3 defocus_disk_v;

	void initialize() {
		image_height = static_cast<int>(image_width / aspect_ratio);
		image_height = (image_height < 1) ? 1 : image_height;

		sqrt_spp = int(sqrt(samples_per_pixel));
		pixel_samples_scale = 1.0 / (sqrt_spp * sqrt_spp);
		recip_sqrt_spp = 1.0 / sqrt_spp;

		center = lookfrom;

		// Determine viewport dimensions.
		auto theta = degrees_to_radians(vfov);
		auto h = tan(theta / 2);
		auto viewport_height = 2 * h * focus_dist;
		auto viewport_width = viewport_height * (static_cast<double>(image_width) / image_height);

		// Compute camera basis
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);

		// Calculate the vectors across the horizontal and down the vertical viewport edges.
		auto viewport_u = viewport_width * u;
		auto viewport_v = viewport_height * -v;

		// Calculate the horizontal and vertical delta vectors from pixel to pixel.
		pixel_delta_u = viewport_u / image_width;
		pixel_delta_v = -viewport_v / image_height;

		// Calculate the location of the upper left pixel.
		auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 + viewport_v / 2;
		pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

		// Compute camera defocus disk basis vectors
		auto defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
		defocus_disk_u = u * defocus_radius;
		defocus_disk_v = v * defocus_radius;
	}

    __device__ 
	color ray_color(const ray& r, curandState* states, int idx, int x, int y, world data) {
		hit_record rec;

		int iter = 0;
		ray current_ray = r;
		int recursionOffset = x * bounce_limit + y * image_width * bounce_limit;

		color finalValue;

		while (iter < bounce_limit) {
			if (data.hit(current_ray, 0.001, INFINITY, rec, states, idx)) {
				// If hit, continue recursion after computing scatter color
				ray scattered;
				color attenuation;
				color emission = emitDispatch(rec.mat_type, rec.mat_idx, rec.u, rec.v, rec.p);
				if (scatterDispatch(current_ray, rec, attenuation, scattered, states, idx)) {
					current_ray = scattered;
					recursionAttenuation[recursionOffset + iter] = attenuation;
					recursionEmission[recursionOffset + iter] = emission;
					iter++;
				}
				else {
					finalValue = emission;
					break;
				}

			}
			else {
				// If no hit, stop recursion here
				finalValue = background;
				break;
			}
		}

		if (iter == bounce_limit) {
			finalValue = color(0, 0, 0);
		}

		// Unwind recursion, multiplying by attenuation and adding emission of each iteration
		while (iter > 0) {
			iter--;
			finalValue = finalValue * recursionAttenuation[recursionOffset + iter] + recursionEmission[recursionOffset + iter];
		}

		return finalValue;
	}

	__device__ 
	void render(uchar4* ptr, curandState* states, world data) {
		int x = threadIdx.x + blockIdx.x * blockDim.x;
		int y = threadIdx.y + blockIdx.y * blockDim.y;
		int offset = x + y * image_width;

		if (x > image_width || y > image_height) return;

		color pixel_color(0, 0, 0);
		for (int s_j = 0; s_j < sqrt_spp; s_j++) {
			for (int s_i = 0; s_i < sqrt_spp; s_i++) {
				ray r = get_ray(x, y, states, offset, s_i, s_j);
				pixel_color += ray_color(r, states, offset, x, y, data);
			}
		}

		pixel_color *= pixel_samples_scale;

		pixel_color[0] = linear_to_gamma(pixel_color[0]);
		pixel_color[1] = linear_to_gamma(pixel_color[1]);
		pixel_color[2] = linear_to_gamma(pixel_color[2]);

		ptr[offset].x = (int)(256 * clamp(pixel_color.x(), 0.0, 0.999));
		ptr[offset].y = (int)(256 * clamp(pixel_color.y(), 0.0, 0.999));
		ptr[offset].z = (int)(256 * clamp(pixel_color.z(), 0.0, 0.999));
		ptr[offset].w = 255;
	}

    __device__ 
	ray get_ray(double u, double v, curandState* states, int idx, int s_i, int s_j) const {
		auto offset = sample_square_stratified(s_i, s_j, states, idx);
		auto pixel_sample = pixel00_loc + ((u + offset.x()) * pixel_delta_u) + ((v + offset.y()) * pixel_delta_v);

		auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(states, idx);
		auto ray_direction = pixel_sample - ray_origin;
		auto ray_time = random_float(states, idx);

		return ray(ray_origin, ray_direction, ray_time);
    }

	__device__ 
	vec3 pixel_sample_square(curandState* states, int idx) const {
		// Returns a random point in the square surrounding a pixel at the origin.
		auto px = -0.5 + random_float(states, idx);
		auto py = -0.5 + random_float(states, idx);
		return (px * pixel_delta_u) + (py * pixel_delta_v);
	}

	__device__ 
	point3 defocus_disk_sample(curandState* states, int idx) const {
		auto p = random_in_unit_disk(states, idx);
		return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
	}

	__device__
	vec3 sample_square_stratified(int s_i, int s_j, curandState* states, int idx) const {
		auto px = ((s_i + random_float(states, idx)) * recip_sqrt_spp) - 0.5;
		auto py = ((s_j + random_float(states, idx)) * recip_sqrt_spp) - 0.5;

		return vec3(px, py, 0);
	}
};

#endif