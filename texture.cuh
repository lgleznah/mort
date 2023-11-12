#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include <cuda_runtime_api.h>

#include "vec3.cuh"
#include "img_loader.h"
#include "interval.cuh"

#define TEXTURE_SOLID 1
#define TEXTURE_CHECKER 2
#define TEXTURE_IMAGE 3
#define TEXTURE_NOISE 4

color valueDispatch(int texType, int texIdx, float u, float v, const point3& p);

struct solid_color {
	public:
		solid_color() {}
		solid_color(color c): color_value(c) { idx = global_idx++; }

		solid_color(float red, float green, float blue): color_value(red, green, blue) {}

		__device__
		color value(float u, float v, const point3& p) {
			return color_value;
		}

		int getType() const { return TEXTURE_SOLID; }
		int getIdx() const { return idx; }

	private:
		color color_value;
		int idx;
		static int global_idx;
};

struct checker_texture {
	public:
		checker_texture() {}

		checker_texture(float _scale, int _evenType, int _evenIdx, int _oddType, int _oddIdx) {
			inv_scale = 1.0 / _scale;
			evenTextureType = _evenType;
			evenTextureIdx = _evenIdx;
			oddTextureType = _oddType;
			oddTextureIdx = _oddIdx;

			idx = global_idx++;
		}

		__device__
		color value(float u, float v, const point3& p) const {
			int xInt = (int)(floorf(inv_scale * p.x()));
			int yInt = (int)(floorf(inv_scale * p.y()));
			int zInt = (int)(floorf(inv_scale * p.z()));

			bool isEven = (xInt + yInt + zInt) % 2 == 0;
			return isEven ? valueDispatch(evenTextureType, evenTextureIdx, u, v, p) : valueDispatch(oddTextureType, oddTextureIdx, u, v, p);
		}

		int getType() const { return TEXTURE_CHECKER; }
		int getIdx() const { return idx; }

	private:
		float inv_scale;
		int evenTextureType;
		int evenTextureIdx;
		int oddTextureType;
		int oddTextureIdx;
		int idx;
		static int global_idx;
};

struct image_texture {
	public:
		image_texture() {}

		image_texture(const char* filename) {
			idx = global_idx++;

			img_loader img(filename);
			width = img.width();
			height = img.height();
			image_into_device(img);
		}

		// TODO: passing by value somehow corrupts the data array in img
		void image_into_device(const img_loader& img) {
			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, 0);
			const int pitchAlignment = prop.texturePitchAlignment;
			
			const int num_rows = img.height();
			const int num_cols = ceil(img.bytes_scanline() / pitchAlignment) * pitchAlignment;
			const int gpu_bytes = num_rows * num_cols;
			const int cpu_bytes = num_rows * img.width() * img.bytes_pixel();
			
			int remaining_in_scanline = img.bytes_scanline();
			int total_scanlines = 0;
			unsigned char* gpu_data = (unsigned char*) calloc(gpu_bytes, sizeof(unsigned char));
			unsigned char* cpu_data = img.raw_data();
			for (int gpu_data_offset = 0, cpu_data_offset = 0; cpu_data_offset < cpu_bytes; cpu_data_offset++) {
				gpu_data[gpu_data_offset++] = cpu_data[cpu_data_offset];
				remaining_in_scanline--;
				if (remaining_in_scanline == 0) {
					remaining_in_scanline = img.bytes_scanline();
					total_scanlines++;
					gpu_data_offset = total_scanlines * num_cols;
				}
			}
			unsigned char* dataDev = 0;
			cudaMalloc((void**)&dataDev, gpu_bytes);
			cudaMemcpy(dataDev, gpu_data, gpu_bytes, cudaMemcpyHostToDevice);
			struct cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(resDesc));
			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr = dataDev;
			resDesc.res.pitch2D.width = num_cols;
			resDesc.res.pitch2D.height = num_rows;
			resDesc.res.pitch2D.desc = cudaCreateChannelDesc<unsigned char>();
			resDesc.res.pitch2D.pitchInBytes = num_cols * sizeof(unsigned char);
			struct cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(texDesc));
			cudaCreateTextureObject(&gpu_tex, &resDesc, &texDesc, NULL);
			free(gpu_data);
		}

		__device__
		color value(float u, float v, const point3& p) const {
			if (height <= 0) return color(0, 1, 1);

			u = interval(0, 1).clamp(u);
			v = 1.0 - interval(0,1).clamp(v);

			int i = (int)(u * width);
			int j = (int)(v * height);

			int r = tex2D<unsigned char>(gpu_tex, i * 3 + 0, j);
			int g = tex2D<unsigned char>(gpu_tex, i * 3 + 1, j);
			int b = tex2D<unsigned char>(gpu_tex, i * 3 + 2, j);

			float color_scale = 1.0 / 255.0;
			//return color(0.0, 0.0, p.z());
			return color(color_scale*r, color_scale*g, color_scale*b);
		}

		int getType() const { return TEXTURE_IMAGE; }
		int getIdx() const { return idx; }

	private:
		cudaTextureObject_t gpu_tex;
		int width, height;
		int idx;
		static int global_idx;
};

#define POINT_COUNT 256

struct noise_texture {
	public:
		noise_texture() {}

		noise_texture(float sc): scale(sc) {
			for (int i = 0; i < POINT_COUNT; ++i) {
				ranvec[i] = unit_vector(vec3(random_float(-1, 1), random_float(-1, 1), random_float(-1, 1)));
			}

			perlin_generate_perm(perm_x);
			perlin_generate_perm(perm_y);
			perlin_generate_perm(perm_z);
		}

		__device__
		float noise(const point3& p) const {
			float u = p.x() - floorf(p.x());
			float v = p.y() - floorf(p.y());
			float w = p.z() - floorf(p.z());
			u = u * u * (3 - 2 * u);
			v = v * v * (3 - 2 * v);
			w = w * w * (3 - 2 * w);

			int i = (int)(floorf(p.x()));
			int j = (int)(floorf(p.y()));
			int k = (int)(floorf(p.z()));
			vec3 c[2][2][2];

			for (int di = 0; di < 2; di++)
				for (int dj = 0; dj < 2; dj++)
					for (int dk = 0; dk < 2; dk++) {
						int idx = perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255];
						c[di][dj][dk] = ranvec[idx];
					}

			return perlin_interp(c, u, v, w);
		}

		__device__
		color value(float u, float v, const point3& p) const {
			vec3 s = scale * p;
			return color(1,1,1) * 0.5 * (1 + sin(s.z() + 10.0 * turb(s)));
		}

		int getType() const { return TEXTURE_NOISE; }
		int getIdx() const { return idx; }

	private:
		vec3 ranvec[POINT_COUNT];
		int perm_x[POINT_COUNT];
		int perm_y[POINT_COUNT];
		int perm_z[POINT_COUNT];
		float scale;
		float _padding[2];
		int idx;
		static int global_idx;

		static void perlin_generate_perm(int* arr) {
			for (int i = 0; i < POINT_COUNT; i++) {
				arr[i] = i;
			}
			permute(arr, POINT_COUNT);
		}

		static void permute(int* arr, int n) {
			for (int i = n - 1; i > 0; i--) {
				int target = (int)random_float(0.0, i);
				int tmp = arr[i];
				arr[i] = arr[target];
				arr[target] = tmp;
			}
		}

		__device__
		static float perlin_interp(vec3 c[2][2][2], double u, double v, double w) {
			auto uu = u * u * (3 - 2 * u);
			auto vv = v * v * (3 - 2 * v);
			auto ww = w * w * (3 - 2 * w);
			auto accum = 0.0;

			for (int i = 0; i < 2; i++)
				for (int j = 0; j < 2; j++)
					for (int k = 0; k < 2; k++) {
						vec3 weight_v(u - i, v - j, w - k);
						accum += (i * uu + (1 - i) * (1 - uu))
							* (j * vv + (1 - j) * (1 - vv))
							* (k * ww + (1 - k) * (1 - ww))
							* dot(c[i][j][k], weight_v);
					}

			return accum;
		}

		__device__
		float turb(const point3& p, int depth = 7) const {
			auto accum = 0.0;
			auto temp_p = p;
			auto weight = 1.0;

			for (int i = 0; i < depth; i++) {
				accum += weight * noise(temp_p);
				weight *= 0.5;
				temp_p *= 2;
			}

			return fabsf(accum);
		}
};

int solid_color::global_idx = 0;
int checker_texture::global_idx = 0;
int image_texture::global_idx = 0;
int noise_texture::global_idx = 0;


#define NUM_SOLIDS 400
__constant__ solid_color dev_solid_colors[NUM_SOLIDS];

#define NUM_CHECKERS 400
__constant__ checker_texture dev_checkers[NUM_CHECKERS];

#define NUM_IMAGES 400
__constant__ image_texture dev_images[NUM_IMAGES];

#define NUM_NOISE 1
__constant__ noise_texture dev_noises[NUM_NOISE];

void texturesToDevice(solid_color* solids, checker_texture* checkers, image_texture* images, noise_texture* noises) {
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_solid_colors, solids, NUM_SOLIDS * sizeof(solid_color), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_checkers, checkers, NUM_CHECKERS * sizeof(checker_texture), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_images, images, NUM_IMAGES * sizeof(image_texture), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_noises, noises, NUM_NOISE * sizeof(noise_texture), 0, cudaMemcpyHostToDevice));
}

__device__
color valueDispatch(int texType, int texIdx, float u, float v, const point3& p) {
	switch (texType) {
		case TEXTURE_SOLID:
			return dev_solid_colors[texIdx].value(u, v, p);
			break;

		case TEXTURE_CHECKER:
			return dev_checkers[texIdx].value(u, v, p);
			break;

		case TEXTURE_IMAGE:
			return dev_images[texIdx].value(u, v, p);
			break;

		case TEXTURE_NOISE:
			return dev_noises[texIdx].value(u, v, p);
			break;
	}
}

#endif