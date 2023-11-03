#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include <cuda_runtime_api.h>

#include "vec3.cuh"
#include "img_loader.h"
#include "interval.cuh"

#define TEXTURE_SOLID 1
#define TEXTURE_CHECKER 2
#define TEXTURE_IMAGE 3

color valueDispatch(int texType, int texIdx, double u, double v, const point3& p);

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
		color value(double u, double v, const point3& p) const {
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
		color value(double u, double v, const point3& p) const {
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

int solid_color::global_idx = 0;
int checker_texture::global_idx = 0;
int image_texture::global_idx = 0;


#define NUM_SOLIDS 500
__constant__ solid_color dev_solid_colors[NUM_SOLIDS];

#define NUM_CHECKERS 500
__constant__ checker_texture dev_checkers[NUM_CHECKERS];

#define NUM_IMAGES 500
__constant__ image_texture dev_images[NUM_IMAGES];

void texturesToDevice(solid_color* solids, checker_texture* checkers, image_texture* images) {
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_solid_colors, solids, NUM_SOLIDS * sizeof(solid_color), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_checkers, checkers, NUM_CHECKERS * sizeof(checker_texture), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_images, images, NUM_IMAGES * sizeof(image_texture), 0, cudaMemcpyHostToDevice));
}

__device__
color valueDispatch(int texType, int texIdx, double u, double v, const point3& p) {
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
	}
}

#endif