#ifndef TEXTURE_CUH
#define TEXTURE_CUH

#include <cuda_runtime_api.h>

#include "vec3.cuh"

#define TEXTURE_SOLID 1
#define TEXTURE_CHECKER 2

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

int solid_color::global_idx = 0;
int checker_texture::global_idx = 0;



#define NUM_SOLIDS 500
__constant__ solid_color dev_solid_colors[NUM_SOLIDS];

#define NUM_CHECKERS 500
__constant__ checker_texture dev_checkers[NUM_CHECKERS];

void texturesToDevice(solid_color* solids, checker_texture* checkers) {
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_solid_colors, solids, NUM_SOLIDS * sizeof(solid_color), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_checkers, checkers, NUM_CHECKERS * sizeof(checker_texture), 0, cudaMemcpyHostToDevice));

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
	}
}

#endif