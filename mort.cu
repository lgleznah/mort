// Mort -- My Own RayTracer

#include <cuda_runtime.h>
#include <math.h>
#include <Windows.h>

#include "book.h"
#include "cpu_bitmap.h"
#include "gpu_anim.h"
#include "gl_helper.h"

#include "hittable_list.cuh"
#include "materials.cuh"
#include "sphere.cuh"
#include "rng.cuh"
#include "vec3.cuh"
#include "camera.cuh"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

#define INF 2e10f
#define rnd(x) (x * rand() / RAND_MAX)

struct DataBlock {
	cudaEvent_t start, stop;
	GPUAnimBitmap* bitmap;
	float totalTime;
	float frames;
	Camera cam;
	curandState* rand_states;
	int prevMouseX, prevMouseY;
	hittable_list data;
};

__global__ void renderKernel(Camera camera, uchar4* ptr, curandState* states, hittable_list* data) {
	camera.render(ptr, states, data);
}

void input(DataBlock* d) {

	// Keyboard movement
	if (GetKeyState('W') & 0x8000) {
		d->cam.lookat += -(d->cam.w);
		d->cam.lookfrom += -(d->cam.w);
	}
	if (GetKeyState('S') & 0x8000) {
		d->cam.lookat += d->cam.w;
		d->cam.lookfrom += d->cam.w;
	}
	if (GetKeyState('A') & 0x8000) {
		d->cam.lookat += -(d->cam.u);
		d->cam.lookfrom += -(d->cam.u);
	}
	if (GetKeyState('D') & 0x8000) {
		d->cam.lookat += d->cam.u;
		d->cam.lookfrom += d->cam.u;
	}

	// Mouse movement
	POINT mousePos;
	GetCursorPos(&mousePos);
	int mouseDeltaX = mousePos.x - d->prevMouseX;
	int mouseDeltaY = mousePos.y - d->prevMouseY;

	if ((GetKeyState(VK_LBUTTON) & 0x8000) != 0) {
		if (mouseDeltaX != 0) {
			vec3 cam_direction = d->cam.lookat - d->cam.lookfrom;
			vec3 rotated = rotate_around(cam_direction, d->cam.vup, -mouseDeltaX / 500.0);
			d->cam.lookat = d->cam.lookfrom + rotated;
		}

		if (mouseDeltaY != 0) {
			vec3 cam_direction = d->cam.lookat - d->cam.lookfrom;
			vec3 rotated = rotate_around(cam_direction, d->cam.u, -mouseDeltaY / 500.0);
			d->cam.lookat = d->cam.lookfrom + rotated;
		}
	}
	d->prevMouseX = mousePos.x;
	d->prevMouseY = mousePos.y;
	d->cam.initialize();
}

void update(uchar4* output_bitmap, DataBlock* d, int ticks) {
	input(d);

	HANDLE_ERROR(cudaEventRecord(d->start, 0));

	//// CUDA setup
	int width_blocks = ceil((float)d->cam.image_width / 16.0);
	int height_blocks = ceil((float)d->cam.image_height / 16.0);

	dim3 blocks(width_blocks, height_blocks);
	dim3 threads(16, 16);

	//// Render
	renderKernel <<<blocks, threads >>> (d->cam, output_bitmap, d->rand_states, d->data.gpu_hittable_list);
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());

	// FPS metrics
	HANDLE_ERROR(cudaEventRecord(d->stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(d->stop));
	float elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, d->start, d->stop));

	d->totalTime += elapsedTime;
	++d->frames;

	//printf("Avg. time per frame: %3.1f ms\n", d->totalTime / d->frames);
}

void anim_exit(DataBlock* d) {
	cudaFree(d->rand_states);

	HANDLE_ERROR(cudaEventDestroy(d->start));
	HANDLE_ERROR(cudaEventDestroy(d->stop));
}

int main(void) {
	cudaEvent_t start, stop;

	// Camera settings
	Camera cam;
	cam.aspect_ratio = 16.0 / 9.0;
	cam.image_width = 1200;
	cam.samples_per_pixel = 5;
	cam.bounce_limit = 5;

	cam.vfov = 20;
	cam.lookfrom = point3(13, 2, 3);
	cam.lookat = point3(0, 0, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0.0;
	cam.focus_dist = 10.0;
	cam.initialize();

	// Object setup
	hittable_list data;

	lambertian* ground_material = new lambertian(color(0.5, 0.5, 0.5));
	data.add(new sphere(point3(0, -1000, 0), 1000, ground_material));

	for (int a = -11; a < 11; a++) { 
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_float();
			point3 center(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

			if ((center - point3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = color::random() * color::random();
					auto center2 = center + vec3(0, random_float(0.0, 0.5), 0);
					lambertian* material = new lambertian(albedo);
					data.add(new sphere(center, center2, 0.2, material));
				}
				else if (choose_mat < 0.95) {
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = random_float(0.0, 0.5);
					metal* material = new metal(albedo, fuzz);
					data.add(new sphere(center, 0.2, material));
				}
				else {
					// glass
					dielectric* material = new dielectric(1.5);
					data.add(new sphere(center, 0.2, material));
				}
			}
		}
	}

	dielectric* material1 = new dielectric(1.5);
	data.add(new sphere(point3(0, 1, 0), 1.0, material1));

	lambertian* material2 = new lambertian(color(0.4, 0.2, 0.1));
	data.add(new sphere(point3(-4, 1, 0), 1.0, material2));

	metal* material3 = new metal(color(0.7, 0.6, 0.5), 0.0);
	data.add(new sphere(point3(4, 1, 0), 1.0, material3));

	data.moveAllToDevice();

	//// CUDA setup
	int width_blocks = ceil((float)cam.image_width / 16.0);
	int height_blocks = ceil((float)cam.image_height / 16.0);

	dim3 blocks(width_blocks, height_blocks);
	dim3 threads(16, 16);

	//// Change maximum CUDA stack size. Required to avoid an unspecified launch failure due to
	//// maximum stack size getting exceeded.
	HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitStackSize, 2048));

	//// RNG initialisation
	curandState* dev_states;
	int seed = 69420;
	HANDLE_ERROR(cudaMalloc((void**)&dev_states, cam.image_width * cam.image_height * sizeof(curandState)));
	setup_rng<<<blocks, threads>>>(dev_states, seed, cam.image_width);

	//// Update function setup
	DataBlock update_data;
	GPUAnimBitmap bitmap(cam.image_width, cam.image_height, &update_data);
	update_data.bitmap = &bitmap;
	update_data.cam = cam;
	update_data.data = data;
	update_data.rand_states = dev_states;
	update_data.totalTime = 0;
	update_data.frames = 0;
	POINT mousePos;
	GetCursorPos(&mousePos);
	update_data.prevMouseX = mousePos.x;
	update_data.prevMouseY = mousePos.y;
	HANDLE_ERROR(cudaEventCreate(&update_data.start));
	HANDLE_ERROR(cudaEventCreate(&update_data.stop));

	bitmap.anim_and_exit((void (*)(uchar4*, void*, int))update, (void (*)(void*))anim_exit);
}