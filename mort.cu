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
#include "objects.cuh"
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

__global__ void renderKernel(Camera camera, uchar4* ptr, curandState* states, hittable_list world) {
	camera.render(ptr, states, world);
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
	renderKernel <<<blocks, threads >>> (d->cam, output_bitmap, d->rand_states, d->data);
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

void random_spheres(hittable_list& data, Camera& cam) {
	solid_color checker1(color(.2, .3, .1));
	solid_color checker2(color(.9, .9, .9));
	checker_texture checker(0.32, checker1.getType(), checker1.getIdx(), checker2.getType(), checker2.getIdx());
	lambertian ground_material(checker.getType(), checker.getIdx());
	data.add(checker1);
	data.add(checker2);
	data.add(checker);
	data.add(ground_material);
	data.add(sphere(point3(0, -1000, 0), 1000, ground_material.getType(), ground_material.getIdx()));

	for (int a = -11; a < 11; a++) {
		for (int b = -11; b < 11; b++) {
			auto choose_mat = random_float();
			point3 center(a + 0.9 * random_float(), 0.2, b + 0.9 * random_float());

			if ((center - point3(4, 0.2, 0)).length() > 0.9) {
				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = color::random() * color::random();
					auto center2 = center + vec3(0, random_float(0.0, 0.5), 0);
					solid_color color(albedo);
					lambertian material(color.getType(), color.getIdx());
					data.add(color);
					data.add(material);
					data.add(sphere(center, center2, 0.2, material.getType(), material.getIdx()));
				}
				else if (choose_mat < 0.95) {
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = random_float(0.0, 0.5);
					metal material(albedo, fuzz);
					data.add(material);
					data.add(sphere(center, 0.2, material.getType(), material.getIdx()));
				}
				else {
					// glass
					dielectric material(1.5);
					data.add(material);
					data.add(sphere(center, 0.2, material.getType(), material.getIdx()));
				}
			}
		}
	}

	dielectric material1(1.5);
	data.add(material1);
	data.add(sphere(point3(0, 1, 0), 1.0, material1.getType(), material1.getIdx()));

	solid_color sph_color(color(0.4, 0.2, 0.1));
	lambertian material2(sph_color.getType(), sph_color.getIdx());
	data.add(sph_color);
	data.add(material2);
	data.add(sphere(point3(-4, 1, 0), 1.0, material2.getType(), material2.getIdx()));

	metal material3(color(0.7, 0.6, 0.5), 0.0);
	data.add(material3);
	data.add(sphere(point3(4, 1, 0), 1.0, material3.getType(), material3.getIdx()));

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

	return;
}

void two_spheres(hittable_list& data, Camera& cam) {
	solid_color checker1(color(.2, .3, .1));
	solid_color checker2(color(.9, .9, .9));
	checker_texture checker(0.32, checker1.getType(), checker1.getIdx(), checker2.getType(), checker2.getIdx());
	lambertian mat(checker.getType(), checker.getIdx());
	data.add(checker1);
	data.add(checker2);
	data.add(checker);
	data.add(mat);
	data.add(sphere(point3(0, -10, 0), 10, mat.getType(), mat.getIdx()));
	data.add(sphere(point3(0, 10, 0), 10, mat.getType(), mat.getIdx()));

	cam.aspect_ratio = 16.0 / 9.0;
	cam.image_width = 1200;
	cam.samples_per_pixel = 20;
	cam.bounce_limit = 50;

	cam.vfov = 20;
	cam.lookfrom = point3(13, 2, 3);
	cam.lookat = point3(0, 0, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0;
}

void earth(hittable_list& data, Camera& cam) {
	image_texture earth_texture("imgs/earthmap.jpg");
	lambertian earth_surface(earth_texture.getType(), earth_texture.getIdx());
	sphere globe(point3(0,0,0), 2, earth_surface.getType(), earth_surface.getIdx());
	data.add(earth_texture);
	data.add(earth_surface);
	data.add(globe);

	cam.aspect_ratio = 16.0 / 9.0;
	cam.image_width = 1200;
	cam.samples_per_pixel = 100;
	cam.bounce_limit = 50;

	cam.vfov = 20;
	cam.lookfrom = point3(0, 0, 12);
	cam.lookat = point3(0, 0, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0;
}

void two_perlin_spheres(hittable_list& data, Camera& cam) {
	noise_texture pertext(4.0);
	lambertian mat(pertext.getType(), pertext.getIdx());
	sphere s1(point3(0, -1000, 0), 1000, mat.getType(), mat.getIdx());
	sphere s2(point3(0, 2, 0), 2, mat.getType(), mat.getIdx());
	data.add(pertext);
	data.add(mat);
	data.add(s1);
	data.add(s2);

	cam.aspect_ratio = 16.0 / 9.0;
	cam.image_width = 1200;
	cam.samples_per_pixel = 5;
	cam.bounce_limit = 10;

	cam.vfov = 20;
	cam.lookfrom = point3(13, 2, 3);
	cam.lookat = point3(0, 0, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0;
}

void quads(hittable_list& data, Camera& cam) {
	solid_color red(color(1.0, 0.2, 0.2));
	solid_color green(color(0.2, 1.0, 0.2));
	solid_color blue(color(0.2, 0.2, 1.0));
	solid_color orange(color(1.0, 0.5, 0.0));
	solid_color teal(color(0.2, 0.8, 0.8));

	lambertian left_mat(red.getType(), red.getIdx());
	lambertian back_mat(green.getType(), green.getIdx());
	lambertian right_mat(blue.getType(), blue.getIdx());
	lambertian upper_mat(orange.getType(), orange.getIdx());
	lambertian lower_mat(teal.getType(), teal.getIdx());

	quad left_quad(point3(-3, -2, 5), vec3(0, 0, -4), vec3(0, 4, 0), left_mat.getType(), left_mat.getIdx());
	quad back_quad(point3(-2, -2, 0), vec3(4, 0, 0), vec3(0, 4, 0), back_mat.getType(), back_mat.getIdx());
	quad right_quad(point3(3, -2, 1), vec3(0, 0, 4), vec3(0, 4, 0), right_mat.getType(), right_mat.getIdx());
	quad upper_quad(point3(-2, 3, 1), vec3(4, 0, 0), vec3(0, 0, 4), upper_mat.getType(), upper_mat.getIdx());
	quad lower_quad(point3(-2, -3, 5), vec3(4, 0, 0), vec3(0, 0, -4), lower_mat.getType(), lower_mat.getIdx());

	data.add(red);
	data.add(green);
	data.add(blue);
	data.add(orange);
	data.add(teal);

	data.add(left_mat);
	data.add(back_mat);
	data.add(right_mat);
	data.add(upper_mat);
	data.add(lower_mat);

	data.add(left_quad);
	data.add(back_quad);
	data.add(right_quad);
	data.add(upper_quad);
	data.add(lower_quad);

	cam.aspect_ratio = 1.0;
	cam.image_width = 400;
	cam.samples_per_pixel = 100;
	cam.bounce_limit = 50;

	cam.vfov = 20;
	cam.lookfrom = point3(0, 0, 9);
	cam.lookat = point3(0, 0, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0;
}

void simple_light(hittable_list& data, Camera& cam) {

	noise_texture pertext(4);
	lambertian permat(pertext.getType(), pertext.getIdx());
	sphere s1(point3(0, -1000, 0), 1000, permat.getType(), permat.getIdx());
	sphere s2(point3(0, 2, 0), 2, permat.getType(), permat.getIdx());
	data.add(pertext);
	data.add(permat);
	data.add(s1);
	data.add(s2);

	solid_color lightcolor(color(4, 4, 4));
	diffuse_light difflight(lightcolor.getType(), lightcolor.getIdx());
	quad farQuad(point3(3, 1, -2), vec3(2, 0, 0), vec3(0, 2, 0), difflight.getType(), difflight.getIdx());
	sphere lightOrb(point3(0, 7, 0), 2, difflight.getType(), difflight.getIdx());
	data.add(lightcolor);
	data.add(difflight);
	data.add(farQuad);
	data.add(lightOrb);

	cam.aspect_ratio = 16.0 / 9.0;
	cam.image_width = 1200;
	cam.samples_per_pixel = 10;
	cam.bounce_limit = 10;
	cam.background = color(0.01, 0.01, 0.01);

	cam.vfov = 20;
	cam.lookfrom = point3(26, 3, 6);
	cam.lookat = point3(0, 2, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0;
}

void cornell_box(hittable_list& data, Camera& cam) {
	solid_color red(color(.65, .05, .05));
	solid_color white(color(.73, .73, .73));
	solid_color green(color(.12, .45, .15));
	solid_color light(color(15, 15, 10));

	lambertian red_wall(red.getType(), red.getIdx());
	lambertian white_wall(white.getType(), white.getIdx());
	lambertian green_wall(green.getType(), green.getIdx());
	diffuse_light lamp(light.getType(), light.getIdx());

	data.add(red);
	data.add(white);
	data.add(green);
	data.add(light);

	data.add(red_wall);
	data.add(white_wall);
	data.add(green_wall);
	data.add(lamp);

	data.add(quad(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green_wall.getType(), green_wall.getIdx()));
	data.add(quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red_wall.getType(), red_wall.getIdx()));
	data.add(quad(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), lamp.getType(), lamp.getIdx()));
	data.add(quad(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white_wall.getType(), white_wall.getIdx()));
	data.add(quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white_wall.getType(), white_wall.getIdx()));
	data.add(quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white_wall.getType(), white_wall.getIdx()));

	rotated_box(point3(130, 0, 65), point3(295, 165, 230), 15, white_wall.getType(), white_wall.getIdx(), data);
	rotated_box(point3(265, 0, 295), point3(430, 330, 460), -18, white_wall.getType(), white_wall.getIdx(), data);

	cam.aspect_ratio = 1.0;
	cam.image_width = 800;
	cam.samples_per_pixel = 1000;
	cam.bounce_limit = 500;
	cam.background = color(0, 0, 0);

	cam.vfov = 40;
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0;
}

int main(void) {
	// Scene setup
	Camera cam;
	hittable_list data;

	int scene_idx = 7;

	switch(scene_idx) {
		case 1:
			random_spheres(data, cam);
			break;

		case 2:
			two_spheres(data, cam);
			break;

		case 3:
			earth(data, cam);
			break;

		case 4:
			two_perlin_spheres(data, cam);
			break;

		case 5:
			quads(data, cam);
			break;

		case 6:
			simple_light(data, cam);
			break;

		case 7:
			cornell_box(data, cam);
			break;
	}

	cam.initialize();
	data.toDevice();

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

	//// Recursion attenuation and emission setup
	color* recursionAttenuation;
	color* recursionEmission;
	HANDLE_ERROR(cudaMalloc((void**)&recursionAttenuation, cam.bounce_limit * cam.image_width * cam.image_height * sizeof(color)));
	HANDLE_ERROR(cudaMalloc((void**)&recursionEmission, cam.bounce_limit * cam.image_width * cam.image_height * sizeof(color)));
	cam.recursionAttenuation = recursionAttenuation;
	cam.recursionEmission = recursionEmission;

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