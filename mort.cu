// Mort -- My Own RayTracer

#include <cuda_runtime.h>
#include <math.h>
#include <Windows.h>

#include "book.h"
#include "cpu_bitmap.h"
#include "gpu_anim.h"
#include "gl_helper.h"

#include "world.cuh"
#include "materials.cuh"
#include "objects.cuh"
#include "rng.cuh"
#include "vec3.cuh"
#include "camera.cuh"
#include "aabb.cuh"

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
	world data;
};

__global__ void renderKernel(Camera camera, uchar4* ptr, curandState* states, world world) {
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

	printf("Avg. time per frame: %3.1f ms\n", d->totalTime / d->frames);
}

void anim_exit(DataBlock* d) {
	cudaFree(d->rand_states);

	HANDLE_ERROR(cudaEventDestroy(d->start));
	HANDLE_ERROR(cudaEventDestroy(d->stop));
}

void random_spheres(world& data, Camera& cam) {
	hittable_list spheres(true);

	solid_color checker1(color(.2, .3, .1));
	solid_color checker2(color(.9, .9, .9));
	checker_texture checker(0.32, checker1.getType(), checker1.getIdx(), checker2.getType(), checker2.getIdx());
	lambertian ground_material(checker.getType(), checker.getIdx());
	sphere ground_sphere(point3(0, -1000, 0), 1000, ground_material.getType(), ground_material.getIdx(), true);
	data.add(checker1);
	data.add(checker2);
	data.add(checker);
	data.add(ground_material);
	data.add(ground_sphere);
	spheres.add(ground_sphere.getType(), ground_sphere.getIdx(), data.objs);

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
					sphere random_sphere(center, center2, 0.2, material.getType(), material.getIdx(), true);
					data.add(color);
					data.add(material);
					data.add(random_sphere);
					spheres.add(random_sphere.getType(), random_sphere.getIdx(), data.objs);
				}
				else if (choose_mat < 0.95) {
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = random_float(0.0, 0.5);
					metal material(albedo, fuzz);
					sphere random_sphere(center, 0.2, material.getType(), material.getIdx(), true);
					data.add(material);
					data.add(random_sphere);
					spheres.add(random_sphere.getType(), random_sphere.getIdx(), data.objs);
				}
				else {
					// glass
					dielectric material(1.5);
					sphere random_sphere(center, 0.2, material.getType(), material.getIdx(), true);
					data.add(material);
					data.add(random_sphere);
					spheres.add(random_sphere.getType(), random_sphere.getIdx(), data.objs);
				}
			}
		}
	}

	dielectric material1(1.5);
	sphere dielectric_sphere(point3(0, 1, 0), 1.0, material1.getType(), material1.getIdx(), true);
	data.add(material1);
	data.add(dielectric_sphere);
	spheres.add(dielectric_sphere.getType(), dielectric_sphere.getIdx(), data.objs);

	solid_color sph_color(color(0.4, 0.2, 0.1));
	lambertian material2(sph_color.getType(), sph_color.getIdx());
	sphere lambertian_sphere(point3(-4, 1, 0), 1.0, material2.getType(), material2.getIdx(), true);
	data.add(sph_color);
	data.add(material2);
	data.add(lambertian_sphere);
	spheres.add(lambertian_sphere.getType(), lambertian_sphere.getIdx(), data.objs);

	metal material3(color(0.7, 0.6, 0.5), 0.0);
	sphere metal_sphere(point3(4, 1, 0), 1.0, material3.getType(), material3.getIdx(), true);
	data.add(material3);
	data.add(metal_sphere);
	spheres.add(metal_sphere.getType(), metal_sphere.getIdx(), data.objs);

	data.add(spheres);

	bvh bvh_spheres(spheres, data.objs, false);
	data.add(bvh_spheres);
	data.bvh_mode = true;

	cam.aspect_ratio = 16.0 / 9.0;
	cam.image_width = 1200;
	cam.samples_per_pixel = 10;
	cam.bounce_limit = 5;

	cam.vfov = 20;
	cam.lookfrom = point3(13, 2, 3);
	cam.lookat = point3(0, 0, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0.0;
	cam.focus_dist = 10.0;

	return;
}

void two_spheres(world& data, Camera& cam) {
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

void out_of_order_spheres(world& data, Camera& cam, int n_spheres) {
	hittable_list spheres(true);

	for (int i = 0; i < n_spheres; i++) {
		auto albedo = color::random() * color::random();
		auto center = vec3(n_spheres - i, n_spheres - i, n_spheres - i);
		solid_color color(albedo);
		lambertian material(color.getType(), color.getIdx());
		sphere test(center, 0.2, material.getType(), material.getIdx(), true);
		data.add(color);
		data.add(material);
		data.add(test);
		spheres.add(test.getType(), test.getIdx(), data.objs);
	}

	data.add(spheres);
	bvh bvh_spheres(spheres, data.objs, false);
	data.add(bvh_spheres);

	cam.aspect_ratio = 16.0 / 9.0;
	cam.image_width = 1200;
	cam.samples_per_pixel = 1;
	cam.bounce_limit = 5;

	cam.vfov = 20;
	cam.lookfrom = point3(13, 2, 3);
	cam.lookat = point3(0, 0, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0.0;
	cam.focus_dist = 10.0;

	return;
}

void earth(world& data, Camera& cam) {
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

void two_perlin_spheres(world& data, Camera& cam) {
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

void quads(world& data, Camera& cam) {
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

void simple_light(world& data, Camera& cam) {

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

void cornell_box(world& data, Camera& cam) {
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

	rotated_box(point3(165, 330, 165), vec3(265, 0, 295), 15, white_wall.getType(), white_wall.getIdx(), data);
	rotated_box(point3(165, 165, 165), vec3(130, 0, 65), -18, white_wall.getType(), white_wall.getIdx(), data);

	cam.aspect_ratio = 1.0;
	cam.image_width = 800;
	cam.samples_per_pixel = 1000;
	cam.bounce_limit = 50;
	cam.background = color(0, 0, 0);

	cam.vfov = 40;
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0;
}

void cornell_smoke(world& data, Camera& cam) {
	solid_color red(color(.65, .05, .05));
	solid_color white(color(.73, .73, .73));
	solid_color green(color(.12, .45, .15));
	solid_color light(color(15, 15, 10));
	solid_color black_smoke_color(color(0, 0, 0));
	solid_color white_smoke_color(color(1, 1, 1));

	lambertian red_wall(red.getType(), red.getIdx());
	lambertian white_wall(white.getType(), white.getIdx());
	lambertian green_wall(green.getType(), green.getIdx());
	diffuse_light lamp(light.getType(), light.getIdx());
	lambertian black_smoke(black_smoke_color.getType(), black_smoke_color.getIdx());
	lambertian white_smoke(white_smoke_color.getType(), white_smoke_color.getIdx());

	data.add(red);
	data.add(white);
	data.add(green);
	data.add(light);
	data.add(black_smoke_color);
	data.add(white_smoke_color);

	data.add(red_wall);
	data.add(white_wall);
	data.add(green_wall);
	data.add(lamp);
	data.add(black_smoke);
	data.add(white_smoke);

	data.add(quad(point3(555, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), green_wall.getType(), green_wall.getIdx()));
	data.add(quad(point3(0, 0, 0), vec3(0, 555, 0), vec3(0, 0, 555), red_wall.getType(), red_wall.getIdx()));
	data.add(quad(point3(343, 554, 332), vec3(-130, 0, 0), vec3(0, 0, -105), lamp.getType(), lamp.getIdx()));
	data.add(quad(point3(0, 0, 0), vec3(555, 0, 0), vec3(0, 0, 555), white_wall.getType(), white_wall.getIdx()));
	data.add(quad(point3(555, 555, 555), vec3(-555, 0, 0), vec3(0, 0, -555), white_wall.getType(), white_wall.getIdx()));
	data.add(quad(point3(0, 0, 555), vec3(555, 0, 0), vec3(0, 555, 0), white_wall.getType(), white_wall.getIdx()));

	rotated_smoke_box(point3(165, 330, 165), vec3(265, 0, 295), 15, 0.01, black_smoke.getType(), black_smoke.getIdx(), data);
	rotated_smoke_box(point3(165, 165, 165), vec3(130, 0, 65), -18, 0.01, white_smoke.getType(), white_smoke.getIdx(), data);

	cam.aspect_ratio = 1.0;
	cam.image_width = 800;
	cam.samples_per_pixel = 2000;
	cam.bounce_limit = 50;
	cam.background = color(0, 0, 0);

	cam.vfov = 40;
	cam.lookfrom = point3(278, 278, -800);
	cam.lookat = point3(278, 278, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0;
}

void final_scene(world& data, Camera& cam, int image_width, int samples_per_pixel, int max_depth) {

	// Add ground boxes
	solid_color ground_color(color(0.48, 0.83, 0.53));
	lambertian ground_mat(ground_color.getType(), ground_color.getIdx());
	data.add(ground_color);
	data.add(ground_mat);

	int boxes_per_side = 20;
	for (int i = 0; i < boxes_per_side; i++) {
		for (int j = 0; j < boxes_per_side; j++) {
			auto w = 100.0;
			auto x0 = -1000.0 + i * w;
			auto z0 = -1000.0 + j * w;
			auto y0 = 0.0;
			auto x1 = x0 + w;
			auto y1 = random_float(1, 101);
			auto z1 = z0 + w;

			box(point3(x0, y0, z0), point3(x1, y1, z1), ground_mat.getType(), ground_mat.getIdx(), data);
		}
	}
	
	// Add light
	solid_color light_color(color(7, 7, 7));
	diffuse_light light_mat(light_color.getType(), light_color.getIdx());
	quad light(point3(123, 554, 147), vec3(300, 0, 0), vec3(0, 0, 265), light_mat.getType(), light_mat.getIdx());
	data.add(light_color);
	data.add(light_mat);
	data.add(light);

	// Add moving sphere
	vec3 center1 = point3(400, 400, 200);
	vec3 center2 = center1 + vec3(30, 0, 0);
	solid_color moving_sphere_color(color(0.7, 0.3, 0.1));
	lambertian moving_sphere_mat(moving_sphere_color.getType(), moving_sphere_color.getIdx());
	sphere moving_sphere(center1, center2, 50, moving_sphere_mat.getType(), moving_sphere_mat.getIdx());
	data.add(moving_sphere_color);
	data.add(moving_sphere_mat);
	data.add(moving_sphere);

	// Add crystal sphere
	dielectric glass_sphere_mat(1.5);
	sphere glass_sphere(point3(260, 150, 45), 50, glass_sphere_mat.getType(), glass_sphere_mat.getIdx());
	data.add(glass_sphere_mat);
	data.add(glass_sphere);

	// Add metal sphere
	metal metal_sphere_mat(color(0.8, 0.8, 0.9), 1.0);
	sphere metallic_sphere(point3(0, 150, 145), 50, metal_sphere_mat.getType(), metal_sphere_mat.getIdx());
	data.add(metal_sphere_mat);
	data.add(metallic_sphere);

	// Add blue subsurface sphere
	solid_color subsurface_sphere_color(color(0.2, 0.4, 0.9));
	lambertian subsurface_sphere_mat(subsurface_sphere_color.getType(), subsurface_sphere_color.getIdx());
	sphere subsurface_sphere(point3(360, 150, 145), 70, glass_sphere_mat.getType(), glass_sphere_mat.getIdx());
	constant_medium subsurface_sphere_inside(subsurface_sphere.getType(), subsurface_sphere.getIdx(), 0.2, subsurface_sphere_mat.getType(), subsurface_sphere_mat.getIdx(), data.objs);
	data.add(subsurface_sphere_color);
	data.add(subsurface_sphere_mat);
	data.add(subsurface_sphere);
	data.add(subsurface_sphere_inside);

	// Add fog
	solid_color boundary_color(color(1, 1, 1));
	lambertian boundary_mat(boundary_color.getType(), boundary_color.getIdx());
	sphere boundary_sphere(point3(0, 0, 0), 5000, glass_sphere_mat.getType(), glass_sphere_mat.getIdx());
	constant_medium boundary(boundary_sphere.getType(), boundary_sphere.getIdx(), 0.0001, boundary_mat.getType(), boundary_mat.getIdx(), data.objs);
	data.add(boundary_color);
	data.add(boundary_mat);
	data.add(boundary_sphere);
	data.add(boundary);

	// Add earth sphere
	image_texture earth_tex("imgs/earthmap.jpg");
	lambertian earth_mat(earth_tex.getType(), earth_tex.getIdx());
	sphere earth(point3(400, 200, 400), 100, earth_mat.getType(), earth_mat.getIdx());
	data.add(earth_tex);
	data.add(earth_mat);
	data.add(earth);

	// Add perlin sphere
	noise_texture noise_tex(0.1);
	lambertian noise_mat(noise_tex.getType(), noise_tex.getIdx());
	sphere noise_sphere(point3(220, 280, 300), 80, noise_mat.getType(), noise_mat.getIdx());
	data.add(noise_tex);
	data.add(noise_mat);
	data.add(noise_sphere);

	// Add clusterfuck of spheres
	solid_color cluster_color(color(.73, .73, .73));
	lambertian cluster_mat(cluster_color.getType(), cluster_color.getIdx());
	int ns = 1000;
	hittable_list cluster_base(true);

	for (int j = 0; j < ns; j++) {
		sphere cluster_sphere(point3::random(0, 165), 10, cluster_mat.getType(), cluster_mat.getIdx(), true);
		data.add(cluster_sphere);
		cluster_base.add(cluster_sphere.getType(), cluster_sphere.getIdx(), data.objs);
	}

	rotate_y cluster_rotate(cluster_base.getType(), cluster_base.getIdx(), 15, data.objs, true);
	translate cluster(cluster_rotate.getType(), cluster_rotate.getIdx(), vec3(-100, 270, 395), data.objs);

	data.add(cluster_color);
	data.add(cluster_mat);
	data.add(cluster_base);
	data.add(cluster_rotate);
	data.add(cluster);

	cam.aspect_ratio = 1.0;
	cam.image_width = image_width;
	cam.samples_per_pixel = samples_per_pixel;
	cam.bounce_limit = max_depth;
	cam.background = color(0, 0, 0);

	cam.vfov = 40;
	cam.lookfrom = point3(478, 278, -600);
	cam.lookat = point3(278, 278, 0);
	cam.vup = vec3(0, 1, 0);

	cam.defocus_angle = 0;
}

int main(void) {
	// Scene setup
	Camera cam;
	world data;

	int scene_idx = 1;

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

		case 8:
			cornell_smoke(data, cam);
			break;
		
		case 9:
			final_scene(data, cam, 800, 1000, 40);
			break;

		case 10:
			final_scene(data, cam, 400, 250, 4);
			break;

		case 11:
			out_of_order_spheres(data, cam, 35);
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
	HANDLE_ERROR(cudaDeviceSetLimit(cudaLimitStackSize, 8192));

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