#ifndef UTILS_H
#define UTILS_H

#include <cmath>
#include <limits>
#include <memory>

#ifndef SOME_THREAD_ONLY
#define SOME_THREAD_ONLY(whatevs) {if ((threadIdx.x < 100) && (threadIdx.y < 100) && (blockIdx.x < 100) && (blockIdx.y < 100)) {whatevs;}}
#endif

// Usings

using std::shared_ptr;
using std::make_shared;
using std::sqrt;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

// Utility Functions

inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

// Common Headers

#include "ray.cuh"
#include "vec3.cuh"
#include "camera.cuh"

__device__ inline float clamp(float x, float min, float max) {
    if (x < min) return min;
    if (x > max) return max;
    return x;
}

__device__ inline float linear_to_gamma(float color) {
    return sqrt(color);
}

void box(const point3& a, const point3& b, int matType, int matIdx, hittable_list& data) {
    point3 min = point3(fmin(a.x(), b.x()), fmin(a.y(), b.y()), fmin(a.z(), b.z()));
    point3 max = point3(fmax(a.x(), b.x()), fmax(a.y(), b.y()), fmax(a.z(), b.z()));

    vec3 dx = vec3(max.x() - min.x(), 0, 0);
    vec3 dy = vec3(0, max.y() - min.y(), 0);
    vec3 dz = vec3(0, 0, max.z() - min.z());

    data.add(quad(point3(min.x(), min.y(), max.z()), dx, dy, matType, matIdx)); // front
    data.add(quad(point3(max.x(), min.y(), max.z()), -dz, dy, matType, matIdx)); // right
    data.add(quad(point3(max.x(), min.y(), min.z()), -dx, dy, matType, matIdx)); // back
    data.add(quad(point3(min.x(), min.y(), min.z()), dz, dy, matType, matIdx)); // left
    data.add(quad(point3(min.x(), max.y(), max.z()), dx, -dz, matType, matIdx)); // top
    data.add(quad(point3(min.x(), min.y(), min.z()), dx, dz, matType, matIdx)); // bottom

    return;
}

void rotated_box(const point3& size, const point3& translation, float theta, int matType, int matIdx, hittable_list& data) {

    vec3 dx = vec3(size.x(), 0, 0);
    vec3 dy = vec3(0, size.y(), 0);
    vec3 dz = vec3(0, 0, size.z());

    quad front(point3(0,0,size.z()), dx, dy, matType, matIdx, true); // front
    quad right(point3(size.x(), 0, size.z()), -dz, dy, matType, matIdx, true); // right
    quad back(point3(size.x(), 0, 0), -dx, dy, matType, matIdx, true); // back
    quad left(point3(0, 0, 0), dz, dy, matType, matIdx, true); // left
    quad top(point3(0, size.y(), size.z()), dx, -dz, matType, matIdx, true); // top
    quad bottom(point3(0, 0, 0), dx, dz, matType, matIdx, true); // bottom

    // The original name of this variable was front_rot, but I had to remember the promise
    rotate_y rot_front(front.getType(), front.getIdx(), theta, true);
    rotate_y rot_right(right.getType(), right.getIdx(), theta, true);
    rotate_y rot_back(back.getType(), back.getIdx(), theta, true);
    rotate_y rot_left(left.getType(), left.getIdx(), theta, true);
    rotate_y rot_top(top.getType(), top.getIdx(), theta, true);
    rotate_y rot_bottom(bottom.getType(), bottom.getIdx(), theta, true);
    
    translate tr_front(rot_front.getType(), rot_front.getIdx(), translation);
    translate tr_right(rot_right.getType(), rot_right.getIdx(), translation);
    translate tr_back(rot_back.getType(), rot_back.getIdx(), translation);
    translate tr_left(rot_left.getType(), rot_left.getIdx(), translation);
    translate tr_top(rot_top.getType(), rot_top.getIdx(), translation);
    translate tr_bottom(rot_bottom.getType(), rot_bottom.getIdx(), translation);

    data.add(front); data.add(rot_front); data.add(tr_front);
    data.add(right); data.add(rot_right); data.add(tr_right);
    data.add(back); data.add(rot_back); data.add(tr_back);
    data.add(left); data.add(rot_left); data.add(tr_left);
    data.add(top); data.add(rot_top); data.add(tr_top);
    data.add(bottom); data.add(rot_bottom); data.add(tr_bottom);
}

void rotated_box(const point3& size, const point3& translation, float theta, int matType, int matIdx, hittable_list& data) {

    vec3 dx = vec3(size.x(), 0, 0);
    vec3 dy = vec3(0, size.y(), 0);
    vec3 dz = vec3(0, 0, size.z());

    quad front(point3(0,0,size.z()), dx, dy, matType, matIdx, true); // front
    quad right(point3(size.x(), 0, size.z()), -dz, dy, matType, matIdx, true); // right
    quad back(point3(size.x(), 0, 0), -dx, dy, matType, matIdx, true); // back
    quad left(point3(0, 0, 0), dz, dy, matType, matIdx, true); // left
    quad top(point3(0, size.y(), size.z()), dx, -dz, matType, matIdx, true); // top
    quad bottom(point3(0, 0, 0), dx, dz, matType, matIdx, true); // bottom

    // The original name of this variable was front_rot, but I had to remember the promise
    rotate_y rot_front(front.getType(), front.getIdx(), theta, true);
    rotate_y rot_right(right.getType(), right.getIdx(), theta, true);
    rotate_y rot_back(back.getType(), back.getIdx(), theta, true);
    rotate_y rot_left(left.getType(), left.getIdx(), theta, true);
    rotate_y rot_top(top.getType(), top.getIdx(), theta, true);
    rotate_y rot_bottom(bottom.getType(), bottom.getIdx(), theta, true);
    
    translate tr_front(rot_front.getType(), rot_front.getIdx(), translation);
    translate tr_right(rot_right.getType(), rot_right.getIdx(), translation);
    translate tr_back(rot_back.getType(), rot_back.getIdx(), translation);
    translate tr_left(rot_left.getType(), rot_left.getIdx(), translation);
    translate tr_top(rot_top.getType(), rot_top.getIdx(), translation);
    translate tr_bottom(rot_bottom.getType(), rot_bottom.getIdx(), translation);

    data.add(front); data.add(rot_front); data.add(tr_front);
    data.add(right); data.add(rot_right); data.add(tr_right);
    data.add(back); data.add(rot_back); data.add(tr_back);
    data.add(left); data.add(rot_left); data.add(tr_left);
    data.add(top); data.add(rot_top); data.add(tr_top);
    data.add(bottom); data.add(rot_bottom); data.add(tr_bottom);
}

#endif