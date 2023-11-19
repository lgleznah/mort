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

#endif