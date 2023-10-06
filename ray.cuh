#ifndef RAY_H
#define RAY_H

#include "vec3.cuh"

class ray {
public:
    __host__ __device__ ray() {}
    __host__ __device__ ray(const point3& origin, const vec3& direction) : orig(origin), dir(direction), t(0.0) {}
    __host__ __device__ ray(const point3& origin, const vec3& direction, float time=0.0) : orig(origin), dir(direction), t(time) {}

    __host__ __device__ point3 origin() const { return orig; }
    __host__ __device__ vec3 direction() const { return dir; }
    __host__ __device__ float time() const { return t; }

    __device__
    point3 at(double t) const {
        return orig + t * dir;
    }

public:
    point3 orig;
    vec3 dir;
    float t;
};

#endif