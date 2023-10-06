#ifndef VEC3_H
#define VEC3_H

#include "rng.cuh"

#include <cmath>
#include <iostream>

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

class vec3 {
    public:
        __host__ __device__ vec3() {}
        __host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

        __host__ __device__ float x() const { return e[0]; }
        __host__ __device__ float y() const { return e[1]; }
        __host__ __device__ float z() const { return e[2]; }

        __host__ __device__ vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
        __host__ __device__ float operator[](int i) const { return e[i]; }
        __host__ __device__ float& operator[](int i) { return e[i]; }

        __host__ __device__
        vec3& operator+=(const vec3& v) {
            e[0] += v.e[0];
            e[1] += v.e[1];
            e[2] += v.e[2];
            return *this;
        }

        __host__ __device__
        vec3& operator*=(const float t) {
            e[0] *= t;
            e[1] *= t;
            e[2] *= t;
            return *this;
        }

        __host__ __device__
        vec3& operator/=(const float t) {
            return *this *= 1 / t;
        }

        __host__ __device__
        float length() const {
            return sqrtf(length_squared());
        }

        __host__ __device__
        float length_squared() const {
            return e[0] * e[0] + e[1] * e[1] + e[2] * e[2];
        }

        __host__ __device__
        bool near_zero() const {
            auto s = 1e-8;
            return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
        }

        __host__ static vec3 random() {
            return vec3(random_float(), random_float(), random_float());
        }

        __host__ static vec3 random(float min, float max) {
            return vec3(random_float(min, max), random_float(min, max), random_float(min, max));
        }

    private:
        float e[3];
};

// Type aliases for vec3
using point3 = vec3;   // 3D point
using color = vec3;    // RGB color

__host__ __device__
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

__host__ __device__
inline vec3 operator+(const vec3 &u, const vec3 &v) {
    return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

__host__ __device__
inline vec3 operator-(const vec3 &u, const vec3 &v) {
    return vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

__host__ __device__
inline vec3 operator*(const vec3 &u, const vec3 &v) {
    return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__
inline vec3 operator*(float t, const vec3 &v) {
    return vec3(t*v.x(), t*v.y(), t*v.z());
}

__host__ __device__
inline vec3 operator*(const vec3 &v, float t) {
    return t * v;
}

__host__ __device__
inline vec3 operator/(vec3 v, float t) {
    return (1/t) * v;
}

__host__ __device__
inline float dot(const vec3 &u, const vec3 &v) {
    return u.x() * v.x()
         + u.y() * v.y()
         + u.z() * v.z();
}

__host__ __device__
inline vec3 cross(const vec3 &u, const vec3 &v) {
    return vec3(u.y() * v.z() - u.z() * v.y(),
                u.z() * v.x() - u.x() * v.z(),
                u.x() * v.y() - u.y() * v.x());
}

__host__ __device__
inline vec3 elementwise_mult(const vec3& u, const vec3& v) {
    return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__
inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__device__ inline static vec3 random(curandState* states, int idx) {
    return vec3(random_float(states, idx), random_float(states, idx), random_float(states, idx));
}

__device__ inline static vec3 random(curandState* states, int idx, float min, float max) {
    return vec3(random_float(states, idx, min, max), random_float(states, idx, min, max), random_float(states, idx, min, max));
}

__device__ inline static vec3 random_in_unit_sphere(curandState* states, int idx) {
    while (true) {
        auto p = random(states, idx, -1, 1);
        if (p.length_squared() >= 1)  continue;
        return p;
    }
}

__device__ inline vec3 random_in_unit_disk(curandState* states, int idx) {
    while (true) {
        auto p = vec3(random_float(states, idx, -1, 1), random_float(states, idx, -1, 1), 0);
        if (p.length_squared() < 1)
            return p;
    }
}

__device__ inline static vec3 random_on_hemisphere(curandState* states, int idx, const vec3& normal) {
    vec3 on_unit_sphere = unit_vector(random_in_unit_sphere(states, idx));
    if (dot(on_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    auto cos_theta = min(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

__device__ float reflectance(float cosine, float ref_idx) {
    // Use Schlick's approximation for reflectance.
    float r0 = (1 - ref_idx) / (1 + ref_idx);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}

__device__ __host__ vec3 rotate_around(const vec3& vec, const vec3& axis, float theta) {
    vec3 a_parallel_b = (dot(vec, axis) / dot(axis, axis)) * axis;
    vec3 a_orthogonal_b = vec - a_parallel_b;
    vec3 w = cross(axis, a_orthogonal_b);

    float x1 = cos(theta) / a_orthogonal_b.length();
    float x2 = sin(theta) / w.length();

    vec3 a_rot_orthogonal_b = a_orthogonal_b.length() * (x1*a_orthogonal_b + x2*w);
    vec3 rotated = a_rot_orthogonal_b + a_parallel_b;

    return rotated;
}

#endif