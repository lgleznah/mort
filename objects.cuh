#ifndef OBJECTS_CUH
#define OBJECTS_CUH

#include <math_constants.h>

#include "hit_record.cuh"
#include "vec3.cuh"

#define SOME_THREAD_ONLY(whatevs) {if ((threadIdx.x < 100) && (threadIdx.y < 100) && (blockIdx.x < 100) && (blockIdx.y < 100)) {whatevs;}}

#define OBJ_SPHERE 1
#define OBJ_QUAD 2
#define OBJ_TRANSLATE 3
#define OBJ_ROTATE_Y 4

bool hitDispatch(int objType, int objIdx, const ray& r, float t_min, float t_max, hit_record& rec);

struct sphere {

	public:
		__host__ 
		sphere() {}

		__host__ 
		sphere(point3 cen, float r, int material_type, int material_idx, bool _skip=false) : center1(cen), radius(r), mat_type(material_type), mat_idx(material_idx), moves(false) {
			auto rvec = vec3(radius, radius, radius);
			idx = global_idx++;
			skip = _skip;
		}

		__host__ 
		sphere(point3 cen1, point3 cen2, float r, int material_type, int material_idx, bool _skip = false) : center1(cen1), radius(r), mat_type(material_type), mat_idx(material_idx) {
			moves = true;
			center_vec = cen2 - cen1;
			auto rvec = vec3(radius, radius, radius);
			idx = global_idx++;
			skip = _skip;
		}

		point3 center1;
		float radius;

		bool moves;
		vec3 center_vec;

		int mat_type;
		int mat_idx;

		int idx;
		static int global_idx;
		bool skip;

		int getType() const { return OBJ_SPHERE; }
		int getIdx() const { return idx; }

		__device__ 
		bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
			const vec3 oc = r.origin() - center(r.time());
			const auto a = r.direction().length_squared();
			const auto half_b = dot(oc, r.direction());
			const auto c = oc.length_squared() - radius * radius;

			const auto discriminant = half_b * half_b - a * c;
			if (discriminant < 0) return false;
			const auto sqrtd = sqrt(discriminant);

			// Find the nearest root that lies in the acceptable range.
			auto root = (-half_b - sqrtd) / a;
			if (root < t_min || t_max < root) {
				root = (-half_b + sqrtd) / a;
				if (root < t_min || t_max < root)
					return false;
			}

			rec.t = root;
			rec.p = r.at(rec.t);
			const vec3 outward_normal = (rec.p - center(r.time())) / radius;
			rec.set_face_normal(r, outward_normal);
			compute_uv(outward_normal, rec.u, rec.v);
			rec.mat_type = mat_type;
			rec.mat_idx = mat_idx;

			return true;
		}
	
		__device__ 
		point3 center(float time) const {
			if (!moves) {
				return center1;
			}

			return center1 + time * center_vec;
		}

		// TODO: using CUDART_PI_F somehow causes u and v to get the SAME value. WAAAAAAAAAAT
		// Apparently, using the "f" suffix at the end of pi somehow causes this behaviour
		__device__
		void compute_uv(const point3& p, float& u, float& v) const {
			float theta = acos(-p.y());
			float phi = atan2(-p.z(), p.x()) + 3.141592565;

			u = phi / (2.0 * 3.141592565);
			v = theta / 3.141592565;
		}
};

struct quad {
	public:
		__host__
		quad() {}
		
		__host__
		quad(const point3& _Q, const point3& _u, const point3& _v, int material_type, int material_index, bool _skip=false)
			: Q(_Q), u(_u), v(_v), mat_type(material_type), mat_idx(material_index) 
		{
			vec3 n = cross(u, v);
			normal = unit_vector(n);
			D = dot(normal, Q);
			w = n / dot(n,n);
			idx = global_idx++;
			skip = _skip;
		}

		int getType() const { return OBJ_QUAD; }
		int getIdx() const { return idx; }

		__device__
		bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
			float denom = dot(normal, r.direction());

			if (fabsf(denom) < 1e-8) return false;

			float t = (D - dot(normal, r.origin())) / denom;
			if (t < t_min || t > t_max) return false;

			vec3 intersection = r.at(t);
			vec3 planar_hitp_vector = intersection - Q;
			float alpha = dot(w, cross(planar_hitp_vector, v));
			float beta = dot(w, cross(u, planar_hitp_vector));

			if ((alpha < 0) || (alpha > 1) || (beta < 0) || (beta > 1)) return false;

			rec.t = t;
			rec.p = intersection;
			rec.mat_type = mat_type;
			rec.mat_idx = mat_idx;
			rec.u = alpha;
			rec.v = beta;
			rec.set_face_normal(r, normal);

			return true;
		}

	public:
		point3 Q;
		vec3 u, v;
		vec3 normal;
		vec3 w;
		float D;
		int mat_type;
		int mat_idx;
		int idx;
		static int global_idx;
		bool skip;
};

struct translate {
	public:
		__host__
		translate() {}

		__host__
		translate(int _obj_type, int _obj_idx, const vec3& displacement, bool _skip=false) : obj_type(_obj_type), obj_idx(_obj_idx), offset(displacement) {
			idx = global_idx++;
			skip = _skip;
		}

		int getType() const { return OBJ_TRANSLATE; }
		int getIdx() const { return idx; }

		__device__
		bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
			ray offset_r(r.origin() - offset, r.direction(), r.time());

			if (!hitDispatch(obj_type, obj_idx, offset_r, t_min, t_max, rec)) {
				return false;
			}

			rec.p += offset;
			return true;
		}

	public:
		int obj_type, obj_idx;
		vec3 offset;

		int idx;
		static int global_idx;
		bool skip;
};

struct rotate_y {
	public:
		__host__
		rotate_y() {}

		__host__
		rotate_y(int _obj_type, int _obj_idx, float theta, bool _skip=false) : obj_type(_obj_type), obj_idx(_obj_idx) {
			float radians = theta * 3.1415926535897932385 / 180.0;;
			sin_theta = sin(radians);
			cos_theta = cos(radians);
			idx = global_idx++;
			skip = _skip;
		}

		int getType() const { return OBJ_ROTATE_Y; }
		int getIdx() const { return idx; }

		__device__
		bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
			// Change the ray from world space to object space
			auto origin = r.origin();
			auto direction = r.direction();

			origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
			origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

			direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
			direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

			ray rotated_r(origin, direction, r.time());

			// Determine where (if any) an intersection occurs in object space
			if (!hitDispatch(obj_type, obj_idx, rotated_r, t_min, t_max, rec))
				return false;

			// Change the intersection point from object space to world space
			auto p = rec.p;
			p[0] = cos_theta * rec.p[0] + sin_theta * rec.p[2];
			p[2] = -sin_theta * rec.p[0] + cos_theta * rec.p[2];

			// Change the normal from object space to world space
			auto normal = rec.normal;
			normal[0] = cos_theta * rec.normal[0] + sin_theta * rec.normal[2];
			normal[2] = -sin_theta * rec.normal[0] + cos_theta * rec.normal[2];

			rec.p = p;
			rec.normal = normal;

			return true;
		}

	public:
		int obj_type, obj_idx;
		float sin_theta, cos_theta;

		int idx;
		static int global_idx;
		bool skip;
};

int sphere::global_idx = 0;
int quad::global_idx = 0;
int translate::global_idx = 0;
int rotate_y::global_idx = 0;

#define NUM_SPHERES 50
__constant__ sphere dev_spheres[NUM_SPHERES];

#define NUM_QUADS 50
__constant__ quad dev_quads[NUM_QUADS];

#define NUM_TRANSLATE 50
__constant__ translate dev_translates[NUM_TRANSLATE];

#define NUM_ROTATE_Y 50
__constant__ rotate_y dev_rotate_y[NUM_ROTATE_Y];

void objectsToDevice(sphere* spheres, int num_spheres, quad* quads, int num_quads, translate* translates, int num_translates, rotate_y* rotate_ys, int num_rotate_ys) {
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_spheres, spheres, num_spheres * sizeof(sphere), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_quads, quads, num_quads * sizeof(quad), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_translates, translates, num_translates * sizeof(translate), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_rotate_y, rotate_ys, num_rotate_ys * sizeof(rotate_y), 0, cudaMemcpyHostToDevice));
}

__device__
bool hitDispatch(int objType, int objIdx, const ray& r, float t_min, float t_max, hit_record& rec) {
	switch (objType) {
		case OBJ_SPHERE:
			return dev_spheres[objIdx].hit(r, t_min, t_max, rec);
			break;

		case OBJ_QUAD:
			return dev_quads[objIdx].hit(r, t_min, t_max, rec);
			break;

		case OBJ_TRANSLATE:
			return dev_translates[objIdx].hit(r, t_min, t_max, rec);
			break;

		case OBJ_ROTATE_Y:
			return dev_rotate_y[objIdx].hit(r, t_min, t_max, rec);
			break;
	}

	return false;
}

#endif