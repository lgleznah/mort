#ifndef OBJECTS_CUH
#define OBJECTS_CUH

#include <math_constants.h>

#include "hit_record.cuh"
#include "vec3.cuh"
#include "aabb.cuh"
#include "onb.cuh"

#define SOME_THREAD_ONLY(whatevs) {if ((threadIdx.x < 100) && (threadIdx.y < 100) && (blockIdx.x < 100) && (blockIdx.y < 100)) {whatevs;}}

#define OBJ_SPHERE 1
#define OBJ_QUAD 2
#define OBJ_TRANSLATE 3
#define OBJ_ROTATE_Y 4
#define OBJ_CONSTANT_MEDIUM 5
#define OBJ_HITTABLE_LIST 6
#define OBJ_BVH 7

#define OBJ_SWAP(type, list) {type temp = list[obj_idx_1]; list[obj_idx_1] = list[obj_idx_2]; list[obj_idx_2] = temp;}

bool hitDispatch(int objType, int objIdx, const ray& r, float t_min, float t_max, hit_record& rec, curandState* states, int idx);
aabb getBboxInfo(int objType, int objIdx);
aabb host_getBboxInfo(int objType, int objIdx, const struct world_objects& objs);
int compare_by_axis(const struct hittable_list& list, int obj1, int obj2, const struct world_objects& objs, int axis);
void swap_objects(int obj_type, int obj_idx_1, int obj_idx_2, world_objects& objs);
float pdfValueDispatch(int objType, int objIdx, const point3& origin, const vec3& direction);
vec3 randomDispatch(int objType, int objIdx, const point3& origin, curandState* states, int idx);

struct sphere {

	public:
		__host__ 
		sphere() {}

		__host__ 
		sphere(point3 cen, float r, int material_type, int material_idx, bool _skip=false) : center1(cen), radius(r), mat_type(material_type), mat_idx(material_idx), moves(false) {
			auto rvec = vec3(radius, radius, radius);
			idx = global_idx++;
			skip = _skip;
			bbox = aabb(center1 - rvec, center1 + rvec);
		}

		__host__ 
		sphere(point3 cen1, point3 cen2, float r, int material_type, int material_idx, bool _skip = false) : center1(cen1), radius(r), mat_type(material_type), mat_idx(material_idx) {
			moves = true;
			center_vec = cen2 - cen1;
			auto rvec = vec3(radius, radius, radius);
			idx = global_idx++;
			skip = _skip;
			aabb box1(cen1 - rvec, cen1 + rvec);
			aabb box2(cen2 - rvec, cen2 + rvec);
			bbox = aabb(box1, box2);
		}

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

		__device__
		float pdf_value(const point3& origin, const vec3& direction) const {
			hit_record rec;

			if (!hit(ray(origin, direction, 0), 0.001, HUGE_VALF, rec)) {
				return false;
			}

			float cos_theta_max = sqrt(1 - radius * radius / (center1 - origin).length_squared());
			float solid_angle = 2 * 3.1415926 * (1 - cos_theta_max);

			return 1.0 / solid_angle;
		}

		__device__
		vec3 random(const point3& origin, curandState* states, int idx) const {
			vec3 direction = center1 - origin;
			float distance_squared = direction.length_squared();
			onb uvw;
			uvw.build_from_w(direction);
			return uvw.local(random_to_sphere(radius, distance_squared, states, idx));
		}

		__device__
		static vec3 random_to_sphere(float radius, float distance_squared, curandState* states, int idx) {
			float r1 = random_float(states, idx);
			float r2 = random_float(states, idx);

			float z = 1 + r2 * (sqrt(1 - radius * radius / distance_squared) - 1);

			float phi = 2 * 3.141592 * r1;
			float x = cos(phi) * sqrt(1 - z * z);
			float y = sin(phi) * sqrt(1 - z * z);

			return vec3(x, y, z);
		}

	public:
		point3 center1;
		float radius;

		bool moves;
		vec3 center_vec;

		int mat_type;
		int mat_idx;

		int idx;
		static int global_idx;
		bool skip;

		aabb bbox;
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
			area = n.length();
			idx = global_idx++;
			skip = _skip;

			// Set bounding box
			auto bbox_diagonal1 = aabb(Q, Q + u + v);
			auto bbox_diagonal2 = aabb(Q + u, Q + v);
			bbox = aabb(bbox_diagonal1, bbox_diagonal2);
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

		__device__
		float pdf_value(const point3& origin, const vec3& direction) const {
			hit_record rec;

			// TODO: this should reflect hit time
			if (!hit(ray(origin, direction, 0), 0.001, HUGE_VALF, rec))
				return 0;

			auto distance_squared = rec.t * rec.t * direction.length_squared();
			auto cosine = fabs(dot(direction, rec.normal) / direction.length());

			return distance_squared / (cosine * area);
		}

		__device__
		vec3 random(const point3& origin, curandState* states, int idx) const {
			auto p = Q + (random_float(states, idx) * u) + (random_float(states, idx) * v);
			return p - origin;
		}

	public:
		point3 Q;
		vec3 u, v;
		vec3 normal;
		vec3 w;
		aabb bbox;
		float D;
		float area;
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
		translate(int _obj_type, int _obj_idx, const vec3& displacement, const world_objects& objs, bool _skip=false) : obj_type(_obj_type), obj_idx(_obj_idx), offset(displacement) {
			idx = global_idx++;
			skip = _skip;

			bbox = host_getBboxInfo(_obj_type, _obj_idx, objs) + displacement;
		}

		int getType() const { return OBJ_TRANSLATE; }
		int getIdx() const { return idx; }

		__device__
		bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* states, int idx) const {
			ray offset_r(r.origin() - offset, r.direction(), r.time());

			if (!hitDispatch(obj_type, obj_idx, offset_r, t_min, t_max, rec, states, idx)) {
				return false;
			}

			rec.p += offset;
			return true;
		}

	public:
		int obj_type, obj_idx;
		vec3 offset;
		aabb bbox;

		int idx;
		static int global_idx;
		bool skip;
};

struct rotate_y {
	public:
		__host__
		rotate_y() {}

		__host__
		rotate_y(int _obj_type, int _obj_idx, float theta, const world_objects& objs, bool _skip=false) : obj_type(_obj_type), obj_idx(_obj_idx) {
			float radians = theta * 3.1415926535897932385 / 180.0;;
			sin_theta = sin(radians);
			cos_theta = cos(radians);
			idx = global_idx++;
			skip = _skip;

			// Compute bounding box
			bbox = host_getBboxInfo(_obj_type, _obj_idx, objs);
			point3 pmin(HUGE_VALF, HUGE_VALF, HUGE_VALF);
			point3 pmax(-HUGE_VALF, -HUGE_VALF, -HUGE_VALF);

			for (int i = 0; i < 2; i++) {
				for (int j = 0; j < 2; j++) {
					for (int k = 0; k < 2; k++) {
						auto x = i * bbox.x.imax + (1 - i) * bbox.x.imin;
						auto y = j * bbox.y.imax + (1 - j) * bbox.y.imin;
						auto z = k * bbox.z.imax + (1 - k) * bbox.z.imin;

						auto newx = cos_theta * x + sin_theta * z;
						auto newz = -sin_theta * x + cos_theta * z;

						vec3 tester(newx, y, newz);

						for (int c = 0; c < 3; c++) {
							pmin[c] = fmin(pmin[c], tester[c]);
							pmax[c] = fmax(pmax[c], tester[c]);
						}
					}
				}
			}

			bbox = aabb(pmin, pmax);
		}

		int getType() const { return OBJ_ROTATE_Y; }
		int getIdx() const { return idx; }

		__device__
		bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* states, int idx) const {
			// Change the ray from world space to object space
			auto origin = r.origin();
			auto direction = r.direction();

			origin[0] = cos_theta * r.origin()[0] - sin_theta * r.origin()[2];
			origin[2] = sin_theta * r.origin()[0] + cos_theta * r.origin()[2];

			direction[0] = cos_theta * r.direction()[0] - sin_theta * r.direction()[2];
			direction[2] = sin_theta * r.direction()[0] + cos_theta * r.direction()[2];

			ray rotated_r(origin, direction, r.time());

			// Determine where (if any) an intersection occurs in object space
			if (!hitDispatch(obj_type, obj_idx, rotated_r, t_min, t_max, rec, states, idx))
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
		aabb bbox;

		int idx;
		static int global_idx;
		bool skip;
};

struct constant_medium {
	public:
		__host__
		constant_medium() {}

		__host__
		constant_medium(int _objType, int _objIdx, float d, int _matType, int _matIdx, const world_objects& objs, bool _skip=false) {
			obj_type = _objType;
			obj_idx = _objIdx;
			neg_inv_density = -(1.0 / d);
			mat_type = _matType;
			mat_idx = _matIdx;
			idx = global_idx++;
			skip = _skip;

			bbox = host_getBboxInfo(_objType, _objIdx, objs);
		}

		__device__
		bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* states, int idx) const {
			hit_record rec1, rec2;

			if (!hitDispatch(obj_type, obj_idx, r, -CUDART_INF_F, CUDART_INF_F, rec1, states, idx)) {
				return false;
			}

			if (!hitDispatch(obj_type, obj_idx, r, rec1.t + 0.0001, CUDART_INF_F, rec2, states, idx)) {
				return false;
			}


			if (rec1.t < t_min) rec1.t = t_min;
			if (rec2.t > t_max) rec2.t = t_max;

			if (rec1.t >= rec2.t)
				return false;

			if (rec1.t < 0)
				rec1.t = 0;

			auto ray_length = r.direction().length();
			auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
			auto hit_distance = neg_inv_density * log(random_float(states, idx));

			if (hit_distance > distance_inside_boundary)
				return false;

			rec.t = rec1.t + hit_distance / ray_length;
			rec.p = r.at(rec.t);

			rec.normal = vec3(1, 0, 0);  // arbitrary
			rec.front_face = true;     // also arbitrary
			rec.mat_type = mat_type;
			rec.mat_idx = mat_idx;

			return true;
		}

		int getType() const { return OBJ_CONSTANT_MEDIUM; }
		int getIdx() const { return idx; }


	public:
		int obj_type, obj_idx;
		double neg_inv_density;
		int mat_type, mat_idx;
		aabb bbox;

		int idx;
		static int global_idx;
		bool skip;
};

#define LIST_MAX_OBJS 1000

struct hittable_list {
	public:
		__host__
		hittable_list() {}

		__host__
		hittable_list(bool _skip): skip(_skip) { idx = global_idx++; num_objs = 0; bbox = aabb(vec3(0, 0, 0), vec3(0, 0, 0)); }

		__host__
		void add(int _objType, int _objIdx, const struct world_objects& objs) {
			if (num_objs < LIST_MAX_OBJS) {
				obj_types[num_objs] = _objType;
				obj_idxs[num_objs] = _objIdx;
				bbox = (num_objs == 0) ? host_getBboxInfo(_objType, _objIdx, objs) : aabb(bbox, host_getBboxInfo(_objType, _objIdx, objs));
				num_objs += 1;
			}
		}

		__device__
		bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* states, int idx) const {
			hit_record temp_rec;
			bool hit_anything = false;
			auto closest_so_far = t_max;

			for (uint16_t i = 0; i < num_objs; i++) {
				if (hitDispatch(obj_types[i], obj_idxs[i], r, t_min, closest_so_far, temp_rec, states, idx)) {
					hit_anything = true;
					closest_so_far = temp_rec.t;
					rec = temp_rec;
				}
			}

			return hit_anything;
		}

		__device__
		float pdf_value(const point3& origin, const vec3& direction) const {
			float weight = 1.0 / (float)(num_objs);
			float sum = 0.0;

			for (uint16_t i = 0; i < num_objs; i++) {
				sum += weight * pdfValueDispatch(obj_types[i], obj_idxs[i], origin, direction);
			}

			return sum;
		}

		__device__
		vec3 random(const point3& origin, curandState* states, int idx) const {
			int random_idx = random_int(states, idx, 0, num_objs - 1);
			return randomDispatch(obj_types[random_idx], obj_idxs[random_idx], origin, states, idx);
		}

		int getType() const { return OBJ_HITTABLE_LIST; }
		int getIdx() const { return idx; }


	public:
		int obj_types[LIST_MAX_OBJS], obj_idxs[LIST_MAX_OBJS];
		int num_objs;

		int idx;
		static int global_idx;
		bool skip;

		aabb bbox;
};

#define MAX_BVH_NODES 1024

struct bvh {
	public:
		__host__
		bvh() {}

		__host__
		bvh(hittable_list& list, world_objects& objs, bool _skip): skip(_skip) {
			int bvh_size = 1;
			int curr_bvh_idx = 0;

			idx = global_idx++;

			int span_beginnings[MAX_BVH_NODES], span_ends[MAX_BVH_NODES];
			span_beginnings[0] = 0;
			span_ends[0] = list.num_objs;

			while (curr_bvh_idx < bvh_size) {
				// Fetch data for current node being built
				int curr_start = span_beginnings[curr_bvh_idx];
				int curr_end = span_ends[curr_bvh_idx];

				// Select axis to sort
				bounding_boxes[curr_bvh_idx] = aabb::empty;
				for (int i = curr_start; i < curr_end; i++) {
					bounding_boxes[curr_bvh_idx] = aabb(bounding_boxes[curr_bvh_idx], host_getBboxInfo(list.obj_types[i], list.obj_idxs[i], objs));
				}

				int axis = bounding_boxes[curr_bvh_idx].largest_axis();

				// Determine if this is a leaf node (1 or 2 elements)
				size_t span = curr_end - curr_start;

				if (span == 1) {
					left_children_types[curr_bvh_idx] = list.obj_types[curr_start];
					left_children_idxs[curr_bvh_idx] = list.obj_idxs[curr_start];

					right_children_types[curr_bvh_idx] = list.obj_types[curr_start];
					right_children_idxs[curr_bvh_idx] = list.obj_idxs[curr_start];

					is_internal_node[curr_bvh_idx] = false;
				}

				else if (span == 2) {
					if (compare_by_axis(list, curr_start, curr_start + 1, objs, axis) <= 0) {
						left_children_types[curr_bvh_idx] = list.obj_types[curr_start];
						left_children_idxs[curr_bvh_idx] = list.obj_idxs[curr_start];

						right_children_types[curr_bvh_idx] = list.obj_types[curr_start + 1];
						right_children_idxs[curr_bvh_idx] = list.obj_idxs[curr_start + 1];
					}

					else {
						left_children_types[curr_bvh_idx] = list.obj_types[curr_start + 1];
						left_children_idxs[curr_bvh_idx] = list.obj_idxs[curr_start + 1];

						right_children_types[curr_bvh_idx] = list.obj_types[curr_start];
						right_children_idxs[curr_bvh_idx] = list.obj_idxs[curr_start];
					}

					is_internal_node[curr_bvh_idx] = false;
				}

				else {
					// Sort given list from given start to given end, using the data contained in the world
					sort_obj_list(list, curr_start, curr_end, objs, axis);

					int mid_point = curr_start + (span / 2 + (span % 2 != 0));

					left_children_types[curr_bvh_idx] = OBJ_BVH;
					left_children_idxs[curr_bvh_idx] = bvh_size;
					span_beginnings[bvh_size] = curr_start;
					span_ends[bvh_size] = mid_point;
					bvh_size += 1;

					right_children_types[curr_bvh_idx] = OBJ_BVH;
					right_children_idxs[curr_bvh_idx] = bvh_size;
					span_beginnings[bvh_size] = mid_point;
					span_ends[bvh_size] = curr_end;
					bvh_size += 1;

					is_internal_node[curr_bvh_idx] = true;
				}

				curr_bvh_idx += 1;
			}

			// Compute AABBs once BVH is built
			//build_aabb_hierarchy(0, objs);
		}

		__host__
		void build_aabb_hierarchy(int idx, world_objects& objs) {
			if (!is_internal_node[idx]) {
				bounding_boxes[idx] = aabb(
										host_getBboxInfo(left_children_types[idx], left_children_idxs[idx], objs),
										host_getBboxInfo(right_children_types[idx], right_children_idxs[idx], objs)
									  );
			}
			else {
				build_aabb_hierarchy(left_children_idxs[idx], objs);
				build_aabb_hierarchy(right_children_idxs[idx], objs);
				bounding_boxes[idx] = aabb(bounding_boxes[left_children_idxs[idx]], bounding_boxes[right_children_idxs[idx]]);
			}

			return;
		}

		__host__
		void sort_obj_list(hittable_list& list, int start, int end, world_objects& objs, int axis) {
			// TODO do something better than bubble sort
			bool swapped;
			int temp;
			for (int i = 0; i < (end - start) - 1; i++) {
				swapped = false;
				for (int j = start; j < end - i - 1; j++) {
					if (compare_by_axis(list, j, j + 1, objs, axis) == 1) {
						// Swap the items themselves if they are of the same type
						if (list.obj_types[j] == list.obj_types[j + 1]) {
							swap_objects(list.obj_types[j], list.obj_idxs[j], list.obj_idxs[j + 1], objs);
						}

						else {
							temp = list.obj_types[j];
							list.obj_types[j] = list.obj_types[j + 1];
							list.obj_types[j + 1] = temp;
						
							temp = list.obj_idxs[j];
							list.obj_idxs[j] = list.obj_idxs[j + 1];
							list.obj_idxs[j + 1] = temp;
						}
						swapped = true;
					}
				}

				if (!swapped) {
					break;
				}
			}
		}


		__device__
		bool hit(const ray& r, float t_min, float t_max, hit_record& rec, curandState* states, int idx) const {
			struct stack_frame {
				int node_id;
				int child_evaluated;
				bool left_child_result;
				float t_max;
			};

			bool retval;
			stack_frame stack[20];
			int stack_index = 0;
			stack_frame curr_frame;
			stack[0] = {0, 0, -1, t_max};

			while (stack_index >= 0) {
				curr_frame = stack[stack_index];

				if (curr_frame.child_evaluated == 0) {

					if (!bounding_boxes[curr_frame.node_id].hit(r, t_min, curr_frame.t_max)) {
						retval = false;
						stack_index--;
						continue;
					}

					if (!is_internal_node[curr_frame.node_id]) {
						bool hit_left = hitDispatch(left_children_types[curr_frame.node_id], left_children_idxs[curr_frame.node_id], r, t_min, curr_frame.t_max, rec, states, idx);
						bool hit_right = hitDispatch(right_children_types[curr_frame.node_id], right_children_idxs[curr_frame.node_id], r, t_min, hit_left ? rec.t : curr_frame.t_max, rec, states, idx);

						retval = hit_left || hit_right;
						stack_index--;
						continue;
					}

					else {
						stack[stack_index].child_evaluated += 1;
						stack_index++;
						stack[stack_index] = { left_children_idxs[curr_frame.node_id], 0, -1, curr_frame.t_max };
						continue;
					}
				}

				else if (curr_frame.child_evaluated == 1) {
					stack[stack_index].left_child_result = retval;
					stack[stack_index].child_evaluated += 1;
					stack_index++;
					stack[stack_index] = { right_children_idxs[curr_frame.node_id], 0, -1, retval ? rec.t : curr_frame.t_max };
					continue;
				}

				else if (curr_frame.child_evaluated == 2) {
					retval = curr_frame.left_child_result || retval;
					stack_index--;
					continue;
				}
			}

			return retval;
		}

	public:

		// Elements indexed per node ID
		int left_children_types[MAX_BVH_NODES], left_children_idxs[MAX_BVH_NODES];
		int right_children_types[MAX_BVH_NODES], right_children_idxs[MAX_BVH_NODES];
		bool is_internal_node[MAX_BVH_NODES];
		aabb bounding_boxes[MAX_BVH_NODES];

		int idx;
		static int global_idx;
		bool skip;
};

int sphere::global_idx = 0;
int quad::global_idx = 0;
int translate::global_idx = 0;
int rotate_y::global_idx = 0;
int constant_medium::global_idx = 0;
int hittable_list::global_idx = 0;
int bvh::global_idx = 0;

#define NUM_SPHERES 1100
__device__ sphere dev_sphere[NUM_SPHERES];

#define NUM_QUADS 2500
__device__ quad dev_quad[NUM_QUADS];

#define NUM_TRANSLATE 50
__constant__ translate dev_translate[NUM_TRANSLATE];

#define NUM_ROTATE_Y 50
__constant__ rotate_y dev_rotate_y[NUM_ROTATE_Y];

#define NUM_CONSTANT_MEDIUM 50
__constant__ constant_medium dev_constant_medium[NUM_CONSTANT_MEDIUM];

#define NUM_HITTABLE_LIST 2
__device__ hittable_list dev_hittable_list[NUM_HITTABLE_LIST];

#define NUM_BVH 2
__device__ bvh dev_bvh[NUM_BVH];

struct world_objects {
	sphere* host_sphere;
	int num_spheres;

	quad* host_quad;
	int num_quads;

	translate* host_translate;
	int num_translates;

	rotate_y* host_rotate_y;
	int num_rotate_y;

	constant_medium* host_constant_medium;
	int num_constant_medium;

	hittable_list* host_hittable_list;
	int num_hittable_list;

	bvh* host_bvh;
	int num_bvh;

	void resetCounters() {
		num_spheres = num_quads = num_translates = num_rotate_y = num_constant_medium = num_hittable_list = num_bvh = 0;
	}

	void resetObjs() {
		resetCounters();
		free(host_sphere);
		free(host_quad);
		free(host_translate);
		free(host_rotate_y);
		free(host_constant_medium);
		free(host_hittable_list);
		free(host_bvh);
	}
	
	void allocObjs() {
		resetCounters();
		host_sphere = (sphere*)malloc(NUM_SPHERES * sizeof(sphere));
		host_quad = (quad*)malloc(NUM_QUADS * sizeof(quad));
		host_translate = (translate*)malloc(NUM_TRANSLATE * sizeof(translate));
		host_rotate_y = (rotate_y*)malloc(NUM_ROTATE_Y * sizeof(rotate_y));
		host_constant_medium = (constant_medium*)malloc(NUM_CONSTANT_MEDIUM * sizeof(constant_medium));
		host_hittable_list = (hittable_list*)malloc(NUM_HITTABLE_LIST * sizeof(hittable_list));
		host_bvh = (bvh*)malloc(NUM_BVH * sizeof(bvh));
	}

	void swap(int obj_type, int obj_idx_1, int obj_idx_2) {
		switch (obj_type) {
		case OBJ_SPHERE:
			OBJ_SWAP(sphere, host_sphere);
			break;

		case OBJ_QUAD:
			OBJ_SWAP(quad, host_quad);
			break;

		case OBJ_TRANSLATE:
			OBJ_SWAP(translate, host_translate);
			break;

		case OBJ_ROTATE_Y:
			OBJ_SWAP(rotate_y, host_rotate_y);
			break;

		case OBJ_CONSTANT_MEDIUM:
			OBJ_SWAP(constant_medium, host_constant_medium);
			break;

		case OBJ_HITTABLE_LIST:
			OBJ_SWAP(hittable_list, host_hittable_list);
			break;

		case OBJ_BVH:
			OBJ_SWAP(bvh, host_bvh);
			break;
		}
	}
};

void objectsToDevice(world_objects objs) {
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_sphere, objs.host_sphere, objs.num_spheres * sizeof(sphere), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_quad, objs.host_quad, objs.num_quads * sizeof(quad), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_translate, objs.host_translate, objs.num_translates * sizeof(translate), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_rotate_y, objs.host_rotate_y, objs.num_rotate_y * sizeof(rotate_y), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_constant_medium, objs.host_constant_medium, objs.num_constant_medium * sizeof(constant_medium), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_hittable_list, objs.host_hittable_list, objs.num_hittable_list * sizeof(hittable_list), 0, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(dev_bvh, objs.host_bvh, objs.num_bvh * sizeof(bvh), 0, cudaMemcpyHostToDevice));
}

__device__
bool hitDispatch(int objType, int objIdx, const ray& r, float t_min, float t_max, hit_record& rec, curandState* states, int idx) {
	switch (objType) {
		case OBJ_SPHERE:
			return dev_sphere[objIdx].hit(r, t_min, t_max, rec);
			break;

		case OBJ_QUAD:
			return dev_quad[objIdx].hit(r, t_min, t_max, rec);
			break;

		case OBJ_TRANSLATE:
			return dev_translate[objIdx].hit(r, t_min, t_max, rec, states, idx);
			break;

		case OBJ_ROTATE_Y:
			return dev_rotate_y[objIdx].hit(r, t_min, t_max, rec, states, idx);
			break;

		case OBJ_CONSTANT_MEDIUM:
			return dev_constant_medium[objIdx].hit(r, t_min, t_max, rec, states, idx);
			break;

		case OBJ_HITTABLE_LIST:
			return dev_hittable_list[objIdx].hit(r, t_min, t_max, rec, states, idx);
			break;
	}

	return false;
}

__device__
aabb getBboxInfo(int objType, int objIdx) {
	switch (objType) {
		case OBJ_SPHERE:
			return dev_sphere[objIdx].bbox;
			break;

		case OBJ_QUAD:
			return dev_quad[objIdx].bbox;
			break;

		case OBJ_TRANSLATE:
			return dev_translate[objIdx].bbox;
			break;

		case OBJ_ROTATE_Y:
			return dev_rotate_y[objIdx].bbox;
			break;

		case OBJ_CONSTANT_MEDIUM:
			return dev_constant_medium[objIdx].bbox;
			break;

		case OBJ_HITTABLE_LIST:
			return dev_hittable_list[objIdx].bbox;
			break;
	}
}

__host__
aabb host_getBboxInfo(int objType, int objIdx, const world_objects& objs) {
	switch (objType) {
		case OBJ_SPHERE:
			return objs.host_sphere[objIdx].bbox;
			break;

		case OBJ_QUAD:
			return objs.host_quad[objIdx].bbox;
			break;

		case OBJ_TRANSLATE:
			return objs.host_translate[objIdx].bbox;
			break;

		case OBJ_ROTATE_Y:
			return objs.host_rotate_y[objIdx].bbox;
			break;

		case OBJ_CONSTANT_MEDIUM:
			return objs.host_constant_medium[objIdx].bbox;
			break;

		case OBJ_HITTABLE_LIST:
			return objs.host_hittable_list[objIdx].bbox;
			break;
		}
}

__device__
float pdfValueDispatch(int objType, int objIdx, const point3& origin, const vec3& direction) {
	switch (objType) {
		case OBJ_SPHERE:
			return dev_sphere[objIdx].pdf_value(origin, direction);
			break;
		case OBJ_QUAD:
			return dev_quad[objIdx].pdf_value(origin, direction);
			break;
		case OBJ_HITTABLE_LIST:
			return dev_hittable_list[objIdx].pdf_value(origin, direction);
			break;
	}

	return 0.0;
}

__device__
vec3 randomDispatch(int objType, int objIdx, const point3& origin, curandState* states, int idx) {
	switch (objType) {
		case OBJ_SPHERE:
			return dev_sphere[objIdx].random(origin, states, idx);
			break;
		case OBJ_QUAD:
			return dev_quad[objIdx].random(origin, states, idx);
			break;
		case OBJ_HITTABLE_LIST:
			return dev_hittable_list[objIdx].random(origin, states, idx);
			break;
	}

	return vec3(1, 0, 0);
}

__host__
int compare_by_axis(const hittable_list& list, int obj1, int obj2, const world_objects& objs, int axis) {
	int obj_types[2] = { list.obj_types[obj1], list.obj_types[obj2] };
	int obj_idxs[2] = { list.obj_idxs[obj1], list.obj_idxs[obj2] };
	aabb objs_aabbs[2];

	for (int i = 0; i < 2; i++) {
		objs_aabbs[i] = host_getBboxInfo(obj_types[i], obj_idxs[i], objs);
	}

	if (objs_aabbs[0].axis(axis).imin < objs_aabbs[1].axis(axis).imin) {
		return -1;
	}

	else if (objs_aabbs[0].axis(axis).imin > objs_aabbs[1].axis(axis).imin) {
		return 1;
	}

	return 0;
}

__host__
void swap_objects(int obj_type, int obj_idx_1, int obj_idx_2, world_objects& objs) {
	objs.swap(obj_type, obj_idx_1, obj_idx_2);
	return;
}

#endif