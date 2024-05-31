#ifndef PDF_CUH
#define PDF_CUH

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "vec3.cuh"
#include "onb.cuh"
#include "objects.cuh"

class pdf {
	public:
		__device__
		virtual ~pdf() {}

		__device__
		virtual float value(const vec3& direction) const = 0;
		
		__device__
		virtual vec3 generate(curandState* states, int idx) const = 0;
};

class sphere_pdf : public pdf {
	public:
		__device__
		sphere_pdf() {}

		__device__
		float value(const vec3& direction) const override {
			return 1 / (4 * 3.1415926);
		}

		__device__
		vec3 generate(curandState* states, int idx) const override {
			return random_unit_vector(states, idx);
		}
};

class cosine_pdf : public pdf {
	public:
		__device__
		cosine_pdf(const vec3& w) { uvw.build_from_w(w); }

		__device__
		float value(const vec3& direction) const override {
			float cosine_theta = dot(unit_vector(direction), uvw.w());
			return fmaxf(0, cosine_theta / 3.1415926);
		}

		__device__
		vec3 generate(curandState* states, int idx) const override {
			return uvw.local(random_cosine_direction(states, idx));
		}

	private:
		onb uvw;
};

class hittable_pdf : public pdf {
	public:
		__device__
		hittable_pdf(int _obj_type, int _obj_idx, const point3& _origin): obj_type(_obj_type), obj_idx(_obj_idx), origin(_origin) {}

		__device__
		float value(const vec3& direction) const override {
			return pdfValueDispatch(obj_type, obj_idx, origin, direction);
		}

		__device__
		vec3 generate(curandState* states, int idx) const override {
			return randomDispatch(obj_type, obj_idx, origin, states, idx);
		}


	private:
		int obj_type;
		int obj_idx;
		point3 origin;
};

class mixture_pdf : public pdf {
	public:
		__device__
		mixture_pdf(pdf* p0, pdf* p1) {
			p[0] = p0;
			p[1] = p1;
		}

		__device__
		float value(const vec3& direction) const override {
			return 0.5 * p[0]->value(direction) + 0.5 * p[1]->value(direction);
		}

		__device__
		vec3 generate(curandState* states, int idx) const override {
			if (random_float(states, idx) < 0.5) {
				return p[0]->generate(states, idx);
			}
			else {
				return p[1]->generate(states, idx);
			}
		}

	private:
		pdf* p[2];
};

#endif