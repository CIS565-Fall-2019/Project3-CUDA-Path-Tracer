#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
	glm::vec3 normal, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	float up = sqrt(u01(rng)); // cos(theta)
	float over = sqrt(1 - up * up); // sin(theta)
	float around = u01(rng) * TWO_PI;// 2 * pi * r^2-> 0.5 superficial area

	// Find a direction that is not the normal based off of whether or not the
	// normal's components are all equal to sqrt(1/3) or whether or not at
	// least one component is less than sqrt(1/3). Learned this trick from
	// Peter Kutz.

	glm::vec3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}

	// Use not-normal direction to generate two perpendicular directions
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;
}

__host__ __device__ float lambert_bsdf_f(glm::vec3 wi, glm::vec3 wo) {
	return InvPi;
}

__host__ __device__ float lambert_bsdf_pdf(glm::vec3 n, glm::vec3 wi) {
	return AbsDot(wi, n) * InvPi;
}

#if DIRECT_LIGHT
__device__ __host__ float cubeArea(const Geom *light) {
	return 2 * light->scale.x * light->scale.y *
		2 * light->scale.z * light->scale.y *
		2 * light->scale.x * light->scale.z;
}

//just consider the bottom
__host__ __device__ glm::vec3 sampleCube(const Geom *light,
	thrust::default_random_engine &rng,
	float &pdf) {

	thrust::uniform_real_distribution<float> u01(0, 1);
	float x = u01(rng), z = u01(rng);

	//0-1 -> -0.5-0.5
	glm::vec3 p = glm::vec3(x - 0.5f, -0.5f, z - 0.5f);

	pdf = 1.0f / (light->scale.x * light->scale.z);
	return glm::vec3(light->transform * glm::vec4(p, 1.0f));
}

//direct light
//Sample_Li(&ligh, intersect, normal, rng, &mywi, &pdf);
__host__ __device__ void Sample_Li(
	const Geom *light,
	const glm::vec3 intersect,
	const glm::vec3 normal,
	thrust::default_random_engine &rng,
	const Material &m,
	glm::vec3 &wi, float &pdf) {

	if (light->type == CUBE) {
		glm::vec3 sample_p = sampleCube(light, rng, pdf);
		wi = glm::normalize(sample_p - intersect);
		float dis2 = glm::distance2(sample_p, intersect);
		pdf *= dis2 / AbsDot(normal, wi);
	}
}
#endif

//get intersection cpu
__host__ __device__ bool getIntersection(
	const Ray& ray
	, Geom* geoms
	, const int geoms_size
	, ShadeableIntersection& intersection
	, Triangle *tri
) {
	float t;
	glm::vec3 intersect_point;
	glm::vec3 normal;
	float t_min = FLT_MAX;//infinite
	int hit_geom_index = -1;//the index of the hit geometry
	bool outside = true;

	glm::vec3 tmp_intersect;
	glm::vec3 tmp_normal;

	// naive parse through global geoms
	for (int i = 0; i < geoms_size; i++) {
		Geom & geom = geoms[i];//get it
		//get global space normal and global space intersection 
		if (geom.type == CUBE) {
			t = boxIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == SPHERE) {
			t = sphereIntersectionTest(geom, ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == MESH) {
			t = meshIntersectionTest(geom, tri, ray, tmp_normal);
		}
		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t) {
			t_min = t;//update t
			hit_geom_index = i;
			intersect_point = tmp_intersect;//glo
			normal = tmp_normal;//global
		}
	}
	if (hit_geom_index == -1) {
		intersection.t = -1.0f;//no intersect-> t =-1
		return false;
	}
	else {
		//The ray hits something
		intersection.t = t_min;
		intersection.materialId = geoms[hit_geom_index].materialid;
		intersection.surfaceNormal = normal;//global
		intersection.outside = outside;
		intersection.point = intersect_point;
		return true;
	}
}

//glass is dielectric
__host__ __device__ float FresnelDielectric(
	float cosThetaI, float etaI, float etaT, 
	float &sinThetaI, float &sinThetaT) {

	cosThetaI = glm::clamp(cosThetaI, -1.f, 1.f);
	float etaItemp, etaTtemp;

	etaItemp = etaI;
	etaTtemp = etaT;

	bool entering = cosThetaI > 0.f;
	if (!entering) {//leaving
		float temp = etaItemp;
		etaItemp = etaTtemp;
		etaTtemp = temp;
		cosThetaI = fabs(cosThetaI);
	}

	//snell law
	sinThetaI = std::sqrt(max(0.f, 1.f - cosThetaI * cosThetaI));
	sinThetaT = etaItemp / etaTtemp * sinThetaI;

	float cosThetaT = std::sqrt(max(0.f, 1.f - sinThetaT * sinThetaT));

	float rparl = ((etaTtemp * cosThetaI) - (etaItemp * cosThetaT)) /
		((etaTtemp * cosThetaI) + (etaItemp * cosThetaT));

	float rperp = ((etaItemp * cosThetaI) - (etaTtemp * cosThetaT)) /
		((etaItemp * cosThetaI) + (etaTtemp * cosThetaT));

	float col = (rparl * rparl + rperp * rperp) / 2.f;

	return col;
}

__host__ __device__ float FresnelConductor(float cosTheta,
	const Material &m, float etaI, float etaT, float k) {
	cosTheta = glm::clamp(cosTheta, -1.f, 1.f);

	float eta = etaT / etaI;
	float etak = k / etaI;

	float costhetasquare = cosTheta * cosTheta;
	float sinthetasquare = 1 - costhetasquare;
	float sin4 = sinthetasquare * sinthetasquare;

	float temp = eta * eta - etak * etak - sinthetasquare;
	float ab = std::sqrt(temp * temp + 4 * etak * etak * eta * eta);
	float a = std::sqrt(0.5f * (ab + temp));

	float rparlmole = ab + costhetasquare - 2.f * a * cosTheta;
	float rparldeno = ab + costhetasquare + 2.f * a * cosTheta;

	float rparl = rparlmole / rparldeno;

	float rperpmole = costhetasquare * ab + sin4 - 2 * a * cosTheta * sinthetasquare;
	float rperpdeno = costhetasquare * ab + sin4 + 2 * a * cosTheta * sinthetasquare;

	float rperp = rparl * (rperpmole / rperpdeno);

	float col = (rparl + rperp) / 2.f;

	return col;
}

__host__ __device__ glm::vec3 Faceforward(const glm::vec3 &n, const glm::vec3 &v) {
	return (glm::dot(n, v) < 0.f) ? -n : n;
}

__host__ __device__ float CosTheta(const glm::vec3 &w) { return w.z; }
/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */

__host__ __device__
void scatterRay2(
	PathSegment & pathSegment
	, glm::vec3 intersect
	, glm::vec3 normal
	, const Material &m
	, thrust::default_random_engine &rng

#if DIRECT_LIGHT
	, const Geom* lights
	, int &light_count
	, Geom* geos
	, int geo_count
	, Material * mats
	, Triangle * tri
#endif	
	, bool outside = true) {
	int pixelIndex = pathSegment.pixelIndex;

	if (m.emittance > 0.0f) {//is light
#if DIRECT_LIGHT
		pathSegment.color = (m.color * m.emittance);
#else
		pathSegment.color *= (m.color * m.emittance);
#endif
		pathSegment.remainingBounces = -1;
		return;
	}
	else {// not light
		if (pathSegment.remainingBounces > 0) {//check if is out of bounce
			if ((!m.hasReflective) && (!m.hasRefractive)) {//pure lambert diffuse
#if !DIRECT_LIGHT
				glm::vec3 wo = -pathSegment.ray.direction;
				//generate a wo
				glm::vec3 wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
				Color3f f = m.color * lambert_bsdf_f(wi, wo);
				float pdf = lambert_bsdf_pdf(normal, wi);
				float absdot = AbsDot(wi, normal);
				pathSegment.color *= f * absdot * 1.f / pdf;
				//pathSegment.color *= m.color;
				//set new ray
				pathSegment.ray.origin = intersect + 0.001f * normal;//move a little bit
				pathSegment.ray.direction = wi;
#else
				//get a random light
				thrust::uniform_real_distribution<float> u02(0, light_count);
				int light_choose = glm::floor(u02(rng));
				Geom ligh = lights[light_choose];
				//get the Li
				Color3f L(0.0f);
				glm::vec3 wo = -pathSegment.ray.direction;
				glm::vec3 wi(0.0);
				float pdf = 0.0;

				Color3f Li = mats[ligh.materialid].color * mats[ligh.materialid].emittance;
				Sample_Li(&ligh, intersect, normal, rng, mats[ligh.materialid], wi, pdf);
				wi = glm::normalize(wi);
				pdf = pdf / (1.f * light_count);
				if (pdf <= 0.f) {
					pathSegment.color = glm::vec3(0.0);
					pathSegment.remainingBounces = -1;
					return;
				}

				float abdot = AbsDot(wi, normal);
				Color3f colbsdf = m.color * lambert_bsdf_f(wi, wo);

				float vis = 1.0;
				Ray intertolight;
				intertolight.origin = intersect + 0.01f * normal;
				intertolight.direction = wi;

				ShadeableIntersection sec;
				bool isinter = getIntersection(intertolight, geos, geo_count, sec, tri);
				if (isinter) {//hit!
					if (mats[sec.materialId].emittance <= 0.0) {//not light
						vis = 0.0;
					}
				}
				L += Li * colbsdf * Color3f(vis) * abdot * 1.f / pdf;
				pathSegment.color += Color3f(2.0) * L;
				//printf("%f, %f, %f", L.x, L.y, L.z);
				pathSegment.remainingBounces = -1;
				return;
#endif			
			}
			else if (m.hasReflective) {
				//pure reflective when hasReflective is 1, otherwise, go 50/50
				if (m.hasReflective == 1) {
					glm::vec3 wi = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
					pathSegment.ray.origin = intersect + 0.001f * normal;
					pathSegment.ray.direction = wi;

#if !DIRECT_LIGHT
					//pathSegment.color *= m.specular.color;
#else

#endif
			}
				else {//combination -> reflect + diffuse
					thrust::uniform_real_distribution<float> u01(0, 1);
					float prob = u01(rng);
#ifdef PROB_ACC
					if (prob < m.hasReflective) {//reflect
#else
					if (prob < 0.5) {//reflect
#endif	
						glm::vec3 wo = -pathSegment.ray.direction;
						glm::vec3 wi = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));//get reflect dir
						pathSegment.ray.origin = intersect + 0.001f * normal;//move a little bit
						pathSegment.ray.direction = wi;

						pathSegment.color *= (m.specular.color / 0.5f);
					}
					else {//diffuse
#if !DIRECT_LIGHT
						glm::vec3 wo = -pathSegment.ray.direction;
						glm::vec3 wi = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));

						Color3f bsdf_f = m.color * lambert_bsdf_f(wi, wo);
						float pdf = lambert_bsdf_pdf(normal, wi);
						float absdot = AbsDot(wi, normal);

						//pathSegment.color *= (bsdf_f * absdot * 1.f / pdf) / 0.5f;
						pathSegment.color *= (m.color / 0.5f);
						//pathSegment.color *= m.color;
						pathSegment.ray.direction = wi;
						pathSegment.ray.origin = intersect + 0.001f * normal;
#else
						thrust::uniform_real_distribution<float> u02(0, light_count);
						int light_choose = glm::floor(u02(rng));
						Geom ligh = lights[light_choose];
						//get the Li
						Color3f L(0.0f);
						glm::vec3 wo = -pathSegment.ray.direction;
						glm::vec3 wi(0.0);
						float pdf = 0.0;

						Color3f Li = mats[ligh.materialid].color * mats[ligh.materialid].emittance;
						Sample_Li(&ligh, intersect, normal, rng, mats[ligh.materialid], wi, pdf);
						wi = glm::normalize(wi);
						pdf = pdf / (1.f * light_count);
						if (pdf <= 0.f) {
							pathSegment.color = glm::vec3(0.0);
							pathSegment.remainingBounces = -1;
							return;
						}

						float abdot = AbsDot(wi, normal);
						Color3f colbsdf = m.color * lambert_bsdf_f(wi, wo);

						float vis = 1.0;
						Ray intertolight;
						intertolight.origin = intersect + 0.01f * normal;
						intertolight.direction = wi;

						ShadeableIntersection sec;
						bool isinter = getIntersection(intertolight, geos, geo_count, sec, tri);
						if (isinter) {//hit!
							if (mats[sec.materialId].emittance <= 0.0) {//not light
								vis = 0.0;
							}
						}
						L += Li * colbsdf * Color3f(vis) * abdot * 1.f / pdf;
						pathSegment.color += Color3f(2.0) * L;
						pathSegment.remainingBounces = -1;
						return;
#endif
					}
				}
			}
			else if (m.hasRefractive) {//refract
				glm::vec3 wo = -pathSegment.ray.direction;
				glm::vec3 n = normal;
				float sinThetaI = 0.0, sinThetaT = 0.0;
				float f_btdf = FresnelDielectric(glm::dot(glm::normalize(wo), glm::normalize(n)), 
					1.0, m.indexOfRefraction, sinThetaI, sinThetaT);
				float etaIn = 1.0, etaOut = m.indexOfRefraction;

				thrust::uniform_real_distribution<float> u04(0, 1);
				float prob = u04(rng);
				if (prob < 0.5) { //specular brdf
					glm::vec3 wi = glm::reflect(pathSegment.ray.direction, normal);
					//pathSegment.color *= (m.specular.color * f_btdf) / 0.5f;
					pathSegment.ray.origin = intersect + 0.001f * normal;
					pathSegment.ray.direction = wi;
				} else { //specularbtdf
					glm::vec3 wi = glm::vec3(0.f);
					Color3f f_col(0.0);
					//check if can not go
					if (sinThetaI >= 1.0) {
						wi = glm::reflect(wo, n);
						pathSegment.color *= (f_btdf * m.specular.color) / 0.5f;
					} else {
						wi = glm::refract(wo, n, etaIn / etaOut);
						f_col =  m.specular.color - Color3f(f_btdf);
						pathSegment.color *= f_col / 0.5f;
					}
					//Color3f f = FresnelConductor(cosTheta, m, etaIn, etaOut, float k);
					pathSegment.ray.direction = wi;
					pathSegment.ray.origin = intersect + 0.001f * normal;
				}
			}
		}
		else {//out of bounce!
			return;
		}
	}
	pathSegment.remainingBounces--;
}

