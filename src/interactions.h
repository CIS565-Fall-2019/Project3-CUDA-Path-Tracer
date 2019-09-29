#pragma once

#include "intersections.h"
#include <math.h>

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
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
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

__host__ __device__ void diffuseReflection(PathSegment& pathSegment, const glm::vec3& normal, const glm::vec3& intersect,
	const Material& m, thrust::default_random_engine &rng) {
	pathSegment.color *= m.color;// / float(1.0 - m.hasReflective - m.hasRefractive);
	pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
	pathSegment.ray.origin = intersect + (normal * EPSILON);
}

__host__ __device__ void specularReflection(PathSegment& pathSegment, const glm::vec3& normal, const glm::vec3& intersect,
	const Material& m) {
	pathSegment.color *= m.specular.color;// / m.hasReflective;
	pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
	pathSegment.ray.origin = intersect + (normal * EPSILON);
}

// fresnel for a dielectric material
__host__ __device__ float fresnel(const PathSegment& pathSegment, const glm::vec3& normal, const Material& m) {
	float cosThetaI = glm::clamp(glm::dot(-pathSegment.ray.direction, normal), -1.f, 1.f);

	bool entering = cosThetaI > 0.f;
	float etaI = 1.f;
	float etaT = m.indexOfRefraction;

	if (!entering) {
		cosThetaI = glm::abs(cosThetaI);
		float tempEta = etaI;
		etaI = etaT;
		etaT = tempEta;
	}

	float sinThetaI = glm::sqrt(glm::max(0.f, 1.f - cosThetaI * cosThetaI));
	float sinThetaT = etaI / etaT * sinThetaI;

	if (sinThetaT >= 1.f) {
		return 1.f;// / glm::abs(cosThetaI);
	}

	float cosThetaT = glm::sqrt(glm::max(0.f, 1.f - sinThetaT * sinThetaT));

	float r_parallel = ((etaT * cosThetaI) - (etaI * cosThetaT)) / ((etaT * cosThetaI) + (etaI * cosThetaT));
	float r_perp = ((etaI * cosThetaI) - (etaT * cosThetaT)) / ((etaI * cosThetaI) + (etaT * cosThetaT));

	return ((r_parallel * r_parallel + r_perp * r_perp) / 2.f) / glm::abs(cosThetaI);
}

__host__ __device__ void specularRefraction(PathSegment& pathSegment, const glm::vec3& normal, const glm::vec3& intersect, 
	const Material& m) {

	glm::vec3 inDir = glm::normalize(pathSegment.ray.direction);
	glm::vec3 nor = glm::normalize(normal);
	float eta = m.indexOfRefraction;

	if (glm::dot(inDir, nor) < 0) {
		eta = 1.f / eta;
	}
	else {
		nor = -nor;
	}

	// check for total internal reflection
	if (glm::length(pathSegment.ray.direction) < 0.001f) {
		pathSegment.ray.direction = glm::normalize(glm::reflect(inDir, nor));
	}
	else {
		pathSegment.ray.direction = glm::normalize(glm::refract(inDir, nor, eta));

	}

	pathSegment.color *= m.specular.color;
	pathSegment.ray.origin = intersect + (nor * EPSILON);



}

__host__ __device__ void glass(PathSegment& pathSegment, const glm::vec3& normal, const glm::vec3& intersect, 
	const Material& m, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	glm::vec3 newColor(0.f);
	glm::vec3 newDir(1.f);
	/*
	float cosTheta = glm::dot(pathSegment.ray.direction, normal);
	bool entering = cosTheta < 0.f;
	glm::vec3 newNormal = entering ? normal : -normal;
	float eta = entering ? (1.f / m.indexOfRefraction) : m.indexOfRefraction;
	glm::vec3 refractDir = glm::refract(pathSegment.ray.direction, newNormal, eta);
	glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, normal);

	newDir = refractDir;

	// check for total internal reflection
	if (glm::length(newDir) < 0.0001f) {
		pathSegment.color *= 0.f;
		newDir = reflectDir;
	}*/

	float cosTheta = glm::dot(pathSegment.ray.direction, normal);
	bool entering = cosTheta < 0.f;
	glm::vec3 newNormal = entering ? normal : -normal;
	float eta = entering ? (1.f / m.indexOfRefraction) : m.indexOfRefraction;
	//float eta = entering ? m.indexOfRefraction : (1.f / m.indexOfRefraction);
	glm::vec3 refractDir = glm::refract(pathSegment.ray.direction, newNormal, eta);
	glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, normal);

	newDir = refractDir;

	// check for total internal reflection
	if (glm::length(newDir) < 0.0001f) {
		pathSegment.color *= 0.f;
		//newDir = reflectDir;
	}

	//float ft = (1.0 - fresnel(pathSegment, normal, m));

	// schlick's approximation
	float num = entering ? (1.f - m.indexOfRefraction) : m.indexOfRefraction - 1.f;
	float R_0 = (1.f - m.indexOfRefraction) / (1.0 + m.indexOfRefraction);
	//float R_0 = num / (1.0 + m.indexOfRefraction);
	R_0 = R_0 * R_0; // squared
	float R_theta = R_0 + (1.0 - R_0) * powf(1.f - glm::abs(cosTheta), 5.f);

	//newDir = R_theta < u01(rng) ? reflectDir : newDir;

	pathSegment.color *= m.specular.color;
	pathSegment.ray.direction = glm::normalize(newDir);
	pathSegment.ray.origin = intersect + (newNormal * EPSILON);

	//pathSegment.color *= m.color;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 */
__host__ __device__ void scatterRay(PathSegment & pathSegment, glm::vec3 intersect, 
	glm::vec3 normal, const Material &m, thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.

	thrust::uniform_real_distribution<float> u01(0, 1);
	float probability = u01(rng);

	// assume m.hasReflective + m.hasRefractive + probability of being diffuse = 1
	if (probability < m.hasReflective) { 
		// specular reflective
		// new direction is old direction of ray reflected across surface normal
		specularReflection(pathSegment, normal, intersect, m);
	}
	else if (probability < m.hasRefractive + m.hasReflective) {
		//specularRefraction(pathSegment, normal, intersect, m);

		float f = fresnel(pathSegment, normal, m);
		
		if (u01(rng) < f) {
			specularReflection(pathSegment, normal, intersect, m);
			//specularRefraction(pathSegment, normal, intersect, m);
		}
		else {
			//specularReflection(pathSegment, normal, intersect, m);
			specularRefraction(pathSegment, normal, intersect, m);
		}
	}
	else {
		// diffuse
		diffuseReflection(pathSegment, normal, intersect, m, rng);
	}

	// update color
	//float lambertian = glm::abs((glm::dot(glm::normalize(normal), glm::normalize(pathSegment.ray.direction)))); // uhh should this be old or new direction...
	//pathSegment.color *= newColor;// *lambertian;

	// update ray
	//pathSegment.ray.direction = glm::normalize(newDir);
	//pathSegment.ray.origin = intersect + (normal * EPSILON);
}