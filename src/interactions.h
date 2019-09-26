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

/*
* Refract pathSegment according to the material m and surface normal 
* normal at the intersection point
*/
__host__ __device__ glm::vec3 refract(
	PathSegment & pathSegment,
	glm::vec3 normal,
	const Material &m,
	thrust::default_random_engine &rng) {
	// Check if ray is coming from inside or outside of the material it struck
	thrust::uniform_real_distribution<float> u01(0, 1);
	float R = 1.0f; // Refraction Probability
	float ior;
	glm::vec3 sur_normal;

	float cosi = glm::dot(pathSegment.ray.direction, normal);
	if (cosi > 0.0f) {
		ior = m.indexOfRefraction;
		sur_normal = normal * -1.0f;
	}
	else {
		ior = 1.0f / m.indexOfRefraction;
		sur_normal = normal;
	}
	// Check for Total Internal Reflection
	float sinr = (1.0f / m.indexOfRefraction) * sqrtf(1 - powf(cosi, 2));

	// Ray is coming from inside and TIR
	if (cosi > 0.0f && sinr > 1.0f) {
		R = 0.0f;
		pathSegment.color *= 0;
	}
	else {
		// Schlick's Approximation for fresnel's effects
		float r0 = powf((1.0f - ior) / (1.0f + ior), 2.0f);
		//printf("Schlick screwing up!! r0: %f R: %f\n", r0, fmaxf(0.0f, cosi));
		R = r0 + ((1 - r0) * powf((1 - fmaxf(0.0f, cosi)), 5));
		pathSegment.color *= m.specular.color;
	}
	float r_num = u01(rng);
	if (r_num < R) {
		return glm::refract(pathSegment.ray.direction, sur_normal, ior);
	}
	else {
		return glm::reflect(pathSegment.ray.direction, normal);
	}
}

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
void scatterRay(
		PathSegment & pathSegment,
		glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec3 scattered_ray_direction;
	// Split according to probability
	float rand_num = u01(rng);
	// Reflection
	if (rand_num < m.hasReflective) {
		scattered_ray_direction = glm::reflect(pathSegment.ray.direction, normal);
		pathSegment.color *= m.specular.color;
	}
	// Refraction
	else if (rand_num < m.hasReflective + m.hasRefractive) {
		scattered_ray_direction = refract(pathSegment, normal, m, rng);
	}
	// Diffusion
	else {
		scattered_ray_direction = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.color *= m.color;
	}
	pathSegment.ray.origin = intersect + scattered_ray_direction *0.01f;
	pathSegment.ray.direction = scattered_ray_direction;
}
