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
float schlick(float cosine, float ref_idx) {
    float r0 = powf((1.f - ref_idx) / (1.f + ref_idx), 2.f);
    return r0 + (1.f - r0) * powf((1.f - cosine), 5.f);
}
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {

	thrust::uniform_real_distribution<float> u01(0, 1);
	float prob = u01(rng);
	if (prob < m.hasRefractive) {
		//Refract
		glm::vec3 inDirection = pathSegment.ray.direction;
		//Dot is positive if normal & inDirection face the same dir -> the ray is inside the object getting out
		bool insideObj = glm::dot(inDirection, normal) > 0.0f;
		
		//glm::refract (followed raytracing.github.io trick for hollow glass sphere effect by reversing normals)
		float eta = insideObj ? m.indexOfRefraction : (1.0f / m.indexOfRefraction);
		glm::vec3 outwardNormal = insideObj ? -1.0f * normal : normal;
		glm::vec3 finalDir = glm::refract(glm::normalize(inDirection), glm::normalize(outwardNormal), eta);

		//Check for TIR (if magnitude of refracted ray is very small)
		if (glm::length(finalDir) < 0.01f) {
			pathSegment.color *= 0.0f;
			finalDir = glm::reflect(inDirection, normal);
		}

		//Use schlicks to calculate reflective probability (also followed raytracing.github.io)
		float cosine = glm::dot(inDirection, normal);
		float reflectProb = schlick(cosine, m.indexOfRefraction);
		float sampleFloat = u01(rng);

		pathSegment.ray.direction = reflectProb < sampleFloat ? glm::reflect(inDirection, normal) : finalDir;
		pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
		pathSegment.color *= m.specular.color;
	}
	else if (prob < m.hasReflective) {
		//Reflective Surface
		pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, normal);
		pathSegment.ray.origin = intersect + 0.01f * normal;
		pathSegment.color *= m.specular.color;
	}
	else {
		//Diffuse Surface
		pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.ray.origin = intersect + EPSILON * normal;
	}
	pathSegment.remainingBounces--;
	pathSegment.color *= m.color;
	pathSegment.color = glm::clamp(pathSegment.color, glm::vec3(0.0f), glm::vec3(1.0f));
}