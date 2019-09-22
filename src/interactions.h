#pragma once

#include "intersections.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
gvec3 calculateRandomDirectionInHemisphere(glm::vec3 normal, 
							thrust::default_random_engine &rng) {

    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrtf(u01(rng)); // cos(theta)
    float over = sqrtf(1 - up * up); // sin(theta)
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
}//calculateRandomInHemisphere

__host__ __device__ gvec3 calculateShinyDirection(gvec3 incoming, gvec3 normal, float exponent,
												thrust::default_random_engine& rng) {
	gvec3 perfectMirror = REFLECT(incoming, normal);//will be adding an offset to this

	thrust::uniform_real_distribution<float> u01(0, 1);

	float costheta = powf(u01(rng), (1.0 / (exponent + 1)));//this op may be expensive
	float sintheta = sqrtf(1.0 - costheta * costheta);
	float phi = TWO_PI * u01(rng);
	/*
	gvec3 offset = gvec3(sintheta * cosf(phi),
						sintheta * sinf(phi),
						costheta);//random direction off of z-axis "reflect-vector"
	*/

	glm::vec3 directionNotNormal;
	if (abs(perfectMirror.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(1, 0, 0);
	}
	else if (abs(perfectMirror.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = glm::vec3(0, 1, 0);
	}
	else {
		directionNotNormal = glm::vec3(0, 0, 1);
	}
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(perfectMirror, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(perfectMirror, perpendicularDirection1));

	return costheta * normal
		+ cos(phi) * sintheta * perpendicularDirection1
		+ sin(phi) * sintheta * perpendicularDirection2;

}//calculateShinyDirection

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
		PathSegment& pathSegment,
        gvec3 intersect,
        gvec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {

	thrust::uniform_real_distribution<float> u01(0, 1);
	float branchRandom = u01(rng);
	float probDiff = glm::length(m.color);
	float probSpec = glm::length(m.specular.color);

	if (probDiff + probSpec < EPSILON) {
		pathSegment.color = gvec3(0.0f, 0.0f, 0.0f);
		pathSegment.remainingBounces = 0;
		return;
	}//if some jackass put a black color in the scene

	//else, probabilistically choose between diffuse/specular
	probDiff /= (probDiff + probSpec);
	probSpec /= (probDiff + probSpec);
	//these now sum to 1

	if (branchRandom < probDiff) {//DIFFUSE
		gvec3 newDirection = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.ray = Ray{ intersect,  newDirection };
		pathSegment.color *= m.color;
	}
	else {//SPECULAR (either mirror or otherwise)
		if (m.hasReflective > 0) {//MIRROR
			gvec3 newDirection = REFLECT(pathSegment.ray.direction, normal);
			pathSegment.ray = Ray{ intersect, newDirection };
		}//if
		else {//NON-MIRROR SPECULAR
			gvec3 newDirection = calculateShinyDirection(pathSegment.ray.direction, normal, m.specular.exponent, rng);
			pathSegment.ray = Ray{ intersect, newDirection };
		}//else
		pathSegment.color *= m.specular.color;
	}

}//scatterRay
