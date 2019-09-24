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
glm::vec3 refract(const glm::vec3 &I, const glm::vec3 &N, float eta)
{
	/*float cosi = glm::clamp(glm::dot(I, N), -1.0f, 1.0f);
	float etai = 1, etat = ior;
	glm::vec3 n = N;
	if (cosi < 0) { cosi = -cosi; }
	else { std::swap(etai, etat); n = -N; }
	float eta = etai / etat;
	float k = 1 - eta * eta * (1 - cosi * cosi);
	return k < 0 ? glm::vec3(0) : eta * I + (eta * cosi - sqrtf(k)) * n;*/
	float cosi = glm::clamp(glm::dot(N, I), -1.0f, 1.0f);
	return (I * eta - N * (-cosi + eta * cosi));
}
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
		const ShadeableIntersection &intersection,
        const Material &m,
        thrust::default_random_engine &rng) {
	glm::vec3 dir = pathSegment.ray.direction;
	glm::vec3 color(1.0f, 1.0f, 1.0f);
	thrust::uniform_real_distribution<float> u01(0, 1);
	float pdf = u01(rng);
	if (pdf < m.hasReflective) {
		dir = glm::normalize(glm::reflect(dir, intersection.surfaceNormal));
		color = m.specular.color;
	}
	else if (pdf  - m.hasReflective < m.hasRefractive) {
		if (!intersection.is_inside)
			dir = glm::normalize(glm::refract(pathSegment.ray.direction, intersection.surfaceNormal, 1/m.indexOfRefraction));
		else
			dir = glm::normalize(glm::refract(pathSegment.ray.direction, intersection.surfaceNormal, m.indexOfRefraction));
		color = m.color;
	}
	else {
		dir = glm::normalize(calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng));
		color = m.color;
	}
	pathSegment.ray.direction = dir;
	pathSegment.ray.origin = intersection.intersect + dir * EPSILON;
	pathSegment.color = pathSegment.color * color;// glm::clamp(pathSegment.color * color, glm::vec3(0.0f), glm::vec3(1.0f));
}
