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
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	glm::vec3 originalDirection = pathSegment.ray.direction;

	thrust::uniform_real_distribution<float> u01(0, 1);
	float p = u01(rng);



	if (p <= m.hasReflective) {
		// Reflection
		pathSegment.ray.direction = glm::normalize(glm::reflect(originalDirection, normal));
		pathSegment.ray.origin = intersect + EPSILON * normal;
		pathSegment.color *= m.specular.color / m.hasReflective;
	}
	else if (p <= m.hasReflective + m.hasRefractive) {
		//bool entering = pathSegment.ray.direction.z > 0;
		glm::vec3 rayDir = glm::normalize(originalDirection);
		glm::vec3 refractedDir = glm::vec3();
		glm::vec3 offsetOrigin = glm::vec3();
		float eta = 1.f / m.indexOfRefraction;

		if (glm::dot(rayDir, glm::normalize(normal)) > 0.0) {
			// Inside sphere leaving
			float theta = acos(glm::dot(normal, rayDir));
			float critAngle = asin((float)1.f / eta);
			offsetOrigin = intersect + EPSILON * normal;
			refractedDir = glm::normalize(glm::refract(rayDir, -normal, eta));
			if ((float)1.f / eta > 1 && theta < critAngle) {
				refractedDir = glm::normalize(glm::reflect(rayDir, normal));
			}
			/*refractedDir = glm::vec3(0.0, 1.0, 0.0);
			offsetOrigin = glm::vec3(1.0);*/
		}
		else {
			// outside sphere entering
			float theta = acos(glm::dot(normal, -rayDir));
			float critAngle = asin(eta);
			offsetOrigin = intersect + EPSILON * (-normal);
			refractedDir = glm::normalize(glm::refract(rayDir, -normal, 1.f/eta));
			if (eta < 1 && theta > critAngle) {
				refractedDir = glm::normalize(glm::reflect(rayDir, normal));
			}
		}
		pathSegment.ray.origin = offsetOrigin;
		pathSegment.ray.direction = refractedDir;
		//pathSegment.color *= m.specular.color;

	}
	else {
		pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		pathSegment.ray.origin = intersect + EPSILON * normal;

		float scale = m.hasReflective >= 1 ? 0 : 1.0 / (1.0 - m.hasReflective);
		pathSegment.color *= m.color * scale;
	}




	
}
