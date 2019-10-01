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

__host__ __device__ glm::vec3 trowbridgeReitzDistribution(const glm::vec3 &wo, float roughness, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);
	float x = u01(rng);
	float y = u01(rng);
	glm::vec3 wh;
	float cosTheta = 0;
	float phi = (2 * 3.1415926) * y;
	float tanTheta2 = roughness * roughness * x / (1.0f - x);
	cosTheta = 1.0 / std::sqrtf(1 + tanTheta2);
	float sinTheta = std::sqrtf(std::fmax((float)0., (float)1. - cosTheta * cosTheta));

	wh = glm::vec3(sinTheta * std::cosf(phi), sinTheta * std::sinf(phi), cosTheta);
	if (!(wo.z * wh.z > 0)) wh = -wh;

	return wh;
}

__host__ __device__
glm::vec3 calculateRandomDirectionInLobe(glm::vec3 normal, glm::vec3 incomingDir, float exponent, thrust::default_random_engine &rng) {
	//trowbridgeReitz doesn't work. lets fake it
	glm::vec3 perfectSpec = glm::reflect(incomingDir, normal);

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

	glm::vec3 perfectDiffuse = up * normal
		+ cos(around) * over * perpendicularDirection1
		+ sin(around) * over * perpendicularDirection2;

	float ratio = std::powf(u01(rng), exponent);

	return perfectDiffuse * ratio + perfectSpec * (1.f - ratio);
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
		PathSegment *pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
		glm::vec2 uv,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	bool doReflective = false;
	bool doRefractive = false;
	if (m.hasReflective && m.hasRefractive) {
		float fresnel = std::abs(glm::dot(pathSegment->ray.direction, normal));
		thrust::uniform_real_distribution<float> u01(0, 1);
		if (u01(rng) < fresnel) {
			doRefractive = true;
		}
		else {
			doReflective = true;
		}
	}
	else if (m.hasReflective) {
		doReflective = true;
	}
	else if (m.hasRefractive) {
		doRefractive = true;
	}

	if (doReflective) {
		pathSegment->ray.direction = glm::reflect(pathSegment->ray.direction, normal);
		pathSegment->color *= m.specular.color;
	}
	else if (doRefractive) {
		glm::vec3 newDir;
		float dotRayNorm = glm::dot(pathSegment->ray.direction, normal);
		float eta;
		if (dotRayNorm == 0) {
			newDir = glm::reflect(pathSegment->ray.direction, normal);
		}
		else {
			if (dotRayNorm < 0) {
				eta = 1.0f / m.indexOfRefraction;
			}
			else if (dotRayNorm > 0) {
				eta = m.indexOfRefraction / 1.0f;
			}

			newDir = glm::refract(pathSegment->ray.direction, normal, eta);

			if (glm::length(newDir) < 0.00001) {
				newDir = glm::reflect(pathSegment->ray.direction, normal);
			}
		}
			
		pathSegment->ray.direction = newDir;
		pathSegment->color *= m.specular.color;
	}
	else {
		if (m.specular.strength == 0) {
			pathSegment->ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(glm::normalize(normal), rng));
			if (m.hasDiffuseMap == 1) {
				pathSegment->color *= glm::vec3(uv, 0);
			}
			else {
				pathSegment->color *= m.color;
			}
		}
		else {
			thrust::uniform_real_distribution<float> u01(0, 1);
			if (u01(rng) < m.specular.strength) {
				pathSegment->ray.direction = glm::normalize(calculateRandomDirectionInLobe(glm::normalize(normal), pathSegment->ray.direction, m.specular.exponent, rng));
				pathSegment->color *= m.specular.color;
			}
			else {
				pathSegment->ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(glm::normalize(normal), rng));
				pathSegment->color *= m.color;
			}
		}
	}
	pathSegment->ray.origin = intersect + 0.001f * pathSegment->ray.direction;
	pathSegment->remainingBounces--;
}
