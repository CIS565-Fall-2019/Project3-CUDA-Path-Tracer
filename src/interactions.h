#pragma once

#include "intersections.h"
#include "pathtrace.h"


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

__host__ __device__ float Fresnel(float cosThetaI, float etaI, float etaT)
{
	cosThetaI = glm::clamp(cosThetaI, -1.0f, 1.0f);

	bool entering = cosThetaI > 0.0f;
	float etaIb = etaI;
	float etaTb = etaT;
	if (!entering) {
		etaIb = etaT;
		etaTb = etaI;
		cosThetaI = glm::abs(cosThetaI);
	}

	float sinThetaI = glm::sqrt(glm::max(0.0f, 1 - cosThetaI * cosThetaI));
	float sinThetaT = etaIb / etaTb * sinThetaI;

	if (sinThetaT >= 1) {
		return 1.0f;
	}

	float cosThetaT = glm::sqrt(glm::max(0.0f, 1 - sinThetaT * sinThetaT));

	float Rparl = ((etaTb * cosThetaI) - (etaIb * cosThetaT)) /
		((etaTb * cosThetaI) + (etaIb * cosThetaT));
	float Rperp = ((etaIb * cosThetaI) - (etaTb * cosThetaT)) /
		((etaIb * cosThetaI) + (etaTb * cosThetaT));
	return (Rparl * Rparl + Rperp * Rperp) / 2;
}

__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
#if GLTF
        const MyMaterial &m,
#else
		const Material &m,
#endif
        thrust::default_random_engine &rng) {
#if GLTF
	glm::vec3 newDir = calculateRandomDirectionInHemisphere(normal, rng);
	pathSegment.ray.direction = newDir;
	pathSegment.ray.origin = intersect + newDir * 0.0001f;
#else
	if (m.hasReflective > 0) {
		if (m.hasRefractive > 0) {
			thrust::uniform_real_distribution<float> u01(0, 1);
			float cosWo = glm::dot(-pathSegment.ray.direction, normal);
			float fresnel;
			if (cosWo < 0.0f) {
				fresnel = Fresnel(cosWo, m.indexOfRefraction, 1.0f) / (-cosWo);
			}
			else {
				fresnel = Fresnel(cosWo, 1.0f, m.indexOfRefraction) / (cosWo + 0.0001f);
			}

			if (u01(rng) < fresnel) {
				pathSegment.color *= m.specular.color;
				glm::vec3 newDir = glm::reflect(pathSegment.ray.direction, normal);
				pathSegment.ray.direction = newDir;
				pathSegment.ray.origin = intersect + newDir * 0.001f;
			}
			else {
				glm::vec3 wo = pathSegment.ray.direction;
				float eta;
				if (glm::dot(wo, normal) > 0.0f) {
					normal = -normal;
					eta = m.indexOfRefraction;
				}
				else {
					eta = 1.0f / m.indexOfRefraction;
				}
				glm::vec3 newDir = glm::refract(wo, normal, eta);
				if (glm::length(newDir) < .01f) {
					//pathSegment.color *= 0.0f;
					newDir = glm::reflect(wo, normal);
				}
				pathSegment.color *= m.specular.color;
				pathSegment.ray.direction = newDir;
				pathSegment.ray.origin = intersect + newDir * 0.001f;
			}
		}
		else {
			pathSegment.color *= m.specular.color;
			glm::vec3 newDir = glm::reflect(pathSegment.ray.direction, normal);
			pathSegment.ray.direction = newDir;
			pathSegment.ray.origin = intersect + newDir * 0.001f;
		}
	}
	else {
		pathSegment.color *= m.color;
		glm::vec3 newDir = calculateRandomDirectionInHemisphere(normal, rng);
		pathSegment.ray.direction = newDir;
		pathSegment.ray.origin = intersect + newDir * 0.001f;
	}
#endif
}
