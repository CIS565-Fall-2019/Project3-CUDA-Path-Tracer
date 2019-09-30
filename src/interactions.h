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
	thrust::uniform_real_distribution<float> u01(0, 1);
	float samplingProbability = u01(rng);
	int operationToPerform = (samplingProbability > m.hasReflective + m.hasRefractive) ? 1 : 0;
	operationToPerform += (samplingProbability > m.hasReflective) ? 1 : 0;


	pathSegment.remainingBounces -= 1;
	if (samplingProbability > m.hasReflective + m.hasRefractive) {
		pathSegment.ray.direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
		pathSegment.color *= m.color;
	}
	else {
		if (samplingProbability > m.hasReflective) {
			bool goingOutside = glm::dot(normal, pathSegment.ray.direction) > 0.0f;
			glm::vec3 newDirection = glm::refract(glm::normalize(pathSegment.ray.direction),
				(goingOutside ? -1.0f : 1.0f) * glm::normalize(normal),
				(goingOutside ? m.indexOfRefraction : 1.0f / m.indexOfRefraction));

			if (glm::length(newDirection) < 0.01f) {
				newDirection = glm::reflect(pathSegment.ray.direction, normal);
			}
			float r0 = powf((m.indexOfRefraction - 1.0f) / (1.0f + m.indexOfRefraction), 2.0f);
			float r = r0 + (1 - r0) * powf(1 - max(0.0f, glm::dot(pathSegment.ray.direction, normal)), 5);
			newDirection = r < u01(rng) ? glm::reflect(pathSegment.ray.direction, normal) : newDirection;
			pathSegment.ray.direction = glm::normalize(newDirection);
			pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
			pathSegment.color *= m.specular.color;
		}
		else {
			pathSegment.ray.direction = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
			pathSegment.ray.origin = intersect + pathSegment.ray.direction * 0.001f;
			pathSegment.color *= m.specular.color;
		}
	}
}


__device__
void directLight(
	int num_lights,
	Light * lights,
	int num_geoms,
	Geom * geoms,
	Material* materials,
	glm::vec3 intersect,
	glm::vec3 normal,
	PathSegment & pathSegment,
	thrust::default_random_engine &rng) {
	// TODO: implement this.
	// A basic implementation of pure-diffuse shading will just call the
	// calculateRandomDirectionInHemisphere defined above.
	thrust::uniform_real_distribution<float> u01(0, 1);
	int randomLightIndex = (int)(u01(rng) * (num_lights - 1));
	Light  & light = lights[randomLightIndex];
	glm::vec3 pointOnLight = glm::vec3(light.geom.transform * glm::vec4(u01(rng) - 0.5f, u01(rng) -0.5f, u01(rng) - 0.5f, 1.0f));
	Ray ray;
	ray.direction = glm::normalize(pointOnLight - intersect);
	ray.origin = intersect =  0.001f * ray.direction;
	for (int i = 0; i < num_geoms; i++)
	{
		Geom & geom = geoms[i];
		float t = 0.0f;
		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		bool outside = true;

		if (geom.type == CUBE)
		{
			t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
		}
		else if (geom.type == SPHERE)
		{
			t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
		}
		if (t > 0.0f && glm::distance(pointOnLight, intersect) > t)
		{
			return;
		}
	}

	Material material = materials[light.geom.materialid];
	glm::vec3 materialColor = material.color;
	pathSegment.color *= (materialColor * material.emittance * glm::abs(glm::dot(ray.direction, normal)));

}
