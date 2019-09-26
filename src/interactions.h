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
		PathSegment &pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	glm::vec3 new_ray;
	glm::vec3 materialColor;


	thrust::uniform_real_distribution<float> u01(0, 1);
	float prob = u01(rng);

	if (prob <= m.hasReflective) {
		new_ray = glm::reflect(pathSegment.ray.direction, normal);

		materialColor = m.specular.color;
	}
	else if (prob <= (m.hasRefractive + m.hasReflective)) {
		
		glm::vec3 new_normal;
		float ior = m.indexOfRefraction;
		float r;
		//Check if ray is from inside to outside object or other direction
		float cosi = glm::dot(pathSegment.ray.direction, normal);

		float n1, n2;
		if (cosi > 0.0f) {
			//Inside to oustide
			n1 = ior;
			n2 = 1; //Air

			new_normal = -1.0f * normal;

			//printf("QQQQQQQQQQQQQQQQQQQQ - Inside to Outside/n");
			//Check total internal reflection
			float sinr = (n1/ n2) * sqrtf(1 - pow(cosi, 2));
			if (sinr > 1.0f) {
				
				//Total internal reflection
				r = 1;
				materialColor = glm::vec3(0.0f);
			}
			else {
				//Calculate Reflectance (r) using Fresnel's law
				float r0 = pow((n1 - n2) / (n1 + n2), 2);
				r = r0 + (1 - r0) * pow((1 - glm::max(0.0f, cosi)), 5);
				materialColor = m.specular.color;
			}
		}
		else {
			n1 = 1; //Air
			n2 = ior;

			new_normal = normal;

			//printf("Outside to inside\n");
			//Calculate Reflectance (r) using Fresnel's law
			float r0 = pow((n1 - n2) / (n1 + n2), 2);
			r = r0 + (1 - r0) * pow((1 - glm::max(0.0f, cosi)), 5);
			materialColor = m.specular.color;
		}

		//Reflect or refract randomnly based on reflectance ratio
		thrust::uniform_real_distribution<float> u01(0, 1);
		float prob_of_reflection = u01(rng);
		//printf("REFL: %f\n", r);
		if (prob_of_reflection <= (1 - r)) {
			//Reflect
			new_ray = glm::reflect(pathSegment.ray.direction, normal);
		}
		else {
			//Refract
			new_ray = glm::refract(pathSegment.ray.direction, new_normal, n1/n2);
		}
		
	}
	else {
		new_ray = calculateRandomDirectionInHemisphere(normal, rng);

		materialColor = m.color;
	}

	//Update the new ray in place in pathSegment
	pathSegment.ray.origin = intersect + new_ray * 0.01f;
	pathSegment.ray.direction = new_ray;

	//Update the color in place
	pathSegment.color *= materialColor;
}
