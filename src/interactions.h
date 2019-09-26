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

__host__ __device__ void imperfectSpecularReflection(
	PathSegment & pathSegment,
	const glm::vec3 intersect,
	const glm::vec3 normal,
	const Material &m,
	thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	float shininess = m.specular.exponent;

	// Use Importance sampling to find the reflected vector
	// Get random vector based on the reflective value
	float st = acos(powf(u01(rng), 1.0f / (shininess + 1.0f))); // Spectral Theta
	float sp = 2.0f * PI * u01(rng); // Spectral Psi
	float cosPsi = cos(sp);
	float sinPsi = sin(sp);
	float cosTheta = cos(st);
	float sinTheta = sin(st);
	glm::vec3 sample(cosPsi*sinTheta, sinPsi*sinTheta, cosTheta);

	// We now have a sample, orient it to the reflected vector.
	// https://stackoverflow.com/questions/20923232/how-to-rotate-a-vector-by-a-given-direction
	glm::vec3 reflected = glm::reflect(pathSegment.ray.direction, normal);
	glm::vec3 transform_z = glm::normalize(reflected);
	glm::vec3 transform_x = glm::normalize(glm::cross(transform_z, glm::vec3(0.0f, 0.0f, 1.0f)));
	glm::vec3 transform_y = glm::normalize(glm::cross(transform_z, transform_x));
	glm::mat3 transform = glm::mat3(transform_x, transform_y, transform_z);

	// Transform the vector so that it aligns with the reflected vector as Z axis
	pathSegment.ray.direction = transform * sample;
	pathSegment.color *= m.specular.color;
	pathSegment.ray.origin = intersect + (.001f) * pathSegment.ray.direction;
}

__host__ __device__ void specularRefraction(
	PathSegment & pathSegment,
	const glm::vec3 intersect,
	const glm::vec3 normal,
	const Material &m,
	thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec3 new_normal = glm::vec3(0);
	glm::vec3 old_dir = glm::vec3(0);
	glm::vec3 new_dir = glm::vec3(0);

	// Specular Refraction/Transmission
	float etaA = 1; // Air
	float etaB = m.indexOfRefraction;

	// Determine which is incident and which is transmitted
	// by looking at direction of neg normal.
	// Vectors are facing same direction if dot product is positive'
	float eta = 0.0f;
	old_dir = pathSegment.ray.direction;
	bool isLeaving = glm::dot(old_dir, normal) > 0;
	if (isLeaving) {
		new_normal = -normal;
		eta = etaB / etaA;
	}
	else {
		new_normal = normal;
		eta = etaA / etaB;
	}

	new_dir = glm::refract(old_dir, new_normal, eta);

	// glm::refract will return a zero vector for total internal reflection
	if (glm::length(new_dir) < EPSILON) {
		pathSegment.color = glm::vec3(0.0f); // Set black for internal reflection (do was I was told)
		new_dir = glm::reflect(old_dir, normal);
	}

	pathSegment.ray.direction = new_dir;
	pathSegment.color *= m.specular.color;
	pathSegment.ray.origin = intersect + (.001f) * pathSegment.ray.direction;

	//// Now check against total internal reflection
	//float sinTheta = eta * (sqrtf(1 - (cosTheta * cosTheta)));
	//if (sinTheta >= 1) {
	//	// Internel reflection means just reflect.
	//	pathSegment.color = glm::vec3(0.0f);
	//	//direction_out = getImperfectSpecularRay(pathSegment, intersect, new_normal, m, rng);
	//	direction_out = glm::refract(pathSegment.ray.direction, new_normal, eta);
	//}
	//else {
	//	// Not total internal reflection, so now we try Shlick's Approximation
	//	// Calculate R_Theta based on Schlick's approximation
	//	// R_Theta tells us the ratio of the amplitude of the reflecrted wave
	//	// to the incident wave.
	//	// We only deal with one ray at a time, so we use it to pick either reflection
	//	// or refraction at random.
	//	float r0 = powf((1 - eta) / (1 + eta), 2);
	//	float r_theta = r0 + (1 - r0)*powf((1 - glm::abs(glm::dot(pathSegment.ray.direction, new_normal))), 5.0f);
	//	bool isRefracted = 1;// r_theta < u01(rng);
	//	if (isRefracted) {
	//		direction_out = glm::normalize(glm::refract(pathSegment.ray.direction, new_normal, eta));
	//	}
	//	else {
	//		//direction_out = getImperfectSpecularRay(pathSegment, intersect, new_normal, m, rng);
	//		direction_out = glm::reflect(pathSegment.ray.direction, new_normal);
	//	}
	//	pathSegment.color *= m.specular.color;
	//}
}

__host__ __device__ void idealDiffuse(
	PathSegment & pathSegment,
	const glm::vec3 intersect,
	const glm::vec3 normal,
	const Material &m,
	thrust::default_random_engine &rng)
{
	pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
	pathSegment.color *= m.color;
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

	float totalProb = fmaxf(m.hasReflective + m.hasRefractive, 1);
	float reflectiveProb = m.hasReflective / totalProb;
	float refractiveProb = m.hasRefractive / totalProb;
	float diffuseProb = 1 - reflectiveProb - refractiveProb;
	float rand = u01(rng);

	// Imperfect Specular Reflection
	if (reflectiveProb > rand) {
		imperfectSpecularReflection(pathSegment, intersect, normal, m, rng);	
	}
	else if (refractiveProb > rand) {
		specularRefraction(pathSegment, intersect, normal, m, rng);
	}
	// Ideal Diffuse
	else {
		idealDiffuse(pathSegment, intersect, normal, m, rng);
	}

	// No matter what, new origin is intersect point.
	pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
	pathSegment.remainingBounces--;
}
