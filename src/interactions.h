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

__host__ __device__ void compute_reflection(PathSegment& path, glm::vec3 normal, float t,const Material& m)
{
	glm::vec3 old_ray_origin = path.ray.origin;
	glm::vec3 old_ray_direction = path.ray.direction;
	glm::vec3 old_color = path.color;

	// based off of our surface normal and ray direction we want to reflect the path
	// https://glm.g-truc.net/0.9.4/api/a00131.html#gabe1fa0bef5f854242eb70ce56e5a7d03
	// glm reflect handles where the new ray shoots out
	glm::vec3 new_ray_direction = glm::reflect(old_ray_direction,normal);

	// we want to accumulate some more color.
	old_color *= m.specular.color;

	// set our new color
	path.color = old_color;

	// compute our new ray origin
	path.ray.origin = (old_ray_origin + old_ray_direction * t) + (new_ray_direction *.001f); // the is some floating error TA said add this
	path.ray.direction = new_ray_direction;
	return;
}

__host__ __device__ void compute_refraction(PathSegment& path, glm::vec3 normal, float t,const Material& m)
{
	glm::vec3 old_ray_origin = path.ray.origin;
	glm::vec3 old_ray_direction = path.ray.direction;
	glm::vec3 old_color = path.color;

	// need to add logic for total internal reflection

	// based off of our surface normal and ray direction we want to reflect the path
	glm::vec3 new_ray_direction = glm::refract(normal, old_ray_direction,m.indexOfRefraction);

	// we want to compute our new color, which for a reflection stays the same.
	old_color = m.color;

	path.color = old_color;// glm::max(old_color, glm::vec3(0.0f));

	// compute our new ray origin
	path.ray.origin = (old_ray_origin + old_ray_direction * t) + (new_ray_direction * .001f); // the is some floating error TA said add this
	path.ray.direction = new_ray_direction;
	return;
}

__host__ __device__ void compute_diffuse(PathSegment& path, glm::vec3 normal, float t, const Material& m, thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec3 old_ray_origin = path.ray.origin;
	glm::vec3 old_ray_direction = path.ray.direction;
	glm::vec3 old_color = path.color;

	// based off of our surface normal and ray direction we want to reflect the path
	glm::vec3 new_ray_direction =glm::normalize( calculateRandomDirectionInHemisphere(normal, rng) );

	float rand = u01(rng);

	// specular bounce?
	if (rand > .5f )
	{
		new_ray_direction = glm::normalize(old_ray_direction);
	}

	//float cos = glm::dot(normal, (new_ray_direction));
	//float denom = cos / glm::pi<float>();

	//float denom = rand;

	// we want to compute our new color, which for a reflection stays the same.
	//old_color *= m.color;

	glm::vec3 c = m.color;
		
	//c*= cos * glm::one_over_pi<float>();

	old_color *= c ;

	path.color = old_color;//glm::max(old_color, glm::vec3(0.f));

	// compute our new ray origin
	path.ray.origin = (old_ray_origin + old_ray_direction * t) + (new_ray_direction *.001f); // the is some floating error TA said add this
	path.ray.direction = new_ray_direction;
	return;
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
        glm::vec3 normal,
        const Material &m,
		float t,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	// calculateRandomDirection or depending on what we want to call
	// is diffuse is reflective is refractive. is opaque? etc etc


	if (m.hasReflective > 0.0f && m.hasRefractive > 0.0f)
	{
		assert(0); // not yet implemented
	}
	// if reflective
	else if (m.hasReflective > 0.0f)
	{
		compute_reflection(pathSegment, normal, t, m);
	}
	//if refractive
	else if (m.hasRefractive > 0.0f) 
	{
		compute_refraction(pathSegment, normal, t, m);
		printf("refract\n");
		assert(0); // not yet implemented
	}
	// else diffuse
	else{
		compute_diffuse(pathSegment, normal, t, m, rng);
	}
}
