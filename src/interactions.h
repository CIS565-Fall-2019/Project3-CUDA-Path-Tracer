#pragma once

#include "intersections.h"


#define INTENSITY 1 // to make things more shiny
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

__host__ __device__ glm::vec3 imperfect_mirror(glm::vec3 perfectMirror, thrust::default_random_engine &rng, float exponent)
{
	thrust::uniform_real_distribution<float> u01(0, 1);

	float costheta = powf(u01(rng), (1.0 / (exponent + 1)));//is this op expensive?
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

	return costheta * perfectMirror
		+ cos(phi) * sintheta * perpendicularDirection1
		+ sin(phi) * sintheta * perpendicularDirection2;
}

__host__ __device__ glm::vec3 compute_reflection(PathSegment& path, glm::vec3 normal, float t,const Material& m, thrust::default_random_engine &rng)
{
	glm::vec3 old_ray_origin = path.ray.origin;
	glm::vec3 old_ray_direction = path.ray.direction;
	//glm::vec3 old_color = path.color;
	float exponent = m.specular.exponent;
	// 
	float shininess = (m.hasReflective * INTENSITY);

	// based off of our surface normal and ray direction we want to reflect the path
	// https://glm.g-truc.net/0.9.4/api/a00131.html#gabe1fa0bef5f854242eb70ce56e5a7d03
	// glm reflect handles where the new ray shoots out
	glm::vec3 new_ray_direction = glm::normalize(glm::reflect(old_ray_direction,normal));

	// need to add some epsilon according to slides od I add to ray or color?
	// think colo
	// http://web.cse.ohio-state.edu/~shen.94/681/Site/Slides_files/reflection_refraction.pdf

	//if not perfect mirror
	//impoerfect_mirror(new_ray_direction, rng, m.specular.color);


	// we want to accumulate some more color.
	//old_color *= m.specular.color;

	// set our new color
	//path.color = old_color ;

	// compute our new ray origin
	//path.ray.origin = (old_ray_origin + old_ray_direction * t) + (new_ray_direction * EPSILON); // the is some floating error TA said add this
	path.ray.direction = new_ray_direction;
	return m.specular.color * shininess;
}

__host__ __device__ glm::vec3 compute_refraction(PathSegment& path, glm::vec3 normal, float t,const Material& m)
{
	glm::vec3 old_ray_origin = path.ray.origin;
	glm::vec3 old_ray_direction = path.ray.direction;
	glm::vec3 color = m.color; // unless we have total internal reflection we will return the material color;
	glm::vec3 new_ray_direction;
	
	float eta; // essentially n1/n2 or n2/n1 depending on persepctive of ray
	
	// notes from https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
	// for refraction we have some corner cases we need to handle.
	// maybe negative?
	float cos_theta1 = glm::dot( old_ray_direction, -normal);
	//normal = glm::faceforward(normal, old_ray_direction, normal);
	if (cos_theta1 < 0.f) {
		// we are outside the surface, we want cos(theta) to be positive
		// we are outside moving towards a surface
		//normal = -normal;
		cos_theta1 = -cos_theta1;
		eta = m.indexOfRefraction; // we are doing pretty much air over our index of refraction.. airs index ~= 1.
	}
	else {
		// we are inside the surface, cos(theta) is already positive but reverse normal direction
		// we are inside the material moving towards air
		normal = -normal;
		//cos_theta1 = -cos_theta1;
		//assert(0);
		eta = 1/m.indexOfRefraction; // material index over air index
	}

	// I think there was some rounding errors... dot product was producing values of 1.00000012
	// so camp our values....
	cos_theta1 = (cos_theta1 > 1) ? 1 : cos_theta1;

	// list of trig identities for making sense of fresnels and snells
	// https://en.wikipedia.org/wiki/Snell%27s_law
	// cos theta1 = -surface vector dot light vector; //
	// sin theta2 = ( (n1/n2) * sin theta1 ) = (n1/n2) * (sqrt ( 1 - cos_theta1^2))  
	// cos theta2 = (sqrt ( 1 - sin theta2^2))  
	// vrefract = (n1/n2) * ray + ( ( n1/n2 * cos theta1 ) - cos theta2  ) * nomal
	// normal must be positive 
	float sin_theta2 = (eta) * (sqrtf(1 - (cos_theta1 * cos_theta1)));
	assert((1 - (cos_theta1 * cos_theta1) >= 0)); // make sure no NaN's

	
	// total internal reflection so just reflect
	if (sin_theta2 >= 1)
	{
		//assert(0);
		// return 0
		//new_ray_direction = glm::reflect(old_ray_direction, normal);
		//color = m.specular.color;
		new_ray_direction = glm::refract(old_ray_direction, normal, eta);
		color = glm::vec3(0.f);

	}
	
	else 
	{
		// do more trig identity ... gross 
		//float cos_theta2 = sqrtf(1 - (sin_theta2 * sin_theta2)); // may need to check if negative
		//assert((1 - (sin_theta2 * sin_theta2) >= 0)); // make sure no NaN's
		new_ray_direction = glm::refract(old_ray_direction, normal, eta);
		// finally refract
		//new_ray_direction = (eta * old_ray_direction) + (((eta * cos_theta1) - cos_theta2) * normal); // cos theta1 and normal must be positive ... this is guaranteed by statemetns above
		//new_ray_direction *= -1;
		color = m.color;
	}

	// compute our new ray origin
	//path.ray.origin = (old_ray_origin + old_ray_direction * t) + (new_ray_direction * EPSILON); // the is some floating error TA said add this
	path.ray.direction = glm::normalize(new_ray_direction);
	return color;
}

__host__ __device__ glm::vec3 compute_diffuse(PathSegment& path, glm::vec3 normal, float t, const Material& m, thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	glm::vec3 old_ray_origin = path.ray.origin;
	glm::vec3 old_ray_direction = path.ray.direction;
	glm::vec3 color;
	glm::vec3 new_ray_direction;

	float rand = u01(rng);

	float probdiffuse = glm::length(m.color);
	float probspec = glm::length(m.specular.color);
	float total = probdiffuse + probspec;
	//printf("diffuse %f : spec %f \n", probdiffuse, probspec);

	probdiffuse /= total;

	// diffuse bounce is random specular is a reflect
	if (rand < probdiffuse )
	{
		new_ray_direction = glm::normalize(calculateRandomDirectionInHemisphere(normal, rng));
		assert(probdiffuse <= 1.f && probdiffuse >= 0);
		color = m.color / probdiffuse;
		//printf("diffuse %f : spec %f color %f rand %f \n", probdiffuse, probspec, m.color, rand);
	}
	else
	{
		// based off of our surface normal and ray direction we want to reflect the path
		new_ray_direction = glm::normalize(glm::reflect(old_ray_direction, normal));
		assert(probspec <= 1.f && probspec >= 0);
		color = m.specular.color / probspec;
		//printf("diffuse %f : spec %f color %f \n", probdiffuse, probspec, m.color);
	}

	// compute our new ray origin
	//path.ray.origin = (old_ray_origin + old_ray_direction * t) + (new_ray_direction *EPSILON);
	path.ray.direction = glm::normalize(new_ray_direction);
	//return color/cost;
	return color;
}

__host__ __device__ float fresnels(glm::vec3 normal, glm::vec3 old_ray_direction,const Material& material)
{
	float eta_air = 1; // n1
	float eta_mat = material.indexOfRefraction; //n2 
	float Fresnels_number;
	float eta; // n1/n2 or n2/n1
	float n1;
	float n2;
	// list of trig identities for making sense of fresnels and snells
	// https://en.wikipedia.org/wiki/Snell%27s_law
	// cos theta1 = -surface vector dot light vector; //
	// sin theta2 = ( (n1/n2) * sin theta1 ) = (n1/n2) * (sqrt ( 1 - cos_theta1^2))  
	// cos theta2 = (sqrt ( 1 - sin theta2^2))  

	float cos_theta1 = glm::dot(-normal, old_ray_direction);
	//normal = glm::faceforward(normal, old_ray_direction, normal);
	if (cos_theta1 > 0.f) {
		// we are outside the surface, we want cos(theta) to be positive
		// we are outside moving towards a surface
		//cos_theta1 = -cos_theta1;
		normal = -normal;
		eta = material.indexOfRefraction; // we are doing pretty much air over our index of refraction.. airs index ~= 1.
		n1 = eta_mat;
		n2 = eta_air;
	}
	else {
		// we are inside the surface, cos(theta) is already positive but reverse normal direction
		// we are inside the material moving towards air
		cos_theta1 = -cos_theta1;
		// normal = -normal;
		//assert(0);
		eta = 1/material.indexOfRefraction; // material index over air index
		n1 = eta_air;
		n2 = eta_mat;
	}

	// I think there was some rounding errors... dot product was producing values of 1.00000012
	// so camp our values....
	cos_theta1 = (cos_theta1 > 1) ? 1 : cos_theta1;

	// do more trig identity ... gross
	float sin_theta2 = (eta) * (sqrtf( 1 - (cos_theta1 * cos_theta1) ) );
	assert((1 - (cos_theta1 * cos_theta1) >= 0)); // make sure no NaN's
	
	// total internal reflection
	if (sin_theta2 >= 1)
	{
		Fresnels_number = 1;
		return Fresnels_number;
	}
	else
	{
		// do more trig identity ... gross 
		float cos_theta2 = sqrtf( 1 - (sin_theta2 * sin_theta2) ); // cant use std in device code TODO // check that aboce 0
		assert((1 - (sin_theta2 * sin_theta2) >= 0)); // make sure no NaN's

		// this is fresnels equation finally
		float Fresnel_R_Parallel = ((n2 * cos_theta1) - (n1 * cos_theta2)) / ((n2 * cos_theta1) + (n1 * cos_theta2));
		float Fresnel_R_Perp = ((n1 * cos_theta2) - (n2 * cos_theta1)) / ((n1 * cos_theta2) + (n2 * cos_theta1));
		// 
		Fresnels_number = ( (Fresnel_R_Parallel * Fresnel_R_Parallel) + (Fresnel_R_Perp * Fresnel_R_Perp) ) / 2;
		return Fresnels_number;
	}

}

// we come here when we have a material that is both reflective AND refractive
// based off of fresenels equation we follow the route of reflection or refraction
// resources: https://computergraphics.stackexchange.com/questions/2482/choosing-reflection-or-refraction-in-path-tracing
// https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
__host__ __device__ glm::vec3 compute_fresnels(PathSegment& path, glm::vec3 normal, float t, const Material& m, thrust::default_random_engine &rng)
{
	thrust::uniform_real_distribution<float> u01(0, 1);
	float rand = u01(rng);
	glm::vec3 color;

	// fresnelse number needs to be added to the color like an intensity
	// fresnels number gives us how much light is reflected. the higher the number the higher the reflectivity
	float Fresnels_Number = fresnels(normal,path.ray.direction,m);
	assert(Fresnels_Number >= 0); // make sure its reasonable

	//float probrefract = m.hasRefractive;
	//float probreflect = m.hasReflective;
	//float total = probrefract + probreflect;
	//probrefract /= total;

	// the lower the fresnels number the more 
	if ( Fresnels_Number < rand )
	{
		color = compute_refraction(path, normal, t, m);
		Fresnels_Number = 1 - Fresnels_Number;
		//printf(" refract fresenels %f color.x y z %f %f %f \n", Fresnels_Number, color.x, color.y, color.z);
		assert(Fresnels_Number >= 0);
	}
	else
	{
		color = compute_reflection(path, normal, t, m,rng);
		//printf(" reflect fresenels %f color.x y z %f %f %f \n", Fresnels_Number, color.x, color.y, color.z);
		//color *= (m.hasReflective * 500);
	}

	//assert(Fresnels_Number >= 0 && Fresnels_Number < 1); // make sure its reasonable
	return Fresnels_Number * color;
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
		float t,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
	// calculateRandomDirection or depending on what we want to call
	// is diffuse is reflective is refractive. is opaque? etc etc
	glm::vec3 color;

	// TODO add concept of "shininess" 
	if (m.hasReflective > 0.0f && m.hasRefractive > 0.0f)
	{
		color = compute_fresnels(pathSegment, normal, t, m,rng);
	}
	// if reflective
	else if (m.hasReflective > 0.0f)
	{
		color = compute_reflection(pathSegment, normal, t, m,rng);
	}
	//if refractive
	else if (m.hasRefractive > 0.0f) 
	{
		color = compute_refraction(pathSegment, normal, t, m);
	}
	// else diffuse
	else{
		color = compute_diffuse(pathSegment, normal, t, m, rng);
	}

	pathSegment.ray.origin = intersect;
	pathSegment.color *= color; 
}
