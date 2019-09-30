#pragma once

#include "intersections.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <texture_types.h>
#define ECONST 	2.71828182845904523536

/**
This may end up with performance improvements over the glm implementation; giving my compiler a chance to compete
*/
__host__ __device__ gvec3 normalize(const gvec3 vector) {
	float mag = sqrtf(vector.x * vector.x + vector.y * vector.y + vector.z * vector.z);
	return gvec3(vector.x / mag, vector.y / mag, vector.z / mag);
}//normalize

/**
Reflects an incoming vector by the normal
Note: does not normalize the output
*/
__host__ __device__ gvec3 reflectIncomingByNormal(const gvec3 incoming, const gvec3 normal) {
	return incoming - 2 * DOTP(incoming, normal) * normal;
}//reflectIncomingByNormal

__host__ __device__ gvec3 refractIncomingByNormal(const gvec3 incoming, const gvec3 normal, const float iorInc, const float iorMat, bool* wentThrough) {
	//float critAngle = asinf(iorMat / iorInc);
	float3 incomingf = { incoming.x, incoming.y, incoming.z };
	float cosIncTheta = DOTP(-1.0 * incomingf, normal);
	float sinIncTheta = 1.0 - cosIncTheta * cosIncTheta;
	if (sinIncTheta >= iorMat / iorInc) {
		*wentThrough = false;
		return reflectIncomingByNormal(incoming, normal);
	}// if greater than or equal to the critical angle (then, reflect)
	else {
		*wentThrough = true;
		float c = cosIncTheta;
		float r = iorInc / iorMat;
		gvec3 refract = r * incoming + (r * c - sqrtf(1 - r * r * (1 - c * c))) * normal;
		float3 refractf = { refract.x, refract.y, refract.z };
		return gvec3(refractf.x, refractf.y, refractf.z);
	}//else
}//refractIncomingByNormal

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
	//gvec3 perfectMirror = REFLECT(incoming, normal);//will be adding an offset to this
	gvec3 perfectMirror = normalize(reflectIncomingByNormal(incoming, normal));//will be adding an offset to this

	thrust::uniform_real_distribution<float> u01(0, 1);

	float costheta = powf(u01(rng), (1.0 / (exponent + 1)));//is this op expensive?
	float sintheta = sqrtf(1.0 - costheta * costheta);
	float phi = TWO_PI * u01(rng);
	/*
	gvec3 offset = gvec3(sintheta * cosf(phi),
						sintheta * sinf(phi),
						costheta);//random direction off of z-axis "reflect-vector"
	*/

	gvec3 directionNotNormal;
	if (abs(perfectMirror.x) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = gvec3(1, 0, 0);
	}
	else if (abs(perfectMirror.y) < SQRT_OF_ONE_THIRD) {
		directionNotNormal = gvec3(0, 1, 0);
	}
	else {
		directionNotNormal = gvec3(0, 0, 1);
	}
	gvec3 perpendicularDirection1 =
		normalize(glm::cross(perfectMirror, directionNotNormal));
	gvec3 perpendicularDirection2 =
		normalize(glm::cross(perfectMirror, perpendicularDirection1));

	return costheta * perfectMirror
		+ cos(phi) * sintheta * perpendicularDirection1
		+ sin(phi) * sintheta * perpendicularDirection2;

}//calculateShinyDirection

#if TEX_NORM
__device__ gvec3 modifyNormalWithMap(const gvec3 normal, float2 uv, cudaTextureObject_t myTexture) {

	float4 normText = tex2DLayered<float4>(myTexture, uv.x, uv.y, TEXTURE_LAYER_NORMAL);

	float costheta = normText.z;
	float sintheta = sqrtf(1.0 - costheta * costheta);
	//float phi = atan2f(normMap.y, normMap.x);

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
	glm::vec3 perpendicularDirection1 =
		glm::normalize(glm::cross(normal, directionNotNormal));
	glm::vec3 perpendicularDirection2 =
		glm::normalize(glm::cross(normal, perpendicularDirection1));

	return costheta * normal
		//+ cos(phi) * sintheta * perpendicularDirection1
		+ normText.x * sintheta * perpendicularDirection1
		//+ sin(phi) * sintheta * perpendicularDirection2;
		+ normText.y * sintheta * perpendicularDirection2; 

}//modifyNormalWithMap
#endif

__device__
gvec3 getMaterialColor(
		const Material& m,
		float2 uv,
		cudaTextureObject_t myTexture) {
	gvec3 color = m.color;
#if TEX_COLOR
	if (m.textureMask & TEXTURE_BASECOLOR) {
		float4 colorText = tex2DLayered<float4>(myTexture, uv.x, uv.y, TEXTURE_LAYER_BASECOLOR);
		color = gvec3(colorText.x, colorText.y, colorText.z);
	}
#endif
	

	return color;
}

/**
Maps our "roughness" coefficients to something usable within our specular reflections
This is ENTIRELY made up
Goals:
0 maps to some number of thousads
0.5 maps to 32ish??
1.0 maps to 1
*/
__device__ float roughnessToExponent(float roughness) {
	float invRoughness = 1.0 - roughness;
	float scaledInvRoughness = (expf(invRoughness) - 1) / (ECONST - 1);
	return powf(2.0, scaledInvRoughness * 12);
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
__device__
void scatterRay(
	PathSegment& segment,
	gvec3 intersect,
	gvec3* normal,
	const Material& m,
	bool leavingMaterial,
	float2 uv,
	cudaTextureObject_t textureReference,
	thrust::default_random_engine& rng) {

#if TEX_NORM
	//modify normal if we have a normal map
	if (m.textureMask & TEXTURE_NORMAL) {
		*normal = modifyNormalWithMap(*normal, uv, textureReference);
	}
#endif
	gvec3 finalNormal = *normal;

	gvec3 reverseIncoming = segment.ray.direction;
	reverseIncoming *= -1;
	float lightTerm = DOTP(finalNormal, reverseIncoming);
	

	thrust::uniform_real_distribution<float> u01(0, 1);
	float branchRandom = u01(rng);
	float probDiff = glm::length(m.color) * (1.0 - m.hasRefractive);
	float probSpec = glm::length(m.specular.color) * (1.0 - m.hasRefractive);
	float probMirror = probSpec * m.hasReflective;
	probSpec *= (1.0 - m.hasReflective);

	float totalProbBounce = probDiff + probSpec + probMirror;

	if (totalProbBounce + m.hasRefractive < EPSILON) {
		segment.color = gvec3(0.0f, 0.0f, 0.0f);
		segment.remainingBounces = 0;
		return;
	}//if some jackass put a totally black color in the scene

	//else, probabilistically choose between diffuse/specular
	probDiff /= totalProbBounce;
	probSpec /= totalProbBounce;
	probMirror /= totalProbBounce;
	probDiff *= (1.0 - m.hasRefractive);
	probSpec *= (1.0 - m.hasRefractive);
	probMirror *= (1.0 - m.hasRefractive);
	//these now sum to 1

	float exponent = m.specular.exponent;
	gvec3 diffColor = getMaterialColor(m, uv, textureReference);
	gvec3 specColor = m.specular.color;
	//segment.color *= lightTerm;//scale by that costheta

#if TEX_ROUGH
	if (m.textureMask & TEXTURE_METALLICROUGHNESS) {
		float4 roughText = tex2DLayered<float4>(textureReference, uv.x, uv.y, TEXTURE_LAYER_METALLICROUGHNESS);
		float ambientOcclusion = roughText.x;
		float roughness = roughText.y;
		float metalness = roughText.z;

		specColor = diffColor;
		exponent = roughnessToExponent(roughness);

		probMirror = 0;
		probDiff = (1.0 - metalness) * (1.0 - m.hasRefractive);
		probSpec = metalness * (1.0 - m.hasRefractive);
	}
#endif

	if (leavingMaterial) branchRandom = 1.0;//force refractive on the way out

	if (branchRandom < probDiff) {
		gvec3 newDirection = calculateRandomDirectionInHemisphere(finalNormal, rng);
		segment.ray = Ray{ intersect /*+ EPSILON * newDirection*/,  newDirection };
		segment.color *= diffColor;
		segment.color *= lightTerm;//scale by that costheta
	}//if diffuse
	else if (branchRandom < probDiff + probSpec) {
		gvec3 newDirection = calculateShinyDirection(segment.ray.direction, finalNormal, exponent, rng);
		segment.ray = Ray{ intersect /*+ EPSILON * newDirection*/, newDirection };
		segment.color *= specColor;
		//segment.color *= lightTerm;//scale by that costheta
	}//else if specular
	else if (branchRandom < probDiff + probSpec + probMirror) {
		//gvec3 newDirection = REFLECT(pathSegment.ray.direction, normal);
		gvec3 newDirection = normalize(reflectIncomingByNormal(segment.ray.direction, finalNormal));
		segment.ray = Ray{ intersect /*+ EPSILON * newDirection*/, newDirection };
		segment.color *= specColor;
		//segment.color *= lightTerm;//scale by that costheta
	}//else if mirror
	else {
		bool wentThrough = false;
		float ior1, ior2;
		if (leavingMaterial) {
			ior1 = segment.curIOR;
			ior2 = 1.0;
		}//if
		else {
			ior1 = segment.curIOR;
			ior2 = m.indexOfRefraction;
		}

		gvec3 newDirection = normalize(refractIncomingByNormal(segment.ray.direction, finalNormal, ior1, ior2, &wentThrough));
		
		segment.color *= specColor;
		if (wentThrough) {
			segment.ray = Ray{ intersect - EPSILON * finalNormal, newDirection };
			if (leavingMaterial) segment.curIOR = 1.0;
			else segment.curIOR = m.indexOfRefraction;
		}
		else {
			segment.ray = Ray{ intersect, newDirection };
		}//just bounced
		//segment.color *= lightTerm;//scale by that costheta
	}//else if refractive

}//scatterRay
