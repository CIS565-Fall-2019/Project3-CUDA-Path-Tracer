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

/////// PART 2 //////

// BETTER HEMISPHERE SAMPLING - Cosine weighted - more rays orthogonal to surface

// Helper function - also used for depth of field
__host__ __device__
glm::vec3 squareToDiskConcentric(const glm::vec2* sample) {
	/* Function based on A Low Distortion Map Between Disk and Square
	* by Peter Shirley and Kenneth Chiu */

	float theta, r, u, v;
	// Remapping samples' [0,1] to [-1,1]
	float a = 2.f * sample->x - 1.f;
	float b = 2.f * sample->y - 1.f;

	// The upper-right-most samples
	if (a > -b) {
		// The rightmost samples
		if (a > b) {
			r = a;
			theta = (3.141592653589793238462643 / 4.f) * (b / a);
			// The upper-most samples
		}
		else {
			r = b;
			theta = (3.141592653589793238462643 / 4.f) * (2.f - (a / b));
		}
		// The bottom-left-most samples
	}
	else {
		// The leftmost samples
		if (a < b) {
			r = -a;
			theta = (3.141592653589793238462643 / 4.f) * (4.f + (b / a));
			// The bottommost samples and when denometers = 0
		}
		else {
			r = -b;
			if (b != 0.f) {
				theta = (3.141592653589793238462643 / 4.f) * (6.f - (a / b));
			}
			else {
				theta = 0.f;
			}
		}
	}

	u = r * cos(theta);
	v = r * sin(theta);
	return glm::vec3(u, v, 0.f);
}

// If we flatly project the disk disribution from above onto a hemisphere, there
// is less of a chance for outgoing rays tangent to the surface
__host__ __device__
glm::vec3 calculateCosineDirectionInHemisphere(
	glm::vec3 normal, thrust::default_random_engine &rng) {
	thrust::uniform_real_distribution<float> u01(0, 1);

	glm::vec2 sample = glm::vec2(u01(rng), u01(rng));
	glm::vec3 disk = squareToDiskConcentric(&sample);
	disk.z = sqrt(1.f - disk.x * disk.x - disk.y * disk.y);

	// Creating a set of axes to form the basis of a coordinate system
	// given a single vector normal. - Similar to Peter Kutz's strategy above,
	// but from PBR-book
	glm::vec3 perpendicularDirection1; // tangent
	glm::vec3 perpendicularDirection2; // bitangent

	if (glm::abs(normal.x) > glm::abs(normal.y)) {
		perpendicularDirection1 = glm::vec3(-normal.z, 0, normal.x) /
			std::sqrt(normal.x * normal.x + normal.z * normal.z);
	}
	else {
		perpendicularDirection1 = glm::vec3(0, normal.z, -normal.y) /
			std::sqrt(normal.y * normal.y + normal.z * normal.z);
	}
	perpendicularDirection2 = glm::cross(normal, perpendicularDirection1);

	return glm::mat3(perpendicularDirection1, perpendicularDirection2, normal) *
		disk; // swap normal and bitangent? up is y in this one
}


// NOISE FOR TEXTURE OPTIONS - from my 566 assignments
__host__ __device__
glm::vec4 permute(glm::vec4 &x) { return glm::mod(((x * 34.f) + 1.f) * x, 289.f); }
__host__ __device__
glm::vec4 taylorInvSqrt(glm::vec4 &r) { return 1.79284291400159f - 0.85373472095314f * r; }

__host__ __device__
float snoise(glm::vec3 v) {
	const glm::vec2  C = glm::vec2(1.0 / 6.0, 1.0 / 3.0);
	const glm::vec4  D = glm::vec4(0.0, 0.5, 1.0, 2.0);
	glm::vec3 Cxxx = glm::vec3(C.x, C.x, C.x);

	// First corner
	glm::vec3 i = floor(v + dot(v, glm::vec3(C.y, C.y, C.y)));
	glm::vec3 x0 = v - i + dot(i, Cxxx);

	// Other corners
	glm::vec3 g = step(glm::vec3(x0.y, x0.z, x0.x), x0);
	glm::vec3 l = 1.f - g;
	glm::vec3 i1 = min(g, glm::vec3(l.z, l.x, l.y));
	glm::vec3 i2 = max(g, glm::vec3(l.z, l.x, l.y));

	//  x0 = x0 - 0. + 0.0 * C 
	glm::vec3 x1 = x0 - i1  + 1.f * Cxxx;
	glm::vec3 x2 = x0 - i2  + 2.f * Cxxx;
	glm::vec3 x3 = x0 - 1.f + 3.f * Cxxx;

	// Permutations
	i = glm::mod(i, 289.f);
	glm::vec4 p = permute(permute(permute(
		i.z   + glm::vec4(0.0, i1.z, i2.z, 1.0))
		+ i.y + glm::vec4(0.0, i1.y, i2.y, 1.0))
		+ i.x + glm::vec4(0.0, i1.x, i2.x, 1.0));

	// Gradients
	// ( N*N points uniformly over a square, mapped onto an octahedron.)
	float n_ = 1.0 / 7.0; // N=7
	glm::vec3  ns = n_ * glm::vec3(D.w, D.y, D.z) - glm::vec3(D.x, D.z, D.x);

	glm::vec4 j = p - 49.f * floor(p * ns.z *ns.z);  //  mod(p,N*N)

	glm::vec4 x_ = floor(j * ns.z);
	glm::vec4 y_ = floor(j - 7.f * x_);    // mod(j,N)

	glm::vec4 x = x_ * ns.x + glm::vec4(ns.y, ns.y, ns.y, ns.y);
	glm::vec4 y = y_ * ns.x + glm::vec4(ns.y, ns.y, ns.y, ns.y);
	glm::vec4 h = 1.f - abs(x) - abs(y);

	glm::vec4 b0 = glm::vec4(x.x, x.y, y.x, y.y);
	glm::vec4 b1 = glm::vec4(x.z, x.w, y.z, y.w);

	glm::vec4 s0 = floor(b0) * 2.f + 1.f;
	glm::vec4 s1 = floor(b1) * 2.f + 1.f;
	glm::vec4 sh = -step(h, glm::vec4(0.0));

	glm::vec4 a0 = glm::vec4(b0.x, b0.z, b0.y, b0.w) + 
				   glm::vec4(s0.x, s0.z, s0.y, s0.w) * 
				   glm::vec4(sh.x, sh.x, sh.y, sh.y);
	glm::vec4 a1 = glm::vec4(b1.x, b1.z, b1.y, b1.w) + 
				   glm::vec4(s1.x, s1.z, s1.y, s1.w) *
				   glm::vec4(sh.z, sh.z, sh.w, sh.w);

	glm::vec3 p0 = glm::vec3(a0.x, a0.y, h.x);
	glm::vec3 p1 = glm::vec3(a0.z, a0.w, h.y);
	glm::vec3 p2 = glm::vec3(a1.x, a1.y, h.z);
	glm::vec3 p3 = glm::vec3(a1.z, a1.w, h.w);

	//Normalise gradients
	glm::vec4 norm = taylorInvSqrt(glm::vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;

	// Mix final noise value
	glm::vec4 m = max(0.6f - glm::vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), glm::vec4(0.f));
	m = m * m;
	return 42.0 * dot(m*m, glm::vec4(dot(p0, x0), dot(p1, x1),
		dot(p2, x2), dot(p3, x3)));
}

__host__ __device__
glm::vec3 mix3(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, float f) {
	if (f < 0.6) {
		return mix(v1, v2, f * 1.666666666f);
	}
	else {
		return mix(v2, v3, (f - 0.6) * 2.5f);
	}
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

	glm::vec3 currCol = m.color;
	glm::vec3 currSpecCol = m.specular.color;

	// Pt 2: Procedural textures
	if (m.hasWave) { // The wavy one
		float wavText = sqrt(sqrt(abs(snoise(intersect * 5.f)))) * 0.6f + 
						sqrt(sqrt(abs(snoise(intersect * 2.f)))) * 0.4f;
		glm::vec3 alter = mix3(glm::vec3(1.f, 1.f, 1.f),
							   glm::vec3(199.f / 255.f, 0.f / 255.f, 255.f / 255.f), 
						       glm::vec3(0.f, 0.15f, 0.75f), wavText);
		currCol *= alter;
		currSpecCol *= alter;
	
	} else if (m.hasNoise) { // The splotchy one
		float noise = (snoise(intersect) + 1.f) / 2.f;
		currCol.y *= noise;
		currCol.z = 1.f - noise;
	}

	// Pt 2: Refractive
	// FUNCTION INSPIRED from PBR-book
	if (m.hasRefractive) { 
		glm::vec3 dir = glm::normalize(pathSegment.ray.direction);
		float cosThetaI = glm::dot(normal, -dir); // Had a lot of bugs, but after 
												  // scrambling many +/- signs, I got 
												  // the present result
		float eta = 1.00029f; // air IOR

		// Figure out which  is incident and which is transmitted
		//  - Entering / inside transmissive object:
		if (cosThetaI > 0.f) {		 
			eta /= m.indexOfRefraction;

		// Exiting transmissive object:
		} else {
			normal *= -1;	// Equivalent to PBR's "Faceforward" func
			eta = m.indexOfRefraction / eta;
		}

		// Equivalent to PBR's "Reflect" func - compute ray direction, transmissive
		
		// Compute cos theta using Snell's law
		float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
		float sin2ThetaT = eta * eta * sin2ThetaI;

		// Handle total internal reflection for transmission
		if (sin2ThetaT >= 1.f) {
			pathSegment.color *= currCol;
			pathSegment.ray.direction = glm::reflect(dir,
				normal);

		// Regular ray bounce
		} else {
			float cosThetaT = std::sqrt(1.f - sin2ThetaT);
			pathSegment.ray.direction = eta * pathSegment.ray.direction +
				(eta * cosThetaI - cosThetaT) * normal;

			pathSegment.color *= currCol;
		}

	// Pt 1: Perfectly Specular
	} else if (m.hasReflective) {
		pathSegment.color *= currSpecCol;
		pathSegment.ray.direction = glm::reflect(pathSegment.ray.direction, 
												 normal);

	// Pure/Ideal-diffuse surfaces - a basic implementation, just calls 
	// the calculateRandomDirectionInHemisphere defined above.
	} else {
		pathSegment.color *= currCol;
		pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
	}
		
	// Offset origin to avoid repeat collisions
	pathSegment.ray.origin = intersect + 0.001f * glm::normalize(pathSegment.ray.direction);
}
