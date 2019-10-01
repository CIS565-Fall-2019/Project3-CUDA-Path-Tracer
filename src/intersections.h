#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.transform, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}


// PART 2 Procedural shapes - based on IQ implicit surfaces

//////////////////////////////// CAPPY /////////////////////////////////////////

// Smooth minimum from IQ
__host__ __device__ float smin(float a, float b, float k) {
	float res = exp(-k * a) + exp(-k * b); //expf(-k * a) + expf(-k * b);
	return -log(res) / k;
}

__host__ __device__ float arm(glm::vec3 pos) {
	return smin(glm::length(pos + glm::vec3(0.1, -0.1, 0.2)) - 0.2f, glm::length(glm::vec3(pos.x, pos.y * 0.5, pos.z)) - 0.2f, 16.f);
}

__host__ __device__ float eye(glm::vec3 pos, float degrees) {
	// Tilts the eyes
	float c = cos(glm::radians(degrees));
	float s = sin(glm::radians(degrees));
	glm::mat3  mZ = glm::mat3(c, s, 0.f, -s, c, 0.f, 0.f, 0.f, 1.f);     							// Z-axis rotation
	glm::mat3  mY = glm::mat3(cos(glm::radians(-degrees / 1.4f)), 0.f, -sin(glm::radians(-degrees / 1.4f)), // Y-axis rotation
		0.f, 1.f, 0.f,
		sin(glm::radians(-degrees / 1.4f)), 0.f, cos(glm::radians(-degrees / 1.4f)));

	// The main eyeball
	glm::vec3  eye = mY * mZ * (glm::vec3(pos.x + degrees * 0.0215, pos.y * 0.85 - 0.05, pos.z * 1.5f + 0.38f));

	// The surrounding torus
	glm::vec3  outline = mY * mZ * pos;
	glm::vec2 t = glm::vec2(0.25, 0.05);
	glm::vec2 d = glm::vec2(glm::length(glm::vec2(outline.x, outline.y) * glm::vec2(0.75, 0.65)) - t.x, outline.z);

	float iris = glm::length(glm::vec3(pos.x + degrees * 0.0035, pos.y * 0.85, pos.z * 1.7 - 0.07f)) - 0.13;

	return glm::min(iris, glm::min(glm::length(d) - t.y, glm::length(eye) - 0.5f));
}

__host__ __device__ float box(glm::vec3 p, glm::vec3 b)
{
	glm::vec3 d = glm::abs(p) - b;
	return glm::min(glm::max(d.x, glm::max(d.y, d.z)), 0.f) + glm::length(glm::max(d, 0.f));
}

// Sphere from IQ - used for Cappy's Body
__host__ __device__ float sphere(glm::vec3 pos) {
	return glm::length(pos) - 0.97f;
}

__host__ __device__ float body(glm::vec3 p, glm::vec2 h) { // major warping of IQ's capped cylinder function
	glm::vec2 d = glm::abs(glm::vec2(glm::length(glm::vec2(p.x, p.y)), p.z)) - 
		(h * glm::vec2(p.z / 2.3f - sin(12.f * p.z) * 0.03 * (p.z - 2.f) * 1.2f, 1.f));
	return glm::min(glm::max(d.x, d.y), 0.f) + glm::length(glm::max(d, 0.f));
}

// Cylinder from IQ - used for Cappy's Hat
__host__ __device__ float sdCappedCylinder(glm::vec3 p, glm::vec2 h, float f) {
	glm::vec2 d = glm::abs(glm::vec2(glm::length(glm::vec2(p.x, p.z)), p.y)) - h;
	return glm::min(glm::max(d.x, d.y), 0.f) + glm::length(glm::max(d, 0.f)) - f;
}

__host__ __device__ float hatBrim(glm::vec3 p, glm::vec2 t) {   // Essentially a bent torus
	float c = cos(glm::radians(12.0 * p.x));
	float s = sin(glm::radians(12.0 * p.x));
	glm::mat2  m = glm::mat2(c, -s, s, c);
	glm::vec3  q = glm::vec3(m * glm::vec2(p.x, p.y), p.z);
	glm::vec2 d = glm::vec2(glm::length(glm::vec2(q.x, q.z) * 0.5f) - t.x, q.y + 0.5f);
	return glm::length(d) - t.y;
}

// The main part of the hat, using bent and beveled cylinders
__host__ __device__ float hatBase(glm::vec3 p) {
	float c = cos(glm::radians(12.0 * p.y));
	float s = sin(glm::radians(12.0 * p.y));
	glm::mat2  m = glm::mat2(c, -s, s, c);
	glm::vec3  q = glm::vec3(m * glm::vec2(p.z, p.y), p.x);
	float scaleVal = p.y * 0.2 + 0.7;
	// Hat and ribbon!
	return glm::min(sdCappedCylinder(q + 
		glm::vec3(0.f, 0.2 * (1.f + cos(p.x)) - 0.25, 0.f), 
		glm::vec2(1.37f * scaleVal, 0.2f), 0.05),
		sdCappedCylinder(q + glm::vec3(0.f, -0.4f, 0.f), 
			glm::vec2(1.2f * scaleVal, 0.7f), 0.15));
}

__host__ __device__ float mySDF(glm::vec3 pos) {
	glm::mat3 rot = glm::mat3(glm::vec3(cos(glm::radians(150.f)), 0.f, -sin(glm::radians(150.f))),
		glm::vec3(0.f, 1.f, 0.f),
		glm::vec3(sin(glm::radians(150.f)), 0.f, cos(glm::radians(150.f))));
	//pos = pos + Vector3f(0.f, sin(u_Time * 0.07) * 0.7, 0.f);

	pos = rot * pos;
	float hat = glm::min(hatBase(pos), hatBrim(pos * glm::vec3(1.f, 1.f, 1.f), 
											   glm::vec2(0.6f, 0.2f)));
	float yScale = 0.7f - pos.z / 2.2f;
	float bod = smin(sphere(pos + glm::vec3(0.f, 0.7f, 0.03f)),
		body(pos + glm::vec3(sin(3.f * pos.z) * 0.1, 
			yScale + sin(10.f * pos.z) * 0.1, 1.93f), 
			glm::vec2(1.f, 1.7f)), 22.f);
	float arms = glm::min(arm(pos - glm::vec3(0.8f, -1.f, 1.f)), 
						  arm((pos - glm::vec3(-0.8f, -1.f, 1.f)) *
									 glm::vec3(-1.f, 1.f, 1.f)));
	bod = smin(bod, arms, 20.f);
	float eyes = smin(eye(pos - glm::vec3(0.34f, 0.25f, 0.98f), 23.f), 
					  eye(pos - glm::vec3(-0.34f, 0.25f, 0.98f), -23.f), 100.f);

	float cappy = glm::min(glm::min(hat, bod), eyes);
	return cappy;
}

// From Jamie Wong's Ray Marching and Signed Distance Functions webpage:
__host__ __device__ glm::vec3 estimateNormal(glm::vec3 p) {
	return glm::normalize(glm::vec3(
		mySDF(glm::vec3(p.x + EPSILON, p.y, p.z)) - mySDF(glm::vec3(p.x - EPSILON, p.y, p.z)),
		mySDF(glm::vec3(p.x, p.y + EPSILON, p.z)) - mySDF(glm::vec3(p.x, p.y - EPSILON, p.z)),
		mySDF(glm::vec3(p.x, p.y, p.z + EPSILON)) - mySDF(glm::vec3(p.x, p.y, p.z - EPSILON))));
}

__host__ __device__ float cappyIntersectionTest(Geom cappy, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
	//Transform the ray
	glm::vec3 ro = multiplyMV(cappy.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(cappy.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = glm::normalize(rd);

	bool geo = false; 

	float maxLoops = 0.f; // Ensures program doesn't crash

	float t = mySDF(rt.origin);
	float dist = t;

	while (t < 25.f && maxLoops < 100.f) {
		rt.origin += t * rt.direction;
		float i = mySDF(rt.origin);
		dist += i;
		t = i;
		if (i < 0.0001f && i > -0.0001f) { 
			geo = true;
			break;
		}
		maxLoops++;
	}

	if (geo) {
		glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

		intersectionPoint = multiplyMV(cappy.transform, glm::vec4(objspaceIntersection, 1.f));
		normal = glm::normalize(multiplyMV(cappy.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

		//InitializeIntersection(isect, t, Point3f(pos));
		//outside = true;

		normal = estimateNormal(intersectionPoint);

		return glm::length(r.origin - intersectionPoint);
	}
	else {
		return -1;
	}
}

/////////////////////////////////// DUCK ///////////////////////////////////////
// Built off of an old case study I'd done

// Triangular prism from IQ
__host__ __device__ float sdTriPrism(glm::vec3 p, glm::vec2 h)
{
	glm::vec3 q = glm::abs(p);
	return glm::max(q.z - h.y, glm::max(q.x * 0.866025f + p.y * 0.5f, -p.y) - h.x * 0.5f);
}


__host__ __device__ float myDuckSDF(glm::vec3 pos) {

	// Adjusts initial duck angle:
	float cView = cos(glm::radians(26.f));
	float sView = sin(glm::radians(26.f));
	float cViewX = cos(glm::radians(-26.f));
	float sViewX = sin(glm::radians(-26.f));
	float cViewZ = cos(glm::radians(5.f));
	float sViewZ = sin(glm::radians(5.f));
	glm::mat3 view = glm::mat3(cViewZ, sViewZ, 0.f, -sViewZ, cViewZ, 0.f, 0.f, 0.f, 1.f) *
		glm::mat3(1.f, 0.f, 0.f, 0.f, cViewX, sViewX, 0.f, -sViewX, cViewX) *
		glm::mat3(cView, 0.f, -sView, 0.f, 1.f, 0.f, sView, 0.f, cView);

	glm::vec3 oldPos = view * pos;

	// ~~~~~BODY~~~~~
	pos = oldPos;

	// An x-axis rotation matrix used for elements of the body
	float c = cos(glm::radians(-30.f));
	float s = sin(glm::radians(-30.f));
	glm::mat3  mX = glm::mat3(1.f, 0.f, 0.f, 0.f, c, s, 0.f, -s, c);

	float bod = glm::length(mX * pos * glm::vec3(1.f, 1.f, 0.8) +
		glm::vec3(0.f, 0.25f, 0.4f)) - 0.8f;

	float wing = glm::length((mX * pos + glm::vec3(-0.3f, 0.1f, 0.7f)) / glm::vec3(0.9f, cos(pos.z / 1.4 - 0.5), 1.f)) - 0.69;
	float wing2 = glm::length((mX * mX * pos + glm::vec3(-0.2f, 0.7f, 0.5f)) / glm::vec3(1.f, cos(pos.z / 1.4 - 0.5), 1.f)) - 0.69;
	float wing3 = glm::length((mX * pos + glm::vec3(0.3f, 0.1f, 0.7f)) / glm::vec3(0.9f, cos(pos.z / 1.4 - 0.5), 1.f)) - 0.69;
	float wing4 = glm::length((mX * mX * pos + glm::vec3(0.2f, 0.7f, 0.5f)) / glm::vec3(1.f, cos(pos.z / 1.4 - 0.5), 1.f)) - 0.69;
	wing = min(min(wing, wing2), min(wing3, wing4));

	float tailC = cos(glm::radians(7.f * sin((pos.z * 2.f) * 0.18) * pos.z));
	float tailS = sin(glm::radians(7.f * sin((pos.z * 2.f) * 0.18) * pos.z));
	glm::vec3 tailPos = glm::mat3(tailC, 0.f, -tailS, 0.f, 1.f, 0.f, tailS, 0.f, tailC) * pos;

	float tail = sdCappedCylinder(glm::vec3(tailPos.x, tailPos.z, tailPos.y) + glm::vec3(0.f, 1.6f, (tailPos.z + 0.8f) - 0.05),
		glm::vec2(0.8f * sin(tailPos.z + 1.95f), 0.7f), 0.f);

	// Motion:
	float upC = cos(glm::radians(4.f));
	float upS = sin(glm::radians(4.f));
	glm::vec3 upperTilt = glm::mat3(1.f, 0.f, 0.f,
		0.f, upC, upS,
		0.f, -upS, upC) * oldPos;
	glm::vec3 neckPos = pos;

	float neck = sdCappedCylinder(neckPos - glm::vec3(0.f, 0.7f, -0.1f),
		glm::vec2(0.45, 0.7f), 0.f);

	// Motion:
	glm::vec3 headPos = upperTilt;

	float head = glm::length(headPos - glm::vec3(0.f, 1.4f, 0.f)) - 0.5;

	float body = glm::min(wing, smin(tail, smin(head, smin(neck, bod, 20.f), 20.f), 20.f));


	// ~~~~~FACIAL FEATURES~~~~~

	// Motion:
	glm::vec3 turn = glm::mat3(1.f, 0.f, 0.f,
		0.f, upC, upS,
		0.f, -upS, upC) *
		(glm::vec3(0.f, 1.5f, 0.f) +
		(oldPos + glm::vec3(0.f, -1.5f, 0.f)));

	// An x-axis rotation matrix to angle the beak
	float c1 = cos(glm::radians(-50.f));
	float s1 = sin(glm::radians(-50.f));
	glm::mat3  mBeakX = glm::mat3(1.f, 0.f, 0.f, 0.f, c1, s1, 0.f, -s1, c1);

	float beak = sdTriPrism(mBeakX * (glm::vec3(0.f, -3.5f, 0.1f) +
		glm::vec3(turn.x * 0.9, turn.y * 2.0 + 0.8 * cos(turn.x * 2.0), 1.2 * turn.z - 0.5 * cos(turn.x * 2.0))),
		glm::vec2(0.4f, 0.3f));

	float eyes = glm::min(glm::length(turn + glm::vec3(-0.24, -1.53f, -0.43)) - 0.05,
		glm::length(turn + glm::vec3(0.24, -1.53f, -0.43)) - 0.05);

	float face = glm::min(beak, eyes);


	// ~~~~~LEGS~~~~~

	// Rotation matrix to angle webbed feet
	glm::mat3 footRot = // Y-axis rotation
		glm::mat3(cos(glm::radians(45.f)), 0.f, -sin(glm::radians(45.f)),
			0.f, 1.f, 0.f,
			sin(glm::radians(45.f)), 0.f, cos(glm::radians(45.f))) *
		// X-axis rotation
		glm::mat3(1.f, 0.f, 0.f,
			0.f, cos(glm::radians(60.f)), sin(glm::radians(60.f)),
			0.f, -sin(glm::radians(60.f)), cos(glm::radians(60.f)));
	// Matrices to tilt legs
	glm::mat3 mLeftZ = glm::mat3(cos(glm::radians(10.f)), sin(glm::radians(10.f)), 0.f,
		-sin(glm::radians(10.f)), cos(glm::radians(10.f)), 0.f,
		0.f, 0.f, 1.f);
	glm::mat3 mRightZ = glm::mat3(cos(glm::radians(-10.f)), sin(glm::radians(-10.f)), 0.f,
		-sin(glm::radians(-10.f)), cos(glm::radians(-10.f)), 0.f,
		0.f, 0.f, 1.f);

	// LEFT 
	// Movement:
	//float xL = sin(u_Time * 0.18);
	float stepLC = cos(glm::radians(20.f));
	float stepLS = sin(glm::radians(20.f));
	glm::vec3  stepLeft = glm::vec3(0.f, -0.35f, -0.4f) + 
						  glm::mat3(1.f, 0.f, 0.f, 0.f, stepLC, stepLS, 0.f, -stepLS, stepLC) * 
						  (pos + glm::vec3(0.f, 0.35f, 0.4f));
	// Structure
	float leftStick = sdCappedCylinder(mLeftZ * (stepLeft + glm::vec3(0.5f, 0.95f, 0.4f)), glm::vec2(0.09, 0.35f), 0.f);
	glm::vec3 footPos = stepLeft + glm::vec3(0.55, 2.f, -0.12f);
	float footLeft = smin(leftStick, box((footRot * (glm::vec3(0.f, 0.f, cos(footPos.x * 2.f) - 0.5) + footPos)), glm::vec3(0.5f, 0.5f, 0.5f)), 20.f);
	float leftLeg = max(-box(footPos + glm::vec3(0.f, 0.9f, 0.f), glm::vec3(1.3f, 1.3f, 1.5f)), footLeft);

	// RIGHT
	// Movement:
	float xR = sin(PI);
	float stepRC = cos(glm::radians(60.f * xR + 20.f));
	float stepRS = sin(glm::radians(60.f * xR + 20.f));
	glm::vec3  stepRight = glm::vec3(0.f, -0.35f, -0.4f) + glm::mat3(1.f, 0.f, 0.f, 0.f, stepRC, stepRS, 0.f, -stepRS, stepRC) * (pos + glm::vec3(0.f, 0.35f, 0.4f));
	// Structure:
	float rightStick = sdCappedCylinder(mRightZ * (stepRight + glm::vec3(-0.5f, 0.95f, 0.4f)), glm::vec2(0.09, 0.35f), 0.f);
	glm::vec3 rfootPos = stepRight + glm::vec3(-0.55, 2.f, -0.12f);
	float footRight = smin(rightStick, box((footRot * (glm::vec3(0.f, 0.f, cos(rfootPos.x * 2.f) - 0.5) + rfootPos)), glm::vec3(0.5f, 0.5f, 0.5f)), 20.f);
	float rightLeg = glm::max(-box(rfootPos + glm::vec3(0.f, 0.9f, 0.f), glm::vec3(1.3f, 1.3f, 1.5f)), footRight);

	float legs = glm::max(-box(oldPos + glm::vec3(0.f, 2.8f, 0.f), glm::vec3(3.f, 1.3f, 3.f)), glm::min(leftLeg, rightLeg));

	return min(legs, min(face, body));;
}

// From Jamie Wong's Ray Marching and Signed Distance Functions webpage:
__host__ __device__ glm::vec3 estimateDuckNormal(glm::vec3 p) {
	return glm::normalize(glm::vec3(
		myDuckSDF(glm::vec3(p.x + EPSILON, p.y, p.z)) - myDuckSDF(glm::vec3(p.x - EPSILON, p.y, p.z)),
		myDuckSDF(glm::vec3(p.x, p.y + EPSILON, p.z)) - myDuckSDF(glm::vec3(p.x, p.y - EPSILON, p.z)),
		myDuckSDF(glm::vec3(p.x, p.y, p.z + EPSILON)) - myDuckSDF(glm::vec3(p.x, p.y, p.z - EPSILON))));
}

__host__ __device__ float duckIntersectionTest(Geom duck, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
	//Transform the ray
	glm::vec3 ro = multiplyMV(duck.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(duck.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = glm::normalize(rd);

	bool geo = false;

	float maxLoops = 0.f; // Ensures program doesn't crash

	float t = myDuckSDF(rt.origin);
	float dist = t;

	while (t < 25.f && maxLoops < 100.f) {
		rt.origin += t * rt.direction;
		float i = myDuckSDF(rt.origin);
		dist += i;
		t = i;
		if (i < 0.0001f && i > -0.0001f) {
			geo = true;
			break;
		}
		maxLoops++;
	}

	if (geo) {
		glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

		intersectionPoint = multiplyMV(duck.transform, glm::vec4(objspaceIntersection, 1.f));
		normal = glm::normalize(multiplyMV(duck.invTranspose, glm::vec4(objspaceIntersection, 0.f)));

		//InitializeIntersection(isect, t, Point3f(pos));
		//outside = true;

		normal = estimateDuckNormal(intersectionPoint);

		return glm::length(r.origin - intersectionPoint);
	}
	else {
		return -1;
	}
}