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

/**
 * Test intersection between a ray and a transformed triangle. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, 
	Triangle *triangles) {

	// transform ray into object space
	glm::vec3 ro = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	glm::vec3 rd = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

	Ray rt;
	rt.origin = ro;
	rt.direction = rd;

#if BOUNDING_VOLUME
	// TODO: compare ray with bounding volume
	float min_t = (mesh.minPos.x - ro.x) / rd.x;
	float max_t = (mesh.maxPos.x - ro.x) / rd.x;

	if (min_t > max_t) {
		float temp = min_t;
		min_t = max_t;
		max_t = temp;
	}

	float min_t_y = (mesh.minPos.y - ro.y) / rd.y;
	float max_t_y = (mesh.maxPos.y - ro.y) / rd.y;

	if (min_t_y > max_t_y) {
		float temp = min_t_y;
		min_t_y = max_t_y;
		max_t_y = temp;
	}

	if (min_t > max_t_y || min_t_y > max_t) {
		return -1;
	}

	if (min_t_y < min_t) {
		min_t = min_t_y;
	}

	if (max_t_y > max_t) {
		max_t = max_t_y;
	}

	float min_t_z = (mesh.minPos.z - ro.z) / rd.z;
	float max_t_z = (mesh.maxPos.z - ro.z) / rd.z;

	if (min_t_z > max_t_z) {
		float temp = min_t_z;
		min_t_z = max_t_z;
		max_t_z = temp;
	}

	if (min_t > max_t_z || min_t_z > max_t) {
		return -1;
	}
#endif // #if BOUNDING_VOLUME

	float t_min = FLT_MAX;
	glm::vec3 objSpaceIntersectionPoint(0.f);
	glm::vec3 objSpaceNormal(0.f);
	int triIndex = 0;

	bool didIntersect = false;
	for (int i = mesh.trianglesStart; i < mesh.trianglesEnd; i++) {
		Triangle tri = triangles[i];
		glm::vec3 barycentricPos(0.f);
		float t;
		if (glm::intersectRayTriangle(ro, rd, tri.positions[0], tri.positions[1], tri.positions[2], barycentricPos)) {
			t = barycentricPos.z;
			if (t > 0.0f && t_min > t)
			{
				didIntersect = true;
				t_min = t;
				triIndex = i;
			}
		}
	}

	if (!didIntersect) {
		return -1;
	}

	outside = true; // TODO: hmmm
	intersectionPoint = multiplyMV(mesh.transform, glm::vec4(getPointOnRay(rt, t_min), 1.f));
	normal = glm::normalize(multiplyMV(mesh.invTranspose, glm::vec4(triangles[triIndex].normal, 0.f)));
	if (!outside) {
		normal = -normal;
	}

	return glm::length(r.origin - intersectionPoint);
}
