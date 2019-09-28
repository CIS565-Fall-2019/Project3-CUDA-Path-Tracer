#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

__device__ DebugVector DVV(gvec3 v) {
	return { v.x, v.y, v.z };
}

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
__host__ __device__ gvec3 getPointOnRayEp(Ray r, float t) {
    return r.origin + (t - EPSILON) * glm::normalize(r.direction);
}

__host__ __device__ gvec3 getPointOnRay(Ray r, float t) {
	return r.origin + (t) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ gvec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return gvec3(m * v);
}


__host__ __device__ float triangleIntersectionTest(Triangle tri, Ray r,
	gvec3& intersectionPoint, gvec3& normal, float2& uv, bool& backface) {

	gvec3 results, results2;
	//doing two tests: against the triangle, and then the reverse-triangle (not ideal, but needed for refraction)
	bool didHit = glm::intersectRayTriangle(r.origin, r.direction, tri.vert0, tri.vert1, tri.vert2, results);
	bool didHitReverse = glm::intersectRayTriangle(r.origin, r.direction, tri.vert0, tri.vert2, tri.vert1, results2);

	float alpha, beta;

	//if (!didHit) return -1;
	if (didHit) {
		alpha = results.x;
		beta = results.y;
		backface = false;
	}
	else if (didHitReverse) {
		alpha = results2.x;
		beta = results2.y;
		backface = true;
	}
	else return -1;

	float t = results.z;

	normal = (tri.norm0 * (1.0f - alpha - beta)) + (tri.norm1 * alpha) + (tri.norm2 * beta);
	if (backface) normal *= -1.0;
	uv.x = (tri.uv0.x * (1.0f - alpha - beta)) + (tri.uv1.x * alpha) + (tri.uv2.x * beta);
	uv.y = (tri.uv0.y * (1.0f - alpha - beta)) + (tri.uv1.y * alpha) + (tri.uv2.y * beta);
	intersectionPoint = getPointOnRayEp(r, t);

	return t;

}

__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
	gvec3& intersectionPoint, gvec3& normal, bool& outside, float2* uv);

/**
 * Test intersection between a ray and a transformed bounding box. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside. (as of yet, unchanged)
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r,
	gvec3& intersectionPoint, gvec3& normal, bool& outside, Triangle* tris, int* triIndex, float2* uv) {

	bool boundingBoxOutside = true;
	float2 garbageuv;

	float t = boxIntersectionTest(mesh, r, intersectionPoint, normal, boundingBoxOutside, &garbageuv);
	if (t < 0) return -1;
	//if we hit inside the box, THEN check against our triangles

	gvec3 tmp_intersection, min_intersection;
	gvec3 tmp_normal, min_normal;
	bool tmp_backface = false;
	bool backface = false;
	float2 tmp_uv = { -1.0, -1.0 };
	float2 min_uv = { -1.0, -1.0 };
	float t_min = INFINITY;
	for (int i = mesh.triangleIndex; i < mesh.triangleIndex + mesh.triangleCount; i++) {
		Triangle tri = tris[i];
		tmp_uv = { -1.0, -1.0 };
		tmp_backface = false;
		t = triangleIntersectionTest(tri, r, tmp_intersection, tmp_normal, tmp_uv, tmp_backface);
		if (t > 0.0 && t < t_min) {
			*triIndex = i;
			backface = tmp_backface;
			min_intersection = tmp_intersection;
			min_normal = tmp_normal;
			min_uv = tmp_uv;
			t_min = t;
		}//new minimum
	}//for

	if (t_min > 0.0 && t_min < INFINITY) {
		intersectionPoint = min_intersection;
		normal = min_normal;
		*uv = min_uv;
		if (backface) outside = false;
		else outside = true;
		//outside = !backface;

		return t_min;
	}//if
	else {
		return -1;
	}
}

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
        gvec3 &intersectionPoint, gvec3 &normal, bool &outside, float2* uv) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, gvec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, gvec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    gvec3 tmin_n;
    gvec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            gvec3 n;
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
		gvec3 utInt = getPointOnRay(q, tmin);
		if (utInt.x * utInt.x - 0.25f < EPSILON) {
			uv->x = (utInt.y + 0.5f);
			uv->y = (utInt.z + 0.5f);
		}//if x close to -.5 or .5
		else if (utInt.y * utInt.y - 0.25f < EPSILON) {
			uv->x = (utInt.x + 0.5f);
			uv->y = (utInt.z + 0.5f);
		}//if y close to -.5 or .5
		else {
			uv->x = (utInt.x + 0.5f);
			uv->y = (utInt.y + 0.5f);
		}//hopefully, z close to -.5 or .5

        intersectionPoint = multiplyMV(box.transform, gvec4(getPointOnRayEp(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.transform, gvec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

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
        gvec3 &intersectionPoint, gvec3 &normal, bool &outside, float2* uv) {
    float radius = .5;

    gvec3 ro = multiplyMV(sphere.inverseTransform, gvec4(r.origin, 1.0f));
    gvec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, gvec4(r.direction, 0.0f)));

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

    gvec3 objspaceIntersection = getPointOnRayEp(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, gvec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, gvec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}
