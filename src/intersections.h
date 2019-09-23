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
__host__ __device__ gvec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ gvec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return gvec3(m * v);
}


/**
Lifted nearly verbatim from Wikipedia article on Moller-Trumbore intersection algorithm
*/
__host__ __device__ float triangleIntersectionTest(Triangle tri, Ray r,
	gvec3& intersectionPoint, gvec3& normal) {

	gvec3 results;
	bool didHit = glm::intersectRayTriangle(r.origin, r.direction, tri.vert0, tri.vert1, tri.vert2, results);
	if (!didHit) return -1;
	float alpha = results.x;
	float beta = results.y;
	float t = results.z;

	//WHY IS THE NORMAL WRONG??
	normal = (tri.norm0 * (1.0f - alpha - beta)) + (tri.norm1 * alpha) + (tri.norm2 * beta);
	intersectionPoint = getPointOnRay(r, t);

	return t;

	/*
	//if (DOTP(tri.normal, r.direction) >= 0) return -1;//triangle facing the same way we are
	float rdx = r.direction.x; float rdy = r.direction.y; float rdz = r.direction.z;
	float tv0x = tri.vert0.x; float tv0y = tri.vert0.y; float tv0z = tri.vert0.z;
	float tv1x = tri.vert1.x; float tv1y = tri.vert1.y; float tv1z = tri.vert1.z;
	float tv2x = tri.vert2.x; float tv2y = tri.vert2.y; float tv2z = tri.vert2.z;
	float tnx = tri.normal.x; float tny = tri.normal.y; float tnz = tri.normal.z;

	//gvec3 edge1 = tri.vert1 - tri.vert0;
	//gvec3 edge2 = tri.vert2 - tri.vert0;
	gvec3 edge1 = gvec3(tv1x, tv1y, tv1z) - gvec3(tv0x, tv0y, tv0z);
	gvec3 edge2 = gvec3(tv2x, tv2y, tv2z) - gvec3(tv0x, tv0y, tv0z);

	gvec3 h = CROSSP(gvec3(rdx, rdy, rdz), edge2);
	float a = DOTP(edge1, h);
	if (a > -EPSILON && a < EPSILON) return -1;//ray parallel to triangle
	float f = 1.0 / a;
	gvec3 s = r.origin - tri.vert0;
	float u = f * DOTP(s, h);
	if (u < 0.0 || u > 1.0) return -1;
	gvec3 q = CROSSP(s, edge1);
	float v = f * DOTP(r.direction, q);
	if (v < 0.0 || u + v > 1.0) return -1;

	float share0 = 1.0 - u - v;

	float t = f * DOTP(edge2, q);

	if (t > EPSILON) {
		intersectionPoint = getPointOnRay(r, t);
		//normal = share0 * tri.norm0 + v * tri.norm1 + u * tri.norm2;
		//normal = tri.normal; //BLOCKidx = 2271, 0, 0, THREADidx = 111, 0, 0
		normal = gvec3(tnx, tny, tnz);
		return t;
	}//if intersection in front of us
	else {
		return -1;
	}
	*/

}

__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
	gvec3& intersectionPoint, gvec3& normal, bool& outside);

/**
 * Test intersection between a ray and a transformed bounding box. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float meshIntersectionTest(Geom mesh, Ray r,
	gvec3& intersectionPoint, gvec3& normal, bool& outside, Triangle* tris, int* triIndex) {

	float t = boxIntersectionTest(mesh, r, intersectionPoint, normal, outside);
	if (t < 0) return -1;
	//if we hit inside the box, THEN check against our triangles

	gvec3 tmp_intersection, min_intersection;
	gvec3 tmp_normal, min_normal;
	float t_min = INFINITY;
	for (int i = mesh.triangleIndex; i < mesh.triangleIndex + mesh.triangleCount; i++) {
		Triangle tri = tris[i];
		t = triangleIntersectionTest(tri, r, tmp_intersection, tmp_normal);
		if (t > 0.0 && t < t_min) {
			*triIndex = i;
			//float dinx = tmp_intersection.x; float diny = tmp_intersection.y; float dinz = tmp_intersection.z;
			//float dnox = tmp_normal.x; float dnoy = tmp_normal.y; float dnoz = tmp_normal.z;

			//min_intersection = gvec3(dinx, diny, dinz);
			min_intersection = tmp_intersection;
			//min_normal = gvec3(dnox, dnoy, dnoz);
			min_normal = tmp_normal;
			t_min = t;
		}//new minimum
	}//for

	if (t_min > 0.0 && t_min < INFINITY) {
		intersectionPoint = min_intersection;
		normal = min_normal;

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
        gvec3 &intersectionPoint, gvec3 &normal, bool &outside) {
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
        intersectionPoint = multiplyMV(box.transform, gvec4(getPointOnRay(q, tmin), 1.0f));
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
        gvec3 &intersectionPoint, gvec3 &normal, bool &outside) {
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

    gvec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, gvec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, gvec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}
