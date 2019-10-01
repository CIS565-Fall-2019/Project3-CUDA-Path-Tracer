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
		glm::vec3 pointOnRay = getPointOnRay(q, tmin);
        intersectionPoint = multiplyMV(box.transform, glm::vec4(pointOnRay, 1.0f));
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



__host__ __device__ float triArea(const glm::vec3 &p1, const glm::vec3 &p2, const glm::vec3 &p3) {
	return glm::length(glm::cross(p1 - p2, p3 - p2)) * 0.5f;
}

__host__ __device__ glm::vec3 triGetNormal(const glm::vec3 &P, const Triangle tri) {
	float A = triArea(tri.p1, tri.p2, tri.p3);
	float A0 = triArea(tri.p2, tri.p3, P);
	float A1 = triArea(tri.p1, tri.p3, P);
	float A2 = triArea(tri.p1, tri.p2, P);
	return glm::normalize(tri.n1 * A0 / A + tri.n2 * A1 / A + tri.n3 * A2 / A);
}

__host__ __device__ glm::vec2 triGetUV(const glm::vec3 &P, const Triangle tri) {
	float A = triArea(tri.p1, tri.p2, tri.p3);
	float A0 = triArea(tri.p2, tri.p3, P);
	float A1 = triArea(tri.p1, tri.p3, P);
	float A2 = triArea(tri.p1, tri.p2, P);
	return glm::normalize(tri.uv1 * A0 / A + tri.uv2 * A1 / A + tri.uv3 * A2 / A);
}

__host__ __device__ float triangleIntersectionTest(Triangle triangle, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec2 &uv, bool &outside) {
	glm::vec3 p1 = triangle.p1;
	glm::vec3 p2 = triangle.p2;
	glm::vec3 p3 = triangle.p3;
	glm::vec3 planeNormal = glm::normalize(glm::cross(p2 - p1, p3 - p2));
	//1. Ray-plane intersection
	float t = glm::dot(planeNormal, (p1 - r.origin)) / glm::dot(planeNormal, r.direction);
	if (t < 0) return -1;

	glm::vec3 P = r.origin + t * r.direction;
	//2. Barycentric test
	float S = 0.5f * glm::length(glm::cross(p1 - p2, p1 - p3));
	float s1 = 0.5f * glm::length(glm::cross(P - p2, P - p3)) / S;
	float s2 = 0.5f * glm::length(glm::cross(P - p3, P - p1)) / S;
	float s3 = 0.5f * glm::length(glm::cross(P - p1, P - p2)) / S;
	float sum = s1 + s2 + s3;

	if (s1 >= 0 && s1 <= 1 && s2 >= 0 && s2 <= 1 && s3 >= 0 && s3 <= 1 && std::fabsf(sum - 1.0f) < 0.0001) {
		intersectionPoint = P;
		glm::vec3 n1 = triangle.n1;
		glm::vec3 n2 = triangle.n2;
		glm::vec3 n3 = triangle.n3;
		normal = triGetNormal(intersectionPoint, triangle);
		uv = triGetUV(intersectionPoint, triangle);
		outside = glm::dot(normal, r.direction) < 0;
		if (!outside) normal = -normal;
		return t;
	}
	return -1;
}

__host__ __device__ float meshIntersectionTest(Geom mesh, Triangle *triangles, Ray r,
	glm::vec3 &intersectionPoint, glm::vec3 &normal, glm::vec2 &uv, bool &outside) {
	Ray q;
	q.origin = multiplyMV(mesh.inverseTransform, glm::vec4(r.origin, 1.0f));
	q.direction = glm::normalize(multiplyMV(mesh.inverseTransform, glm::vec4(r.direction, 0.0f)));

	float tmin = -1e38f;
	float tmax = 1e38f;
	glm::vec3 tmin_n;
	glm::vec3 tmax_n;
	for (int xyz = 0; xyz < 3; ++xyz) {
		float qdxyz = q.direction[xyz];
		/*if (glm::abs(qdxyz) > 0.00001f)*/ {
			float t1 = (mesh.bottomCornerBound[xyz] - q.origin[xyz]) / qdxyz;
			float t2 = (mesh.topCornerBound[xyz] - q.origin[xyz]) / qdxyz;
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

	if (!(tmax >= tmin && tmax > 0)) {
		return -1;
	}

	float smallestT = -1;
	for (int i = mesh.firstTriangle; i < mesh.lastTriangle; i++) {
		glm::vec3 intersectionPoint_new;
		glm::vec3 normal_new;
		glm::vec2 uv_new;
		bool outside_new = false;
		float t_new = triangleIntersectionTest(triangles[i], q, intersectionPoint_new, normal_new, uv_new, outside_new);
		if (t_new >= 0 && (smallestT < 0 || t_new < smallestT)) {
			smallestT = t_new;
			intersectionPoint = intersectionPoint_new;
			normal = normal_new;
			outside = outside_new;
			uv = uv_new;
		}
	}

	if (smallestT >= 0) {
		intersectionPoint = multiplyMV(mesh.transform, glm::vec4(intersectionPoint, 1.f));
		normal = glm::normalize(multiplyMV(mesh.transform, glm::vec4(normal, 0.f)));
		return glm::length(r.origin - intersectionPoint);
	}

	return -1;
}