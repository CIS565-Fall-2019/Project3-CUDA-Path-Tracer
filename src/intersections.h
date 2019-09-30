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

// SDFs
// Resources:
// https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
// http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/

__host__ __device__ glm::mat4 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return glm::mat4(
        glm::vec4(1, 0, 0, 0),
        glm::vec4(0, c, -s, 0),
        glm::vec4(0, s, c, 0),
        glm::vec4(0, 0, 0, 1)
    );
}

__host__ __device__ glm::mat4 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return glm::mat4(
        glm::vec4(c, 0, s, 0),
        glm::vec4(0, 1, 0, 0),
        glm::vec4(-s, 0, c, 0),
        glm::vec4(0, 0, 0, 1)
    );
}

__host__ __device__ glm::mat4 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);

    return glm::mat4(
        glm::vec4(c, -s, 0, 0),
        glm::vec4(s, c, 0, 0),
        glm::vec4(0, 0, 1, 0),
        glm::vec4(0, 0, 0, 1)
    );
}

__host__ __device__ float sphereSDF(glm::vec3 p) {
    return glm::length(p) - 0.65;
}

__host__ __device__ float boxSDF(glm::vec3 p) {
    glm::vec3 d = glm::abs(p) - glm::vec3(0.5f);
    return glm::length(glm::max(d, glm::vec3(0.0f)))
        + min(max(d.x, max(d.y, d.z)), 0.0f);
}

__host__ __device__ float cylinderSDF(glm::vec3 p) {
    
    glm::vec2 d = abs(glm::vec2(glm::length(glm::vec2(p.x, p.z)), p.y)) - glm::vec2(0.35f, 0.75f);
    return min(max(d.x, d.y), 0.0f) + glm::length(glm::max(d, glm::vec2(0.0f)));
}

__host__ __device__ float hollowShapeSDF(glm::vec3 p, bool &mat)
{
    // union of three cylinders in each axis
    glm::vec3 pxCyl = glm::vec3((glm::inverse(rotateZ(PI/2.0f)) * glm::vec4(p, 1.0f)));
    glm::vec3 pzCyl = glm::vec3((glm::inverse(rotateX(PI / 2.0f)) * glm::vec4(p, 1.0f)));
    float cylPiece = min(cylinderSDF(p), min(cylinderSDF(pxCyl), cylinderSDF(pzCyl)));

    // intersection of cub and sphere
    float chunkPiece = max(boxSDF(p), sphereSDF(p));

    // setting mat so that we can color the inside and outside of the shape two different materials
    if (chunkPiece > -cylPiece)
    {
        mat = true;
        return chunkPiece;
    }
    else
    {
        mat = false;
        return -cylPiece;
    }
    //return max(chunkPiece, -cylPiece);
}

__host__ __device__ float hollowShapeIntersectionTest(Geom geo, Ray r,
    glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, bool &mat) {
       
    glm::vec3 ro = multiplyMV(geo.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = multiplyMV(geo.inverseTransform, glm::vec4(r.direction, 0.0f));
    
    bool junk;
    // compute t value
    float t = 0;
    float dist;
    outside = hollowShapeSDF(ro, junk) > 0.f;
    for (int i = 0; i < 100; ++i)
    {
        glm::vec3 currPos = rd * t + ro;
        // find distance to the shape from this point on the ray
        float dist = outside ? hollowShapeSDF(currPos, mat) : -hollowShapeSDF(currPos, mat);
        // if we're close enough to the shape we found a good t
        if (glm::abs(dist) < EPSILON) {
            break;
        }
        // otherwise, we keep marching 
        t += dist;
        if (t >= 200) {
            // we assume this ray didn't hit anything
            t = -1;
            break;
        }
    }

    glm::vec3 interpt = ro + t * rd;
    intersectionPoint = multiplyMV(geo.transform, glm::vec4(interpt, 1.0f));

    normal = glm::normalize(glm::vec3(
        hollowShapeSDF(glm::vec3(interpt.x + EPSILON, interpt.y, interpt.z), junk) -
        hollowShapeSDF(glm::vec3(interpt.x - EPSILON, interpt.y, interpt.z), junk),
        hollowShapeSDF(glm::vec3(interpt.x, interpt.y + EPSILON, interpt.z), junk) -
        hollowShapeSDF(glm::vec3(interpt.x, interpt.y - EPSILON, interpt.z), junk),
        hollowShapeSDF(glm::vec3(interpt.x, interpt.y, interpt.z + EPSILON), junk) -
        hollowShapeSDF(glm::vec3(interpt.x, interpt.y, interpt.z - EPSILON), junk)
    ));
    if (!outside) normal = -normal;
    
    return t;
}

__host__ __device__ float torusSDF(glm::vec3 p) {
    glm::vec2 q = glm::vec2(glm::length(glm::vec2(p.x, p.z)) - 0.75, p.y);
    return glm::length(q) - 0.15;
}

__host__ __device__ float twist(glm::vec3 p)
{
    const float k = 5.0; // or some other amount
    float c = cos(k*p.y);
    float s = sin(k*p.y);
    glm::mat2 m = glm::mat2(c, -s, s, c);
    glm::vec3 q = glm::vec3(m*glm::vec2(p.x, p.z), p.y);
    return torusSDF(q);
}

__host__ __device__ float twistIntersectionTest(Geom geo, Ray r,
    glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside, bool &mat) {

    glm::vec3 ro = multiplyMV(geo.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = multiplyMV(geo.inverseTransform, glm::vec4(r.direction, 0.0f));

    // compute t value
    float t = 0;
    float dist;
    outside = twist(ro) > 0.f;
    for (int i = 0; i < 100; ++i)
    {
        glm::vec3 currPos = rd * t + ro;
        // find distance to the shape from this point on the ray
        float dist = outside ? twist(currPos) : -twist(currPos);
        // if we're close enough to the shape we found a good t
        if (glm::abs(dist) < EPSILON) {
            break;
        }
        // otherwise, we keep marching 
        t += dist;
        if (t >= 200) {
            // we assume this ray didn't hit anything
            t = -1;
            break;
        }
    }

    glm::vec3 interpt = ro + t * rd;
    intersectionPoint = multiplyMV(geo.transform, glm::vec4(interpt, 1.0f));

    normal = glm::normalize(glm::vec3(
        twist(glm::vec3(interpt.x + EPSILON, interpt.y, interpt.z)) -
        twist(glm::vec3(interpt.x - EPSILON, interpt.y, interpt.z)),
        twist(glm::vec3(interpt.x, interpt.y + EPSILON, interpt.z)) -
        twist(glm::vec3(interpt.x, interpt.y - EPSILON, interpt.z)),
        twist(glm::vec3(interpt.x, interpt.y, interpt.z + EPSILON)) -
        twist(glm::vec3(interpt.x, interpt.y, interpt.z - EPSILON))
    ));
    if (abs(glm::dot(normal, glm::vec3(0.0f, 0.0f, 1.0f))) < 0.2) mat = false;
    if (!outside) normal = -normal;

    return t;
}

