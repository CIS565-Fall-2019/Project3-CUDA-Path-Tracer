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

__host__ __device__
glm::vec3 refract(const glm::vec3 &rayDir, const glm::vec3 &normal, const float iOfR,
    glm::vec3 diffCol, glm::vec3 specCol, glm::vec3 &col)
{    
    float eta;
    glm::vec3 norm;

    // if the dot between the last intersection and the normal of this intersection is negative, 
    // we should be entering a transmissive surface, otherwise we are exiting it
    if (glm::dot(-rayDir, normal) <= 0)
    {
        // leaving object
        eta = iOfR;
        norm = -normal;
    }
    else
    {
        // entering object
        eta = 1.0f / iOfR;
        norm = normal;
    }

    // Compute cos theta using Snell's law
    float cosThetaI = glm::dot(norm, -rayDir);
    float sin2ThetaI = glm::max(float(0), float(1 - cosThetaI * cosThetaI));
    float sin2ThetaT = eta * eta * sin2ThetaI;
    

    // Handle total internal reflection for transmission
    if (sin2ThetaT >= 1)
    {
        col *= specCol;
        return glm::reflect(rayDir, norm);
    }

    col *= diffCol;
    float cosThetaT = glm::sqrt(1 - sin2ThetaT);
    return eta * rayDir + (eta * cosThetaI - cosThetaT) * glm::vec3(norm);
    //return glm::refract(rayDir, norm, eta);
}

__host__ __device__
glm::vec3 fresnel(const glm::vec3 &rayDir, const glm::vec3 &normal, const float iOfR, 
    glm::vec3 diffCol, glm::vec3 specCol, glm::vec3 &col,
    thrust::default_random_engine &rng)
{   
    float ei, et;

    // if the dot between the last intersection and the normal of this intersection is negative, 
    // we should be entering a transmissive surface, otherwise we are exiting it
    if (glm::dot(rayDir, normal) <= 0)
    {
        // leaving object
        ei = iOfR;
        et = 1.0f;
    }
    else
    {
        // entering object
        ei = iOfR;
        et = 1.0f;
    }
    
    // Compute cos theta using Snell's law
    float cosThetaI = glm::clamp(-1.f, 1.f, glm::dot(rayDir, normal));
    float sinThetaI = glm::sqrt(glm::max(float(0), float(1 - cosThetaI * cosThetaI)));
    float sinThetaT = ei/et * sinThetaI;

    // Handle total internal reflection for transmission
    if (sinThetaT >= 1)
    {
        col *= specCol;
        return glm::reflect(rayDir, normal);
    }

    float cosThetaT = glm::sqrt(glm::max(float(0), float(1 - sinThetaT * sinThetaT)));
    cosThetaI = glm::abs(cosThetaI);

    //find rparallel and rperpendicular
    float rPar = ((et*cosThetaI) - (ei*cosThetaT)) / ((et*cosThetaI) + (ei*cosThetaT));
    float rPerp = ((ei*cosThetaI) - (et*cosThetaT)) / ((ei*cosThetaI) + (et*cosThetaT));

    float reflProb = (rPar*rPar + rPerp*rPerp) / 2.0f;
    thrust::uniform_real_distribution<float> u01(0, 1);

    if (u01(rng) < reflProb)
    {
        col *= specCol;
        return glm::reflect(rayDir, normal);
    }
    else
    {
        return refract(rayDir, normal, iOfR, diffCol, specCol, col);
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
		PathSegment &pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    glm::vec3 rayDir = glm::normalize(pathSegment.ray.direction);
    normal = glm::normalize(normal);

    if (m.hasReflective && m.hasRefractive)
    {
        // Fresnel dielectric
        pathSegment.ray.direction = fresnel(rayDir, normal, m.indexOfRefraction,
            m.color, m.specular.color, pathSegment.color, rng);
    }
    else if (m.hasRefractive)
    {
        pathSegment.ray.direction = refract(rayDir, normal, m.indexOfRefraction,
            m.color, m.specular.color, pathSegment.color);
    }
    else if (m.hasReflective)
    {
        pathSegment.ray.direction = glm::reflect(rayDir, normal);
        pathSegment.color *= m.specular.color;
    }
    else
    {
        // Material must be simple diffuse
        pathSegment.ray.direction = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.color *= m.color;
    }

    pathSegment.ray.origin = intersect + (.0005f * glm::normalize(pathSegment.ray.direction));
}

 __device__
Geom* directRayToLight(
    PathSegment &pathSegment,
    glm::vec3 intersect,
    Geom *lights,
    int numLights,
    thrust::default_random_engine &rng) 
{
    // choose a random light
    thrust::uniform_real_distribution<float> lightsDist(0, numLights);
    int L = floor(lightsDist(rng));
    Geom *chosenLight = lights + L;

    // sample a point on the area light
    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 p(u01(rng) - 0.5f, 0.0f, u01(rng) - 0.5f);
    // transform the point from object to world space
    p = glm::vec3(chosenLight->transform * glm::vec4(p, 1.0f));
    // set new ray direction
    pathSegment.ray.direction = glm::normalize(p - intersect);

    // counterbalance the illumination from multiple lights by multiplying the
    // contribution of the light by the probability it was chosen
    pathSegment.color /= numLights;

    pathSegment.ray.origin = intersect + (.0005f * glm::normalize(pathSegment.ray.direction));
    return chosenLight;
}
