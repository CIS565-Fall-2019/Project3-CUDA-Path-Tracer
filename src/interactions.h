#pragma once

#include "intersections.h"

#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define EPSILON_OFFSET 0.01f

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

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 * 
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive). --- only reflective is that right? diffuse shouldn't have reflected component
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

//it means that we will update the color directly in scatterRay and just need to call scatter Ray
__host__ __device__
void scatterRay(
		PathSegment & pathSegment,
        glm::vec3 intersect,
        glm::vec3 normal,
        const Material &m,
        thrust::default_random_engine &rng,
        bool outside = true) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the

    //first determine the material
    
    //*********TODO, modify such that each material types have a possibility to do its work. use randomly number generator to determine
    //diffuse case
    //if it is pure-diffuse, then it is the its material's color /invPi -- might have other diffuse
    if (m.hasReflective == 0 && m.hasRefractive == 0)
    {

        //if diffuse -- update color but not respawn ray
        pathSegment.color *= m.color;
        if (pathSegment.remainingBounces <= 0)
        {
            return;
        }
        // calculateRandomDirectionInHemisphere defined above.
        glm::vec3 diffuse_dir = calculateRandomDirectionInHemisphere(normal, rng);
        glm::vec3 wo = -pathSegment.ray.direction;
        glm::vec3 wi = diffuse_dir;
        //first update color -- actually not related to ray inforamtion at all for now
        //glm::vec3 temp_col = m.color * (float)InvPi;
        //apply lambert's law
        //float lightTerm = glm::abs(glm::dot(wi, normal));  //don't need, because we don't include pdf
        //update ray -- determine whether we should keep bouncing
        pathSegment.ray.origin = intersect + EPSILON_OFFSET * normal;
        pathSegment.ray.direction = diffuse_dir;


    }
    //if pure-specular, itself doesn't have color, all is about its reflected item's color
    else if (m.hasReflective > 0)
    {
        if (pathSegment.remainingBounces <= 0)
        {
            return;
        }
        //pure reflective when hasReflective is 1, otherwise, go 50/50
        if (m.hasReflective == 1)
        {
            glm::vec3 specular_dir = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
            //no update on color
            //update ray
            pathSegment.ray.origin = intersect + EPSILON_OFFSET * normal;
            pathSegment.ray.direction = specular_dir;
            //pathSegment.color *= m.specular.color;
            //pathSegment.color *= glm::abs(glm::dot(pathSegment.ray.direction, normal)) * m.color;
        }
        else
        {
            //use random generator to genrate a num
            thrust::uniform_real_distribution<float> u01(0, 1);
            float condition = u01(rng);
            //first use simple 5/5 version
            if (condition < 0.5)
            {
                glm::vec3 diffuse_dir = calculateRandomDirectionInHemisphere(normal, rng);
                //first update color -- actually not related to ray inforamtion at all for now
                pathSegment.color *= m.color;
                //update ray
                pathSegment.ray.origin = intersect + EPSILON_OFFSET * normal;
                pathSegment.ray.direction = diffuse_dir;
            }
            else
            {
                glm::vec3 specular_dir = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
                //no update on color
                //update ray
                pathSegment.ray.origin = intersect + EPSILON_OFFSET * normal;
                pathSegment.ray.direction = specular_dir;
                //pathSegment.color *= m.specular.color;
                //pathSegment.color *= glm::abs(glm::dot(pathSegment.ray.direction, normal)) * m.color;
            }
        }

    }
    else if (m.hasRefractive > 0)
    {
        if (pathSegment.remainingBounces <= 0)
        {
            return;
        }

        //the eta of air is always 1 we assume
        float eta_in = 1.0f;
        float eta_out = m.indexOfRefraction;
        if (!outside)
        {
            float temp = eta_in;
            eta_in = eta_out;
            eta_out = temp;
        }

        float eta = eta_in / eta_out;

        //then compute Schlick's_approximation
        float cos_theta = glm::dot(-pathSegment.ray.direction, normal);
        float r0 = pow((1 - m.indexOfRefraction) / (1 + m.indexOfRefraction), 2);
        float fresnel = r0 + (1 - r0) * pow(1 - cos_theta, 5);

        //get a random reflection 
        thrust::uniform_real_distribution<float> u01(0, 1);
        float condition = u01(rng);
        //we reflect if fresnel is larger than the random reflection possibility, internal reflect
        if (fresnel > condition)
        {
            glm::vec3 specular_dir = glm::normalize(glm::reflect(pathSegment.ray.direction, normal));
            //no update on color
            //update ray
            pathSegment.ray.origin = intersect + EPSILON_OFFSET * normal;
            pathSegment.ray.direction = specular_dir;
            
        }
        //else we refract
        {
            glm::vec3 refract_dir = glm::normalize(glm::refract(pathSegment.ray.direction, normal, eta));
            //no update on color
            //update ray
            pathSegment.ray.origin = intersect + EPSILON_OFFSET * pathSegment.ray.direction;
            pathSegment.ray.direction = refract_dir;
        }
    }

    pathSegment.remainingBounces--;
}