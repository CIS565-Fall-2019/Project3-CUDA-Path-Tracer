#pragma once

#include "intersections.h"

#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

__device__ void partition_by_bit(int *values, int size, unsigned int bit);
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
        thrust::default_random_engine &rng) {
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
        pathSegment.ray.origin = intersect + EPSILON * normal;
        pathSegment.ray.direction = diffuse_dir;
        pathSegment.remainingBounces--;


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
            glm::vec3 specular_dir = glm::reflect(pathSegment.ray.direction, normal);
            //no update on color
            //update ray
            pathSegment.ray.origin = intersect + EPSILON * normal;
            pathSegment.ray.direction = specular_dir;
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
                glm::vec3 wo = -pathSegment.ray.direction;
                glm::vec3 wi = diffuse_dir;
                //first update color -- actually not related to ray inforamtion at all for now
                pathSegment.color *= m.color * (float)InvPi;
                //apply lambert's law
                float lightTerm = glm::abs(glm::dot(wi, normal));
                pathSegment.color *= lightTerm;
                //update ray
                pathSegment.ray.origin = intersect;
                pathSegment.ray.direction = diffuse_dir;
            }
            else
            {
                glm::vec3 specular_dir = glm::reflect(pathSegment.ray.direction, normal);
                //no update on color
                //update ray
                pathSegment.ray.origin = intersect;
                pathSegment.ray.direction = specular_dir;
            }
        }
        pathSegment.remainingBounces--;

    }
    else if (m.hasRefractive > 0)
    {
        if (pathSegment.remainingBounces <= 0)
        {
            return;
        }
        //under construction -- two stuffs, come in or come out it should be different

    }

}


////test radix -- values should be unsigned as the most significant bit will affect a lot
//__global__ void radix_sort(int max_bound_bit, int values_size, int *values)
//{
//    int  bit;
//    for (bit = 0; bit < max_bound_bit; ++bit)
//    {
//        partition_by_bit(values, values_size, bit);
//        __syncthreads();
//    }
//}
//
//
//template<class T>
//__device__ T plus_scan(T *x)
//{
//    unsigned int i = threadIdx.x; // id of thread executing this instance
//    unsigned int n = blockDim.x;  // total number of threads in this block
//    unsigned int offset;          // distance between elements to be added
//
//    for (offset = 1; offset < n; offset *= 2) {
//        T t;
//
//        if (i >= offset)
//            t = x[i - offset];
//
//        __syncthreads();
//
//        if (i >= offset)
//            x[i] = t + x[i];      // i.e., x[i] = x[i] + x[i-1]
//
//        __syncthreads();
//    }
//    return x[i];
//}
//
//__device__ void partition_by_bit(int *values, int values_size, unsigned int bit)
//{
//    unsigned int i = threadIdx.x;
//    unsigned int size = blockDim.x;
//    unsigned int x_i = values[i];          // value of integer at position i
//    unsigned int p_i = (x_i >> bit) & 1;   // value of bit at position bit
//
//    // Replace values array so that values[i] is the value of bit bit in
//    // element i.
//    values[i] = p_i;
//
//    // Wait for all threads to finish this.
//    __syncthreads();
//
//    // Now the values array consists of 0's and 1's, such that values[i] = 0
//    // if the bit at position bit in element i was 0 and 1 otherwise.
//
//    // Compute number of True bits (1-bits) up to and including values[i], 
//    // transforming values[] so that values[i] contains the sum of the 1-bits
//    // from values[0] .. values[i]
//    unsigned int T_before = plus_scan(values);
//    /*
//        plus_scan(values) returns the total number of 1-bits for all j such that
//        j <= i. This is assigned to T_before, the number of 1-bits before i
//        (includes i itself)
//    */
//
//    // The plus_scan() function does not return here until all threads have
//    // reached the __syncthreads() call in the last iteration of its loop
//    // Therefore, when it does return, we know that the entire array has had
//    // the prefix sums computed, and that values[size-1] is the sum of all
//    // elements in the array, which happens to be the number of 1-bits in 
//    // the current bit position.
//    unsigned int T_total = values[size - 1];
//    // T_total, after the scan, is the total number of 1-bits in the entire array.
//
//    unsigned int F_total = size - T_total;
//    /*
//        F_total is the total size of the array less the number of 1-bits and hence
//        is the number of 0-bits.
//    */
//    __syncthreads();
//
//    /*
//        The value x_i must now be put back into the values array in the correct
//        position. The array has to satisfy the condition that all values with a 0 in
//        the current bit position must precede all those with a 1 in that position
//        and it must be stable, meaning that if x_j and x_k both had the same bit
//        value before, and j < k, then x_j must precede x_k after sorting.
//
//        Therefore, if x_i had a 1 in the current bit position before, it must now
//        be in the position such that all x_j that had a 0 precede it, and all x_j
//        that had a 1 in that bit and for which j < i, must precede it. Therefore
//        if x_i had a 1, it must go into the index T_before-1 + F_total, which is the
//        sum of the 0-bits and 1-bits that preceded it before (subtracting 1 since
//        T_before includes x_i itself).
//
//        If x_i has a 0 in the current bit position, then it has to be "slid" down
//        in the array before all x_j such that x_j has a 1 in the current bit, but
//        no farther than that. Since there are T_before such j, it has to go to
//        position i - T_before.  (There are T_before such j because x_i had a zero,
//        so in the prefix sum, it does not contribute to the sum.)
//    */
//    if (p_i)
//        values[T_before - 1 + F_total] = x_i;
//    else
//        values[i - T_before] = x_i;
//    /*
//       The interesting thing is that no two values will be placed in the same
//       position. I.e., this is a permutation of the array.
//
//       Proof: Suppose that x_i and x_j both end up in index k. There are three
//       cases:
//         Case 1. x_i and x_j have a 1 in the current bit position
//         Since F_total is the same for all threads, this implies that T_before must
//         be the same for threads i and j. But this is not possible because one must
//         precede the other and therefore the one that precedes it must have smaller
//         T_before.
//
//         Case 2.  x_i and x_j both have a 0 in the current bit position.
//         Since they both are in k, we have
//             k = i - T_bef_i = j - T_Bef_j  or
//             i - j = T_bef_i - T_bef_j
//         Assume i > j without loss of generality.  This implies that the number of
//         1-bits from position j+1 to position i-1 (since both x_j and x_i have
//         0-bits) is i-j. But that is impossible since there are only i-j-2 positions
//         from j+1 to i-1.
//
//         Case 3. x_i and x_j have different bit values.
//         Assume without loss of generality that x_j has the 0-bit and x_i, the 1-bit.
//         T_before_j is the number of 1 bits in positions strictly less than j,
//         because there is a 0 in position j. The total number of positions less than
//         j is j, since the array is 0-based. Therefore:
//
//         j-T_before_j is the number of 0-bits in positions strictly less than j.
//         This must be strictly less than F_total, since x_j has a 0 in position j,
//         so there is at least one more 0 besides those below position j. Hence:
//
//         (1)    F_total > j - T_before_j
//
//         Turning to i, T_before_i is at least 1, since x_i has a 1 in its bit. So,
//         T_before_i - 1 is at least 0, and
//
//         (2)    T_before_i - 1 + F_total >= F_total.
//
//         Therefore, combining (1) and (2)
//
//         (3)   T_before_i - 1 + F_total >= F_total
//                                        >  j - T_before_j
//
//         But if x_i and x_j map to the same position, then
//
//         (4)   j - T_before_j  = T_before_i - 1 + F_total
//                               > j - T_before_j
//
//         which is a contradiction since a number cannot be greater than itself!
//
//         Therefore it is impossible for x_i and x_j to be placed in the same index
//         if i != j.
//
//    */
//
//}