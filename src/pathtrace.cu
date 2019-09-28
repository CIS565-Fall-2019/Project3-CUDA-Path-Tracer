#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtc/matrix_inverse.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define TEST_RADIX 0
#define SORT_MATERIAL 0
#define CACHE_FIRST_BOUNCE 0
#define ANTI_ALIASING 0
#define MOTION_BLUR 1

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
//static first_bounce_intersection and flag -- need to turn false when we change camera
static ShadeableIntersection * dev_first_bounce_intersections = NULL;
static PathSegment * dev_first_bounce_paths = NULL;
//for radix sort
static int n_bit_material_bound = -1;
void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    //same number of path as pixel
  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_first_bounce_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_first_bounce_paths, pixelcount * sizeof(PathSegment));

    //calculate the bit number we need to concern
    if (n_bit_material_bound == -1)
    {
        n_bit_material_bound = std::log2(hst_scene->materials.size()) + 1;
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_first_bounce_intersections);
    cudaFree(dev_first_bounce_paths);
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens -- need to modify if I want to implement
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
        #if ANTI_ALIASING
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float rn_x = u01(rng);
        float rn_y = u01(rng);
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)(x + rn_x) - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)(y + rn_y) - (float)cam.resolution.y * 0.5f)
        );

        #else
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);


        #endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
        segment.terminated = false;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
	, Geom * geoms
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		// naive parse through global geoms

		for (int i = 0; i < geoms_size; i++)
		{
			Geom & geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
			}
			// TODO: add more intersection tests here... triangle? metaball? CSG?

			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
			}
		}

        //no hit
		if (hit_geom_index == -1)
		{
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			intersections[path_index].surfaceNormal = normal;
            intersections[path_index].outside = outside;
		}
	}
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial (
  int iter
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray -- should light source be + instead of mult?
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
        //should we terminate? -- yes
        pathSegments[idx].remainingBounces = -1;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner

      //basic implementation of bsdf
      else {
          //if specular, completely depend on the reflected color
          glm::vec3 intersec_pos = pathSegments[idx].ray.direction * intersection.t + pathSegments[idx].ray.origin;
          if (material.hasRefractive)
          {
              //first determine whether it is inside the object or not by computing the cosTheta of output ray direction -- which is its z value, as it is normalized, I will add to utility though
              scatterRay(pathSegments[idx], intersec_pos, intersection.surfaceNormal, material, rng, intersection.outside);
          }
          else
          {
              scatterRay(pathSegments[idx], intersec_pos, intersection.surfaceNormal, material, rng);
          }
      }

      // fake implementation
      //else {
      //  float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
      //  pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
      //  pathSegments[idx].color *= u01(rng); // apply some noise because why not
      //}

    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
      pathSegments[idx].remainingBounces = -1;//no further bouncing -- help stream compaction
    }
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
        //should here be average?
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

//need to change position later
struct have_more_bounce
{
    __host__ __device__
        bool operator()(const PathSegment p)
    {
        return p.remainingBounces >= 0;
    }
};


//sort rays based on material id -- failed..
//__global__ void compute_b_e(int values_size, int* values, int* dev_b, int* dev_e, unsigned int bit)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < values_size)
//    {
//        int x_i = values[idx];          // value of integer at position i
//        unsigned int p_i = (x_i >> bit) & 1;
//
//        if (p_i)
//        {
//            dev_b[idx] = 1;
//        }
//        else
//        {
//            dev_e[idx] = 1;
//        }
//    }
//}
//
//__global__ void compute_t(int values_size, int* dev_e, int* dev_t, int* dev_f)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < values_size)
//    {
//        int total_false = dev_e[values_size - 1] + dev_f[values_size - 1];
//        dev_t[idx] = idx - dev_f[idx] + total_false;
//    }
//}
//
//__global__ void compute_d(int values_size, int* dev_b, int* dev_t, int* dev_f, int* dev_d)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < values_size)
//    {
//        bool has_bit = dev_b[idx] == 1;
//        int address;
//        if (has_bit)
//        {
//            address = dev_t[idx];
//            //dev_d[address] = dev_values[idx];  --test
//        }
//        else
//        {
//            address = dev_f[idx];
//            //dev_d[address] = dev_values[idx];  --test
//        }
//        dev_d[idx] = address;
//    }
//}
//
//__global__ void apply_address(int values_size, PathSegment* dev_paths, PathSegment* dev_output_paths, ShadeableIntersection* dev_intersections, ShadeableIntersection* dev_output_intersections, int* dev_d)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < values_size)
//    {
//        int address = dev_d[idx]; //address to store to
//        dev_output_paths[address] = dev_paths[idx];
//        dev_output_intersections[address] = dev_intersections[idx];
//    }
//}
//
//__global__ void setup_material_array(int values_size, ShadeableIntersection* dev_input_intersections, int* dev_materialIds)
//{
//    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//    if (idx < values_size)
//    {
//        dev_materialIds[idx] = dev_input_intersections[idx].materialId;
//    }
//
//
//}
//
//void sort_by_materialID(int values_size)
//{
//    //first allocate 5 intermediate buffers  -- similar in slides
//    int* dev_b;
//    int* dev_e;
//    int* dev_f;
//    int* dev_t;
//    int* dev_d;
//    int* dev_materialIds;
//
//    //intermediate buffer
//    PathSegment* dev_output_paths;
//    PathSegment* dev_input_paths;
//    ShadeableIntersection* dev_output_intersections;
//    ShadeableIntersection* dev_input_intersections;
//
//    cudaMalloc(&dev_b, values_size * sizeof(int));
//    cudaMalloc(&dev_e, values_size * sizeof(int));
//    cudaMalloc(&dev_f, values_size * sizeof(int));
//    cudaMalloc(&dev_t, values_size * sizeof(int));
//    cudaMalloc(&dev_d, values_size * sizeof(int));
//    cudaMalloc(&dev_materialIds, values_size * sizeof(int));
//    cudaMalloc(&dev_output_paths, values_size * sizeof(PathSegment));
//    cudaMalloc(&dev_output_intersections, values_size * sizeof(ShadeableIntersection));
//    cudaMalloc(&dev_input_paths, values_size * sizeof(PathSegment));
//    cudaMalloc(&dev_input_intersections, values_size * sizeof(ShadeableIntersection));
//
//    cudaMemcpy(dev_input_paths, dev_paths, values_size * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
//    cudaMemcpy(dev_input_intersections, dev_intersections, values_size * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
//
//
//    for (int bit = 0; bit < n_bit_material_bound; ++bit)
//    {
//        cudaMemset(dev_b, 0, values_size * sizeof(int));
//        cudaMemset(dev_e, 0, values_size * sizeof(int));
//        cudaMemset(dev_d, 0, values_size * sizeof(int));
//        cudaMemset(dev_materialIds, 0, values_size * sizeof(int));
//
//        const int blockSize = 128;
//        dim3 blocksPerGrid((values_size + blockSize - 1) / blockSize);
//
//        //store material id into the int array and use that to do radix sort
//        setup_material_array << <blockSize, blocksPerGrid >> > (values_size, dev_input_intersections, dev_materialIds);
//
//        //compute array b and e
//        compute_b_e<<<blockSize, blocksPerGrid>>>(values_size, dev_materialIds, dev_b, dev_e, bit);
//
//        //thrust to compute the scan of e -- if we input device data, then we should apply thrust::device, otherwise doesn't work
//        thrust::device_ptr<int> dev_temp_e(dev_e);
//        thrust::device_ptr<int> dev_temp_f(dev_f);
//        thrust::exclusive_scan(thrust::device, dev_temp_e, dev_temp_e + values_size, dev_temp_f); //doesn't work, don't know why
//        //compute total false by adding last element in dev_f and dev_e -- no, because need to cpy back, only for that number
//        //directly add in in each kernel
//        compute_t <<<blockSize, blocksPerGrid >> > (values_size, dev_e, dev_t, dev_f);
//
//        //compute the corresponding address of each element in new array and store in dev_d
//        compute_d <<<blockSize, blocksPerGrid >> > (values_size, dev_b, dev_t, dev_f, dev_d);
//
//        //apply dev_d's address back to pathSegment and shaderIntersection for next round
//        apply_address << <blockSize, blocksPerGrid >> > (values_size, dev_paths, dev_output_paths, dev_intersections, dev_output_intersections, dev_d);
//        //apply the output_paths and output_intersections back to dev_paths and dev_intersections
//        cudaMemcpy(dev_paths, dev_output_paths, values_size * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
//        cudaMemcpy(dev_intersections, dev_output_intersections, values_size * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
//    }
//
//    //no need to store another time because we have already done that in the last iteration in the loop
//
//    //Free all cuda stuff
//
//    cudaFree(dev_b);
//    cudaFree(dev_e);
//    cudaFree(dev_f);
//    cudaFree(dev_t);
//    cudaFree(dev_d);
//    cudaFree(dev_materialIds);
//
//    cudaFree(dev_output_paths);
//    cudaFree(dev_output_intersections);
//    cudaFree(dev_input_paths);
//    cudaFree(dev_input_intersections);
//}




//thanks for Jie Meng's help
struct material_comparison
{
    __host__ __device__
        bool operator()(const ShadeableIntersection& i1, const ShadeableIntersection& i2)
    {
        return i1.materialId < i2.materialId;
    }
};

void sort_by_material(int num_paths, PathSegment* dev_paths, ShadeableIntersection* dev_intersections)
{
    //wrapped by device_ptr
    thrust::device_ptr<PathSegment> dev_paths_ptr(dev_paths);
    thrust::device_ptr<ShadeableIntersection> dev_intersections_ptr(dev_intersections);

    thrust::sort_by_key(thrust::device, dev_intersections_ptr, dev_intersections_ptr + num_paths, dev_paths_ptr, material_comparison());
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter) {
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths. -- but you still need their data right?
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    //if there is motion blur, we update the transformation of the geometry
#if MOTION_BLUR
    //actually, the velocity only determine how much we want to move the object
    //float timeStep = iter; //wrong, fly out of scene
    float timeStep = 1 / (hst_scene->state.iterations * 0.1f); // depend on how long you want the motion to arrive on your expected position -- here is 500 iteration
    for (int i = 0; i < hst_scene->geoms.size(); i++)
    {
        hst_scene->geoms[i].translation += hst_scene->geoms[i].velocity * timeStep;
        hst_scene->geoms[i].transform = utilityCore::buildTransformationMatrix(hst_scene->geoms[i].translation, hst_scene->geoms[i].rotation, hst_scene->geoms[i].scale);
        hst_scene->geoms[i].inverseTransform = glm::inverse(hst_scene->geoms[i].transform);
        hst_scene->geoms[i].invTranspose = glm::inverseTranspose(hst_scene->geoms[i].transform);
    }

    cudaMemcpy(dev_geoms, &(hst_scene->geoms)[0], hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    checkCUDAError("motion blur error");

#endif 

#if CACHE_FIRST_BOUNCE && !ANTI_ALIASING && !MOTION_BLUR
    //first iteration, we need to generate Ray and cache
    if (iter == 1)
    {
        generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
        checkCUDAError("generate camera ray");

        //cache the ray
        cudaMemcpy(dev_first_bounce_paths, dev_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);

        int depth = 0;
        PathSegment* dev_path_end = dev_paths + pixelcount; //the tail of path segment array
        int num_paths = dev_path_end - dev_paths; //is that the same as pixel count? -- no when antialiasing, do we need to change?

        // --- PathSegment Tracing Stage ---
        // Shoot ray into scene, bounce between objects, push shading chunks

        bool iterationComplete = false;
        //create a intermediate buffer
        while (!iterationComplete) {

            // clean shading chunks
            cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
            // tracing
            dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
            //cache the intersection
            if (depth == 0)
            {
                cudaMemcpy(dev_first_bounce_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
            }
            depth++;


            // TODO:
            // --- Shading Stage ---
            // Shade path segments based on intersections and generate new rays by
          // evaluating the BSDF.
          // Start off with just a big kernel that handles all the different
          // materials you have in the scenefile.

          // TODO: compare between directly shading the path segments and shading
          // path segments that have been reshuffled to be contiguous in memory.

#if SORT_MATERIAL
    //reshuffle the pathSegments
            sort_by_material(num_paths, dev_paths, dev_intersections);
#endif

            shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
                iter,
                num_paths,
                dev_intersections,
                dev_paths,
                dev_materials
                );
            // TODO: should be based off stream compaction results, and even shot more rays
            // update the dev_path and num_paths -- if determine no_more_bounce by remainingBounce == -1, then our ray will be termianted and no longer take account
            PathSegment* dev_paths_end_result = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, have_more_bounce());
            num_paths = dev_paths_end_result - dev_paths;
            if (depth >= traceDepth || num_paths <= 0)
            {
                iterationComplete = true;
            }
        }

        //remember to recover its num_paths
        num_paths = dev_path_end - dev_paths;
        // Assemble this iteration and apply it to the image

        dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
        finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
    }
    else
    {

        //copy dev_first_bounce_paths and dev_first_bounce_intersections to dev_paths and dev_intersections
        cudaMemcpy(dev_paths, dev_first_bounce_paths, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
        cudaMemcpy(dev_intersections, dev_first_bounce_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        //set the initial depth to 1
        int depth = 1;
        PathSegment* dev_path_end = dev_paths + pixelcount; //the tail of path segment array
        int num_paths = dev_path_end - dev_paths;

        bool iterationComplete = false;
        //no need to generate ray, directly go into the while loop
        while (!iterationComplete) {
            //first go shading and then update the depth
#if SORT_MATERIAL
    //reshuffle the pathSegments
            sort_by_material(num_paths, dev_paths, dev_intersections);
#endif
            dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
            shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
                iter,
                num_paths,
                dev_intersections,
                dev_paths,
                dev_materials
                );
            // TODO: should be based off stream compaction results, and even shot more rays
            // update the dev_path and num_paths -- if determine no_more_bounce by remainingBounce == -1, then our ray will be termianted and no longer take account
            PathSegment* dev_paths_end_result = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, have_more_bounce());
            num_paths = dev_paths_end_result - dev_paths;
            if (depth >= traceDepth || num_paths <= 0)
            {
                iterationComplete = true;
                continue;
            }


            // reset dev_intersections
            cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
            // tracing
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections
                );
            checkCUDAError("trace one bounce");
            cudaDeviceSynchronize();
            depth++;
        }

        //remember to recover its num_paths
        num_paths = dev_path_end - dev_paths;
        // Assemble this iteration and apply it to the image

        dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
        finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
    }
#else
	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount; //the tail of path segment array
	int num_paths = dev_path_end - dev_paths; //is that the same as pixel count? -- no when antialiasing, do we need to change?

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
  //create a intermediate buffer
	while (!iterationComplete) {

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
    //cudaMemset(dev_intersections, 0, num_paths * sizeof(ShadeableIntersection));
	// tracing
	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
	computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		depth
		, num_paths
		, dev_paths
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_intersections
		);
	checkCUDAError("trace one bounce");
	cudaDeviceSynchronize();
	depth++;


	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.

  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.

#if SORT_MATERIAL
    //reshuffle the pathSegments
    sort_by_material(num_paths,dev_paths,dev_intersections);
#endif

  shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
    num_paths,
    dev_intersections,
    dev_paths,
    dev_materials
  );
  // TODO: should be based off stream compaction results, and even shot more rays
  // update the dev_path and num_paths -- if determine no_more_bounce by remainingBounce == -1, then our ray will be termianted and no longer take account
  PathSegment* dev_paths_end_result = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, have_more_bounce());
  num_paths = dev_paths_end_result - dev_paths;
  if (depth >= traceDepth || num_paths <= 0)
  {
      iterationComplete = true;
  }
	}
    //remember to recover its num_paths
    num_paths = dev_path_end - dev_paths;
    // Assemble this iteration and apply it to the image

    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths);
#endif


    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
