#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtx/transform.hpp"
#include "glm/gtc/matrix_inverse.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include <thrust/partition.h>
#include <thrust/sort.h>

#include "efficient.h"

#define ERRORCHECK		1

//=======================
// FEATURE SWITCH
//=======================

//Basic Features
#define SORTBYMATERIAL	0
#define FIRSTCACHE		1

// Advance Features
#define ANTIALIASING	0
#define MOTIONBLUR		0
#define DEPTHOFFIELD	0
#define WORKEFFCOMP     1

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
static ShadeableIntersection * dev_intersections_cache = NULL;
static int * materials_to_sort = NULL;
int * dev_paths_idx = NULL;


void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
	cudaMalloc(&dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections_cache, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&materials_to_sort, pixelcount * sizeof(int));
	cudaMemset(materials_to_sort, 0, pixelcount * sizeof(int));

	cudaMalloc(&dev_paths_idx, pixelcount * sizeof(int));
	cudaMemset(dev_paths_idx, 0, pixelcount * sizeof(int));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);

    // TODO: clean up any extra device memory you created
	cudaFree(dev_intersections_cache);
	cudaFree(materials_to_sort);
	cudaFree(dev_paths_idx);
	
    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments, int * dev_paths_idx)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// TODO: implement antialiasing by jittering the ray
#if ANTIALIASING 
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, segment.remainingBounces, index);
		thrust::uniform_real_distribution<float> u01(-0.49, 0.49);
		x += u01(rng);
		y += u01(rng);
#endif
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		dev_paths_idx[index] = index;
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
	,int iter
	, int *dev_paths_idx
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[dev_paths_idx[path_index]];

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;

		float alpha = 0.9;
		glm::mat4 motion = glm::mat4(1.0f, 0.0f, 0.0f, iter*0.0f,
			0.0f, 1.0f, 0.0f, iter*0.05f,
			0.0f, 0.0f, 1.0f, iter*0.0f,
			0.0f, 0.0f, 0.0f, 1.0f);

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

#if MOTIONBLUR
				geom.transform			= alpha*geom.transformInitial + (1 - alpha)*motion*geom.transformInitial;
				geom.inverseTransform	= glm::inverse(geom.transform);
				geom.invTranspose		= glm::inverseTranspose(geom.transform);
#endif

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

		if (hit_geom_index == -1)
		{
			intersections[dev_paths_idx[path_index]].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[dev_paths_idx[path_index]].t = t_min;
			intersections[dev_paths_idx[path_index]].materialId = geoms[hit_geom_index].materialid;
			intersections[dev_paths_idx[path_index]].surfaceNormal = normal;
			intersections[dev_paths_idx[path_index]].intersectionPoint = intersect_point;
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
	, int depth
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
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
        float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
        pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
        pathSegments[idx].color *= u01(rng); // apply some noise because why not
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
    }
  }
}



// NEW SHADER!


__global__ void shadeMaterial (
	int iter
	, int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, int depth
	, int *dev_paths_idx
	) {
	
	int idxd = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = dev_paths_idx[idxd];

	if (idxd < num_paths && (pathSegments[idx].remainingBounces > 0))
	
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];

		if (intersection.t > 0.0f) { // if the intersection exists...

			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
			thrust::uniform_real_distribution<float> u01(0, 1);

			Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegments[idx].color *= (materialColor * material.emittance);
				pathSegments[idx].remainingBounces = 0;
				dev_paths_idx[idxd] = -1;
			}
			else { 
				// Random sample probabiltiy between the three types of materials
				// material.hasReflective + material.hasRefractive + // 1- sum is diffuese material.
				// assert(material.hasReflective + material.hasRefractive <= 0.1f);
				char scase = 'D';
				if (material.hasReflective == 0.0f && material.hasRefractive == 0.0f) {
					scase = 'D'; // Diffuse
				}
				else if (material.hasRefractive > 0.0f && material.hasReflective == 0.0f) {
					scase = 'F'; // Refractive
				}
				else if (material.hasReflective > 0.0f && material.hasRefractive == 0.0f) {
					scase = 'R'; // Reflective
				}
				else if (material.hasReflective > 0.0f && material.hasRefractive > 0.0f) {
					// randomly pick between the three cases
					float rand = u01(rng);
					float reflect = material.hasReflective;
					float refract = material.hasRefractive + reflect;
					
					if (rand <= reflect) { scase = 'R'; }
					else if (rand > reflect && rand <= refract) { scase = 'F'; }
					else { scase = 'D';}//rand > refract -> diffuese 
				}

				switch (scase) {

					case 'D': // DIFFUSE
						pathSegments[idx].color *= materialColor;
						pathSegments[idx].ray.direction = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
						pathSegments[idx].ray.origin = intersection.intersectionPoint;
						pathSegments[idx].remainingBounces -= 1;

						break;
				
					case 'F': // REFRACTION
						//check change of media
						float cosTheta = glm::dot(glm::normalize(pathSegments[idx].ray.direction), (intersection.surfaceNormal));

						if (cosTheta > 0.0f) { //  Object to Air
							glm::vec3 tmp = glm::refract(pathSegments[idx].ray.direction, (glm::vec3(-1.0f)*intersection.surfaceNormal), material.indexOfRefraction);
							if (glm::length(tmp) > 0.0000001f) {
								pathSegments[idx].color *= material.specular.color;
								pathSegments[idx].ray.direction = tmp;
								pathSegments[idx].ray.origin = intersection.intersectionPoint;
								pathSegments[idx].remainingBounces -= 1;
							}
							else {
								// Reflection
								pathSegments[idx].color *= material.specular.color;
								pathSegments[idx].ray.direction = glm::reflect(pathSegments[idx].ray.direction, (glm::vec3(-1.0f)*intersection.surfaceNormal));
								pathSegments[idx].ray.origin = intersection.intersectionPoint;
								pathSegments[idx].remainingBounces -= 1;
							}
						}
						else { //  Air to Object
							glm::vec3 tmp = glm::refract(pathSegments[idx].ray.direction, glm::vec3(1.0f)*intersection.surfaceNormal, (0.1f/material.indexOfRefraction));
							if (glm::length(tmp) > 0.0000001f) {
								pathSegments[idx].color *= material.specular.color;
								pathSegments[idx].ray.direction = tmp;
								pathSegments[idx].ray.origin = intersection.intersectionPoint;
								pathSegments[idx].remainingBounces -= 1;
							}
							else {
								// Reflection
								pathSegments[idx].color *= material.specular.color;
								pathSegments[idx].ray.direction = glm::reflect(pathSegments[idx].ray.direction, (glm::vec3(1.0f)*intersection.surfaceNormal));
								pathSegments[idx].ray.origin = intersection.intersectionPoint;
								pathSegments[idx].remainingBounces -= 1;
							}
						}
						break; 
				
					case 'R': // REFLECTION
						pathSegments[idx].color *= material.specular.color;
						pathSegments[idx].ray.direction = glm::reflect(pathSegments[idx].ray.direction, intersection.surfaceNormal);
						pathSegments[idx].ray.origin = intersection.intersectionPoint;
						pathSegments[idx].remainingBounces -= 1;
						break;
				} 
				
				// offset ray
				pathSegments[idx].ray.origin = pathSegments[idx].ray.origin + (pathSegments[idx].ray.direction)*glm::vec3(0.015f);// EPSILON);
				// clamp color
				pathSegments[idx].color = glm::clamp(pathSegments[idx].color, glm::vec3(0.0f), glm::vec3(1.0));
				
				if(pathSegments[idx].remainingBounces == 0)
					dev_paths_idx[idxd] = -1;
			}
		}
		else {// If there was no intersection, color the ray black.
				pathSegments[idx].color = glm::vec3(0.0f);
				pathSegments[idx].remainingBounces = 0;
				dev_paths_idx[idxd] = -1;
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
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

struct hasExited
{
	__host__ __device__
		bool operator()(const PathSegment &dev_path)
	{return (dev_path.remainingBounces > 0);}
};

struct materialCmp{
	__host__ __device__
		bool operator()(const ShadeableIntersection& m1, const ShadeableIntersection& m2) {
		return m1.materialId < m2.materialId;
	}
};

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
    //   * TODO: Stream compact away all of the terminated paths.
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

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths, dev_paths_idx);
	checkCUDAError("Error in generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;



	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	dim3 numblocksPathSegmentTracing;

	bool iterationComplete = false;
	
	
	while (!iterationComplete) {

#if FIRSTCACHE
		if (depth == 0) {
			if (iter == 1) {//cache first bounce
				// clean shading chunks
				cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
				// tracing
				numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
				computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
					depth
					, num_paths
					, dev_paths
					, dev_geoms
					, hst_scene->geoms.size()
					, dev_intersections
					, iter
					, dev_paths_idx
					);
				cudaMemcpy(dev_intersections_cache, dev_intersections, 
					pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
				checkCUDAError("error in trace-one-bounce");
				cudaDeviceSynchronize();
			}
			else {// use cached bounce!
				// clean shading chunks
				cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
				
				cudaMemcpy(dev_intersections, dev_intersections_cache,
					pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		}
		else { // non-zero depth
			// clean shading chunks
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

			// tracing
			numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
			computeIntersections <<<numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				, iter
				, dev_paths_idx
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
#else
		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		// tracing
		numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			, iter
			, dev_paths_idx
			);
		checkCUDAError("error in trace-one-bounce");
		cudaDeviceSynchronize();
#endif

		depth++;

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d >>> (
		iter,
		num_paths,
		dev_intersections,
		dev_paths,
		dev_materials,
		depth,
		dev_paths_idx
		);

#if WORKEFFCOMP
		//Compute stream compaction here
		num_paths = StreamCompaction::Efficient::compact(num_paths, dev_paths_idx);
		iterationComplete = (num_paths <= 0) || (depth > traceDepth);
		//cout << "num_paths " << num_paths << endl;
#else
		//Compute stream compaction here
		dev_path_end = thrust::partition(thrust::device, dev_paths_idx, dev_paths_idx + num_paths, hasExited());
		num_paths = dev_path_end - dev_paths;
		iterationComplete = (num_paths <= 0) || (depth > traceDepth);
#endif

#if SORTBYMATERIAL
		sort by matrial 
		if (iterationComplete == false) {
			thrust::stable_sort_by_key(thrust::device, dev_intersections, dev_intersections+num_paths, dev_paths, materialCmp());
		}
#endif

	}

	num_paths = pixelcount;

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
