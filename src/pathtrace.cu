#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

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

struct is_path_complete {
	__host__ __device__ bool operator()(const PathSegment & pathSegment) {
		return pathSegment.remainingBounces > 0;
	}
};


struct compare_materials {
	__host__ __device__ bool operator()(const ShadeableIntersection &  shadeableIntersection1, const ShadeableIntersection &  shadeableIntersection2) {
		return shadeableIntersection1.materialId < shadeableIntersection2.materialId;
	}
};

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
static Light * dev_lights = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;
static float3 * dev_albedos = NULL;
static float3 * dev_normals = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_intersections_cache = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
clock_t timer, blurTimer;
double iteration_time = 0;
double total_time = 0;

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

	cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Light));
	cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Light), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_albedos, pixelcount * sizeof(float3));
	cudaMalloc(&dev_normals, pixelcount * sizeof(float3));

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
    segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
	glm::vec2 jitter = glm::vec2(0.0f, 0.0f);
	thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
	thrust::uniform_real_distribution<float> u01(0.0f, 1.0f);
	if (ANTI_ALIASING) {
		jitter = glm::vec2(u01(rng) - 0.5f, u01(rng) - 0.5f);
	}
		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x + jitter.x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y + jitter.y - (float)cam.resolution.y * 0.5f)
			);

		if (DEPTH_OF_FIELD) {
			glm::vec3 focus = segment.ray.origin + (cam.focalLength) * segment.ray.direction;
			segment.ray.origin.x += max(cam.radius, 0.0f) * (u01(rng) - 0.5f);
			segment.ray.origin.y += max(cam.radius, 0.0f) * (u01(rng) - 0.5f);
			segment.ray.direction = glm::normalize(focus - segment.ray.origin);
	}

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
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
	, float3 *dev_albedos
	, float3 *dev_normals
	, int iter
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
		}


		if (DENOISE && iter == 0) {
			dev_normals[path_index].x = (intersections[path_index].t < 0.0) ? 0.0f : intersections[path_index].surfaceNormal.x;
			dev_normals[path_index].y = (intersections[path_index].t < 0.0) ? 0.0f : intersections[path_index].surfaceNormal.y;
			dev_normals[path_index].z = (intersections[path_index].t < 0.0) ? 0.0f : intersections[path_index].surfaceNormal.z;
			dev_albedos[path_index].x = (intersections[path_index].t < 0.0) ? 0.0f : pathSegment.color.x;
			dev_albedos[path_index].y = (intersections[path_index].t < 0.0) ? 0.0f : pathSegment.color.y;
			dev_albedos[path_index].z = (intersections[path_index].t < 0.0) ? 0.0f : pathSegment.color.z;

		}
	}
}

__global__ void computeNewLocations(
	Geom * geoms
	, int geoms_size
	, float dt
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= geoms_size) {
		return;
	}

	geoms[idx].translation = geoms[idx].translation + geoms[idx].velocity * dt;

	glm::mat4 translationMat = glm::translate(glm::mat4(), geoms[idx].translation);
	glm::mat4 rotationMat = glm::rotate(glm::mat4(), geoms[idx].rotation.x * (float)PI / 180, glm::vec3(1, 0, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), geoms[idx].rotation.y * (float)PI / 180, glm::vec3(0, 1, 0));
	rotationMat = rotationMat * glm::rotate(glm::mat4(), geoms[idx].rotation.z * (float)PI / 180, glm::vec3(0, 0, 1));
	glm::mat4 scaleMat = glm::scale(glm::mat4(), geoms[idx].scale);
	

	geoms[idx].transform = translationMat * rotationMat * scaleMat;
	geoms[idx].inverseTransform = glm::inverse(geoms[idx].transform);
	geoms[idx].invTranspose = glm::inverseTranspose(geoms[idx].transform);
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
	, int depth
  , int num_paths
	, ShadeableIntersection * shadeableIntersections
	, PathSegment * pathSegments
	, Material * materials
	, int num_lights
	, Light * lights
	, int num_geoms
	, Geom * geoms
	)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pathSegments[idx].remainingBounces <= 0) {
	  return;
  }
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
			  pathSegments[idx].color *= (materialColor * material.emittance * glm::abs(glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal)));
			  pathSegments[idx].remainingBounces = 0;
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else {
		  if (pathSegments[idx].remainingBounces > 0) {
			  scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, material, rng);
			  if (pathSegments[idx].remainingBounces == 0 && DIRECT_LIGHTING) {
				  directLight(num_lights, lights, num_geoms, geoms, materials, getPointOnRay(pathSegments[idx].ray, intersection.t), intersection.surfaceNormal, pathSegments[idx], rng);
			  }
			  //pathSegments[idx].color = glm::clamp(pathSegments[idx].color, glm::vec3(0.0f), glm::vec3(1.0f));
		  }
      }
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
	  pathSegments[idx].remainingBounces = 0;
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

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4 *pbo, int frame, int iter, float3* albedos, float3* normals) {
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

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;;

	if (TIMER) {
		timer = clock();
	}
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	if (iter > 0 && MOTION_BLUR) {
		computeNewLocations << <numblocksPathSegmentTracing, blockSize1d >> > (
			dev_geoms, 
			hst_scene->geoms.size(), 
			((double)(clock() - blurTimer)) / CLOCKS_PER_SEC);
	}

	blurTimer = clock();

  bool iterationComplete = false;
	while (!iterationComplete) {

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	if (depth == 0 && CACHE_FIRST_BOUNCE && !ANTI_ALIASING && !MOTION_BLUR) {
		// tracing
		if (iter == 1) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections_cache
				, dev_albedos
				, dev_normals
				, iter
				);
			checkCUDAError("trace one bounce");
		}
		cudaMemcpy(dev_intersections, dev_intersections_cache, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
	}
	else {
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			, dev_albedos
			, dev_normals
			, iter
			);
		checkCUDAError("trace one bounce");
	}
	cudaDeviceSynchronize();
	depth++;

	if (SORT_BY_MATERIAL && num_paths > 0) {
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compare_materials());
	}

	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.

	shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
		iter,
		depth,
		num_paths,
		dev_intersections,
		dev_paths,
		dev_materials,
		hst_scene->lights.size(),
		dev_lights,
		hst_scene->geoms.size(),
		dev_geoms
		);


  if (STREAM_COMPACT) {
	 dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, is_path_complete());
	  num_paths = dev_path_end - dev_paths;
	}

  

	  iterationComplete = num_paths == 0 || depth > traceDepth; // TODO: should be based off stream compaction results.
	  
	}
	if (TIMER) {
		timer = clock() - timer;
		iteration_time = ((double)timer) / CLOCKS_PER_SEC;
		total_time += iteration_time;
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
	// Retrieve image from GPU
	cudaMemcpy(albedos, dev_albedos,
		pixelcount * sizeof(float3), cudaMemcpyDeviceToHost);
	// Retrieve image from GPU
	cudaMemcpy(normals, dev_normals,
		pixelcount * sizeof(float3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");

	if (TIMER) {
		printf("(Time Taken so Far, time for this iteration) : (%f, %f) \n", total_time, iteration_time);
	}
}
