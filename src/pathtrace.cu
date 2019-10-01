#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/count.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <glm/gtc/matrix_inverse.hpp>
#include <chrono>
#include <ctime>
#include <ratio>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "efficient.h"

#define RECORD_TIMING 0
#define ERRORCHECK 1
#define CACHEFIRSTBOUNCE 0
#define RAYSORT 0
#define MOTION_BLUR 0
#define STREAMCOMPACT_BY_THRUST 0
#define STREAM_COMPACT 1
#define ANTI_ALIASING 1

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
static ShadeableIntersection * dev_first_intersections = NULL;
int *dev_remaining_paths = NULL;

cudaEvent_t start, stop;
// TODO: static variables for device memory, any extra info you need, etc
// ...

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

	cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_remaining_paths, pixelcount * sizeof(int));
	cudaMemset(dev_remaining_paths, 0, pixelcount * sizeof(int));

    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
	cudaFree(dev_remaining_paths);
    // TODO: clean up any extra device memory you created

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
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments,int *dev_remaining_paths)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
		
		#if ANTI_ALIASING && !CACHEFIRSTBOUNCE
			thrust::default_random_engine rng1 = makeSeededRandomEngine(iter , x , y);
			thrust::default_random_engine rng2 = makeSeededRandomEngine(iter, y, x);
			thrust::uniform_real_distribution<float> u01(-0.5, 0.5);

			x += u01(rng1);
			y += u01(rng2);
		#endif

		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
		dev_remaining_paths[index] = index;
		//segment.notDead = 1;
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
	,int *dev_remaining_paths
	)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_paths)
	{
		int path_index = dev_remaining_paths[idx];
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
				// Motion Blur Code <<Not working>>
			//for (int i = 0; i < hst_scene->geoms.size(); i++) {
			//	float t = 0.01f;
			//	Geom & motion_geom = hst_scene->geoms[i];
			//	if (motion_geom.hasMotion) {
			//		motion_geom.translation += mot * t;
			//		motion_geom.transform = utilityCore::buildTransformationMatrix(motion_geom.translation, motion_geom.rotation, motion_geom.scale);
			//		motion_geom.inverseTransform = glm::inverse(motion_geom.transform);
			//		motion_geom.invTranspose = glm::inverseTranspose(motion_geom.transform);
			//		printf("Hello motion value for object ID: %d is: %0.02f, %0.02f , %0.02f\n", motion_geom.materialid, motion_geom.translation.x, motion_geom.translation.y, motion_geom.translation.z);
					//printf("Hello motion value for object ID: %d is: %0.02f, %0.02f , %0.02f\n", motion_geom.translation.x, motion_geom.translation.y, motion_geom.translation.z);
					//printf("Hello motion value for object ID: %d is: %0.02f, %0.02f , %0.02f\n", motion_geom.translation.x, motion_geom.translation.y, motion_geom.translation.z);
				
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				
				//cudaMemcpy(dev_geoms, &(hst_scene->geoms)[0], hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);


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
			intersections[path_index].intersectionPoint = tmp_intersect;
		}
		//pathSegment.remainingBounces--;


	}
}

__global__ void shaderKernel(int iter,int numPaths,int depth, ShadeableIntersection* shadeableIntersections, Material* materials, PathSegment* pathsegments,int *dev_remaining_paths) {
	int idxx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idxx >= numPaths || pathsegments[dev_remaining_paths[idxx]].remainingBounces < 0)
		return;
	
	int idx = dev_remaining_paths[idxx];
	ShadeableIntersection &intersection = shadeableIntersections[idx];
	Material &material = materials[intersection.materialId];
	PathSegment &pathsegment = pathsegments[idx];

	if (intersection.t >= 0.0f) {
		if (material.emittance > 0.0f) {
			pathsegment.color *= material.color * material.emittance;
			//printf("Hello \n");
			pathsegment.remainingBounces =0;

		}
		else {
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, idxx, depth);
				//if (intersection.materialId == 4)
					scatterRay(pathsegment, intersection.intersectionPoint, intersection.surfaceNormal, material, rng);
				//else
				//	scatterRay(pathsegment, intersection.intersectionPoint, intersection.surfaceNormal, material, rng);
				pathsegment.remainingBounces--;
		}	
	}
	else {
		pathsegment.color = glm::vec3(0.0f);
		pathsegment.remainingBounces = 0;
		
	}

	#if STREAM_COMPACT
		if (pathsegment.remainingBounces<=0)
			dev_remaining_paths[idxx] = -1;
	#endif	
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


struct pathsDead
{
	__host__ __device__
		bool operator()(const PathSegment &segment)
	{
		return (segment.remainingBounces > 0);
	}
};

struct cmp
{
	__host__ __device__  bool operator()(const ShadeableIntersection& intersect1, const ShadeableIntersection& intersect2) const
	{
		return (intersect1.materialId < intersect2.materialId);
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
	//int traceDepth = 4;
	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths, dev_remaining_paths);
	checkCUDAError("generate camera ray");

	//printf("Hello World\n");

	PathSegment* dev_path_end = dev_paths + pixelcount;
	int numPaths = dev_path_end - dev_paths;
	bool cacheFirstBounce = false;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;
	int depth = 0;

	# if CACHEFIRSTBOUNCE
		if (iter == 1) {
			cacheFirstBounce = true;
			printf("Hello");
		}	
	#endif


	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (numPaths + blockSize1d - 1) / blockSize1d;

		#if RECORD_TIMING
			using namespace std::chrono;
			high_resolution_clock::time_point t1 = high_resolution_clock::now();
		#endif

		if (cacheFirstBounce) {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, numPaths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_first_intersections
				, dev_remaining_paths
				);
			checkCUDAError("First trace bounce failed");
			cudaDeviceSynchronize();
			cudaMemcpy(dev_intersections, dev_first_intersections, numPaths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			cacheFirstBounce = false;
		}
		else if (depth == 0 && CACHEFIRSTBOUNCE) {
			cudaMemcpy(dev_intersections, dev_first_intersections, numPaths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, numPaths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections,
				dev_remaining_paths
				);
			checkCUDAError("Computer Intersections failed");
			cudaDeviceSynchronize();
		}

		depth++;
		
		shaderKernel << <numblocksPathSegmentTracing, blockSize1d >> > (iter, numPaths, depth, dev_intersections, dev_materials, dev_paths, dev_remaining_paths);
		checkCUDAError("Gathering the final Image failed");
		cudaDeviceSynchronize();

		#if STREAMCOMPACT_BY_THRUST
			dev_path_end = thrust::partition(thrust::device,dev_paths, dev_paths + numPaths, pathsDead());
			numPaths = dev_path_end - dev_paths;
		#endif		

		#if STREAM_COMPACT
			numPaths = StreamCompaction::Efficient::compact(numPaths,dev_remaining_paths);
		#endif		

		
		#if RAYSORT
			thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + numPaths, dev_paths, cmp());
		#endif

		if (depth >= traceDepth || numPaths<=0)
			iterationComplete = true; // TODO: should be based off stream compaction results.

		#if RECORD_TIMING
			high_resolution_clock::time_point t2 = high_resolution_clock::now();
			duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
			if (iter < 5)
				std::cout << "For iter "<<iter<<", and depth "<<depth<<", it took" << time_span.count() << " seconds. The number of live paths are "<<numPaths<<endl;
		#endif
		
		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		/*
		shadeFakeMaterial << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
			);
		*/

		// Assemble this iteration and apply it to the image
	}
    ///////////////////////////////////////////////////////////////////////////
	numPaths = pixelcount;
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather << <numBlocksPixels, blockSize1d >> > (numPaths, dev_image, dev_paths);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
