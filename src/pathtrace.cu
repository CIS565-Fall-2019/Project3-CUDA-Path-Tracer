#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/partition.h> // for partition
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"
#include "tiny_gltf.h"
#include <glm/gtc/matrix_inverse.hpp>

// for performance analysis
#include <cuda_runtime.h>

//
//#include <nvToolsExt.h>


#define ERRORCHECK 1

// toggleable part 1 macros
//#define CACHE_ME_OUTSIDE
//#define STREAM_COMPACTION
//#define MATERIAL_SORT
//#define ANTIALIASING
//#define DEPTH_OF_FIELD
#define LENS_RADIUS 0.4f
#define FOCAL_DISTANCE 7.0f
//#define MOTION_BLUR

#ifdef CACHE_ME_OUTSIDE 
#ifdef ANTIALIASING
	static_assert(0,"Anti aliasing and caching can not be combined" );
#endif
#endif

#ifdef CACHE_ME_OUTSIDE 
#ifdef MOTION_BLUR
	static_assert(0, "motion blur and caching can not be combined");
#endif
#endif

#ifdef CACHE_ME_OUTSIDE 
#ifdef DEPTH_OF_FIELD
	static_assert(0, "depth of field and caching can not be combined");
#endif
#endif

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;\
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

// from performance lab 
#define CUDA(call) do {                                 \
    cudaError_t e = (call);                             \
    if (e == cudaSuccess) break;                        \
    fprintf(stderr, __FILE__":%d: %s (%d)\n",           \
            __LINE__, cudaGetErrorString(e), e);        \
    exit(1);                                            \
} while (0)

// also taken from performance lab
void printResults(double timeInMilliseconds, int iterations)
{
	// print out the time required for the kernel to finish the transpose operation
	double bandwidth = (iterations * 1e-9) / (timeInMilliseconds * 1e-3);
	std::cout << "Elapsed Time for " << iterations << " runs = " << round(timeInMilliseconds) << "ms" << std::endl;
}


__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

void memory_debug(int elements, PathSegment* cuda_mem, PathSegment* cpu_mem)
{
	cudaMemcpy(cpu_mem, cuda_mem, elements * sizeof(PathSegment), cudaMemcpyDeviceToHost);
	checkCUDAError("copy out failed!");
	//printf("=============================\n");
	//for (int i = 0; i < elements; i++)
	//{
	//	printf("out[%d] %d\n ", i, cpu_mem[i].remainingBounces);
	//}
	printf("=============================\n");
}

void material_memory_debug(int elements, Material* cuda_mem, Material* cpu_mem)
{
	cudaMemcpy(cpu_mem, cuda_mem, elements * sizeof(PathSegment), cudaMemcpyDeviceToHost);
	checkCUDAError("copy out failed!");
	printf("=============================\n");
	for (int i = 0; i < elements; i++)
	{
		printf("out[%d] %f %f\n ", i, cpu_mem[i].hasReflective, cpu_mem[i].hasRefractive);
	}
	printf("=============================\n");
}

__global__ void sort_material(int active_paths, ShadeableIntersection* intersections, int* path, int* intersect)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < active_paths)
	{
		// we want to group the materials 
		int id = intersections[path_index].materialId;
		path[path_index] = id;
		intersect[path_index] = id;
	}
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

#ifdef CACHE_ME_OUTSIDE
static int* dev_intersectionsCached = NULL;
#endif

#ifdef MATERIAL_SORT
// we want to sort our dev_path and dev_intersections since that what is being
// used in the fake shader and naive sorter
static int* dev_sort_material_intersect = NULL;
static int* dev_sort_material_path = NULL;
#endif
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

    // TODO: initialize any extra device memeory you need
#ifdef CACHE_ME_OUTSIDE
	cudaMalloc(&dev_intersectionsCached, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersectionsCached, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

#ifdef MATERIAL_SORT
	cudaMalloc(&dev_sort_material_intersect, pixelcount * sizeof(int));
	cudaMemset(dev_sort_material_intersect, 0, pixelcount * sizeof(int));
	cudaMalloc(&dev_sort_material_path, pixelcount * sizeof(int));
	cudaMemset(dev_sort_material_path, 0, pixelcount * sizeof(int));
#endif
    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);

#ifdef CACHE_ME_OUTSIDE
	cudaFree(dev_intersectionsCached);
#endif

#ifdef MATERIAL_SORT
	cudaFree(dev_sort_material_intersect);
	cudaFree(dev_sort_material_path);
#endif

    checkCUDAError("pathtraceFree");
}

// this function was also taken/reworked from PBRT pg 667 section 13 
__device__ void ConcentricSampleDisk(float rand1, float rand2, float* dx, float* dy)
{
	float r, theta;
	// map uniform random numbers to -1,1
	float sx = (2 * rand1) - 1;
	float sy = (2 * rand2) - 1;

	//map square to r, theta
	if (sx == 0 && sy == 0)
	{
		// handle degeneracy at the origin
		*dx = 0;
		*dy = 0;
		return;
	}

	if (fabsf(sx) > fabsf(sy))
	{
		r = sx;
		theta = (PI / 4) * (sx / sy);
	}
	else
	{
		r = sy;
		theta = ((PI / 2) - (PI / 4) * (sx / sy));
	}

	// assign
	*dx = r * cosf(theta);
	*dy = r * sinf(theta);
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
	float alias_x = x;
	float alias_y = y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		// whats an appropriate range?-1,1? or more? 
		thrust::default_random_engine rng = makeSeededRandomEngine(x, y, iter);
		thrust::uniform_real_distribution<float> u01(-1, 1);
		thrust::uniform_real_distribution<float> u02(0, 1);

#ifdef ANTIALIASING
		float x_alias = u01(rng);
		float y_alias = u01(rng);
		alias_x += x_alias;
		alias_y += y_alias;
#endif



		// TODO: implement antialiasing by jittering the ray
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * (alias_x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (alias_y - (float)cam.resolution.y * 0.5f)
			);

#ifdef DEPTH_OF_FIELD
		// algorithm from PBRT 6.2.3
		// algorthm is as follows
			//sample point on lens
			//compute point on plane of focus
			// update ray for effect of lens

		// sample point on lens
		float lensU, lensV;
		ConcentricSampleDisk(u02(rng), u02(rng),&lensU, &lensV);
		lensU *= LENS_RADIUS;
		lensV *= LENS_RADIUS;


		// compute point on plane of focus
		float ft = glm::abs(FOCAL_DISTANCE / segment.ray.direction.z);
		glm::vec3 Pfocus = (segment.ray.direction * ft);

		// update ray for lens effect
		segment.ray.origin += glm::vec3(lensU, lensV, 0.f);
		segment.ray.direction = glm::normalize(Pfocus - glm::vec3(lensU, lensV, 0.f));

#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

void add_blur(Geom * geoms, int geoms_size, int iter)
{
	glm::vec3 zero_vec(0.f);
	for (int i = 0; i < geoms_size; i++)
	{
		Geom & geom = geoms[i];
		if (geom.speed != zero_vec)
		{
			// add randomness so object does not run away out of the fr ame... hopefully
			thrust::default_random_engine rng = makeSeededRandomEngine(i, iter, 5);
			thrust::uniform_real_distribution<float> u01(-1, 1);

			geom.translation = geom.translation + (geom.speed * .05f * u01(rng));

			geom.transform = utilityCore::buildTransformationMatrix(geom.translation, geom.rotation, geom.scale);

			geom.inverseTransform = glm::inverse(geom.transform);
			geom.invTranspose = glm::inverseTranspose(geom.transform);
		}
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
	//printf("Hello from block %d thread %d\n", blockIdx.x, threadIdx.x);
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
		glm::dmat3x2 space;
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
			
			// go through everything and then figure out what the closest ray we hit was.
			// this is where you would change to like implementing a KD tree or what not
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				space = geom.space; // store the size of what we hit
				/*if (pathSegment.ray.scattering)
				{
					printf("compute\n");
				}*/
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
			intersections[path_index].space = space;
			// store this?
			//intersections[path_index].intersection_point = intersect_point;
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
	  if (pathSegments[idx].remainingBounces == 0)
		  return;
	  
	  //pathSegments[idx].remainingBounces=0;
	  ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) { // if the intersection exists...
      // Set up the RNG
      // LOOK: this is how you use thrust's RNG! Please look at
      // makeSeededRandomEngine as well.
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, depth);
      thrust::uniform_real_distribution<float> u01(0, 1);

      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

	  if (glm::length(pathSegments[idx].color) < EPSILON)
	  {
		  pathSegments[idx].remainingBounces = 0;
		  return;
	  }

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
		pathSegments[idx].remainingBounces = 0; // may want to terminate the ray here since it hit a light?
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
      // TODO: replace this! you should be able to start with basically a one-liner
      else{
		  scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray,intersection.t),intersection.surfaceNormal, material,intersection.t, rng,intersection.space);
		  pathSegments[idx].remainingBounces--; // decrement our bounce
      }

	  //pathSegments[idx].ray.origin += intersection.speed;
    // If there was no intersection, color the ray black.
    // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
    // used for opacity, in which case they can indicate "no opacity".
    // This can be useful for post-processing and image compositing.
    } else {
      pathSegments[idx].color = glm::vec3(0.0f);
	  pathSegments[idx].remainingBounces = 0; // we want to terminate the ray here as there is no valid intersection
    }
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths,const int depth)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += (iterationPath.color);
	}
}

struct RemainingBounces
{
	__host__ __device__
		bool operator()(const PathSegment &x)
	{
		return (x.remainingBounces > 0);
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

	// Create CUDA events for timing
	cudaEvent_t start, stop, c_start, c_stop, str_start, str_stop, m_start, m_stop;

	float app_time_ms, c_time_ms, str_time_ms, m_time_ms;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventCreate(&str_start);
	cudaEventCreate(&str_stop);
	cudaEventCreate(&m_start);
	cudaEventCreate(&m_stop);
	cudaEventCreate(&c_start);
	cudaEventCreate(&c_stop);


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

	int active_paths = num_paths;

	// start cuda event timer
	//nvtxRangeId_t rangeBenchmark = nvtxRangeStart("full loop start");

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
	bool iterationComplete = false;

	//Start Benchmark
	CUDA(cudaEventRecord(start, 0));

#ifdef MOTION_BLUR
	// create new host space
	Geom* host_geoms = new Geom[hst_scene->geoms.size()];
	assert(host_geoms != NULL);
	
	// copy geoms from device to host
	cudaMemcpy(host_geoms, dev_geoms, hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyDeviceToHost);

	// update geoms for speed
	add_blur(host_geoms, hst_scene->geoms.size(),iter );

	// copy back to device
	cudaMemcpy(dev_geoms, host_geoms, hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
	
	// delete space like a good boy
	delete[] host_geoms;

#endif

	while (!iterationComplete) {
		//printf("depth %d\n", depth);

	// start cuda event timer
	//nvtxRangeId_t loopBenchmark = nvtxRangeStart("loop start");

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	dim3 numblocksPathSegmentTracing = (active_paths + blockSize1d - 1) / blockSize1d;



// enable disable first bounce cachine. 
#ifdef CACHE_ME_OUTSIDE
		//Start Benchmark
		CUDA(cudaEventRecord(c_start, 0));
		//Checking if intersection cached results should be used
		if(iter == 1 && depth == 0)
		{
			computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (depth, active_paths, dev_paths, dev_geoms,
																				 hst_scene->geoms.size(), dev_intersections);
			checkCUDAError("compute Intersections Failed");
			cudaDeviceSynchronize();

			//Copy cached intersections
			cudaMemcpy(dev_intersectionsCached, dev_intersections, active_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}
		else if (iter > 1 && depth == 0)
		{
			//Cache the first Bounce
			cudaMemcpy(dev_intersections, dev_intersectionsCached, active_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}

		if (depth > 0)
		{
			computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (depth, active_paths, dev_paths, dev_geoms,
																				hst_scene->geoms.size(), dev_intersections);
			checkCUDAError("compute Intersections Failed");
			cudaDeviceSynchronize();
		}
		//Start Benchmark
		CUDA(cudaEventRecord(c_stop, 0));
		CUDA(cudaEventSynchronize(c_stop));
		// accumulate 
		CUDA(cudaEventElapsedTime(&c_time_ms, c_start, c_stop));
		printf("cache time %f : depth %d\n", c_time_ms,depth);
		c_time_ms = 0;

#else
	
	computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		depth
		, active_paths
		, dev_paths
		, dev_geoms
		, hst_scene->geoms.size()
		, dev_intersections
		);
	checkCUDAError("trace one bounce");
#endif

	cudaDeviceSynchronize();
	depth++;

#ifdef MATERIAL_SORT

	//Start Benchmark
	CUDA(cudaEventRecord(m_start, 0));
	//https://thrust.github.io/doc/group__sorting_gabe038d6107f7c824cf74120500ef45ea.html#gabe038d6107f7c824cf74120500ef45ea
	// similar to the boids assignment where we sorted to have more contiguous grouped memory
	// material sorting is also helpful to help avoid thread warps from diverging
	sort_material << <numblocksPathSegmentTracing, blockSize1d >> > (active_paths, dev_intersections,dev_sort_material_path , dev_sort_material_intersect);
	checkCUDAError("material sort failed");
	// dev path is our value where our newly sorteed material ID is our keys
	// we want our materials to be grouped together
	//int    keys[N] = { 1,   4,   2,   8,   5,   7 };
	//char values[N] = { 'a', 'b', 'c', 'd', 'e', 'f' };
	//thrust::sort_by_key(thrust::host, keys, keys + N, values);
	// keys is now   {  1,   2,   4,   5,   7,   8}
	// values is now {'a', 'c', 'b', 'e', 'f', 'd'}
	thrust::sort_by_key(thrust::device, dev_sort_material_path, dev_sort_material_path + active_paths, dev_paths);
	thrust::sort_by_key(thrust::device, dev_sort_material_intersect, dev_sort_material_intersect + active_paths, dev_intersections);
	checkCUDAError("key sort failed");

	//Stop Benchmark
	CUDA(cudaEventRecord(m_stop, 0));
	CUDA(cudaEventSynchronize(m_stop));

	CUDA(cudaEventElapsedTime(&m_time_ms, m_start, m_stop));
	printf("material time %f \n", m_time_ms);
	m_time_ms = 0;

#endif


	// TODO:
	// --- Shading Stage ---
	// Shade path segments based on intersections and generate new rays by
  // evaluating the BSDF.
  // Start off with just a big kernel that handles all the different
  // materials you have in the scenefile.
  // TODO: compare between directly shading the path segments and shading
  // path segments that have been reshuffled to be contiguous in memory.

  shadeFakeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
    iter,
	active_paths,
    dev_intersections,
    dev_paths,
    dev_materials,
	depth
  );
  checkCUDAError("shader failed");
  cudaDeviceSynchronize();

  //Material* path = new Material[hst_scene->materials.size()];
  //material_memory_debug(hst_scene->materials.size(), dev_materials, path);
 // memory_debug(num_paths, dev_paths, path);


#ifdef STREAM_COMPACTION
  //thrust::partition returns a pointer to the element in the array where the partition occurs
  // reference https://thrust.github.io/doc/group__partitioning_gac5cdbb402c5473ca92e95bc73ecaf13c.html#gac5cdbb402c5473ca92e95bc73ecaf13c
  // we have an array of pixel count so... dev_paths[pixel_count] we want the pointer that is the last element then we know everything is done
  // partition acts similar to stream compaction. Again we want to group rays that are done and rays that still are bouncing.
  // by grouping we can perform less work by not checking rays that are already done
  //Start Benchmark
  CUDA(cudaEventRecord(str_start, 0));

  dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, RemainingBounces());
  active_paths = dev_path_end - dev_paths;

  if (active_paths == 0 || depth >= traceDepth)// if we have reached the end of our pointers
  {
	  iterationComplete = true;
  }

  //Start Benchmark
  CUDA(cudaEventRecord(str_stop, 0));
  CUDA(cudaEventSynchronize(str_stop));
  // accumulate 
  CUDA(cudaEventElapsedTime(&str_time_ms, str_start, str_stop));
  printf("compact time %f \n", str_time_ms);
  str_time_ms = 0;

#else
if( depth >= traceDepth)
{
  iterationComplete = true;
}
#endif


	// end the loop
	//nvtxRangeEnd(loopBenchmark);


	}// while depth
	
		// record end 
	CUDA(cudaEventRecord(stop, 0));
	CUDA(cudaEventSynchronize(stop));

	// accumulate 
	CUDA(cudaEventElapsedTime(&app_time_ms, start, stop));


	//printf("time %f \n", app_time_ms);

	//printResults(time_ms, depth);

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths,traceDepth);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
