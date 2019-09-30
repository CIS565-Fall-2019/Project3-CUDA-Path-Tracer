#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/partition.h>

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

#define TIME_ITER 1

#define STREAM_COMPACTION 1
#define SORT_BY_MATERIAL 0
#define CACHE_FIRST_BOUNCE 0

#define DIRECT_LIGHTING 0

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

#if TIME_ITER
PerformanceTimer& timer()
{
    static PerformanceTimer timer;
    return timer;
}
#endif

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
#if CACHE_FIRST_BOUNCE
static ShadeableIntersection * dev_first_bounce = NULL;
#endif
static Geom * dev_lights = NULL;
static int num_lights = 0;
static int num_materials = 0;

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
    num_materials = scene->materials.size();

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    #if CACHE_FIRST_BOUNCE
    cudaMalloc(&dev_first_bounce, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_first_bounce, 0, pixelcount * sizeof(ShadeableIntersection));
    #endif

    #if DIRECT_LIGHTING
    std::vector<Geom> lights;
    for (Geom g : scene->geoms)
    {
        if (scene->materials[g.materialid].emittance > 0.0f)
        {
            lights.push_back(g);
        }
    }
    num_lights = lights.size();

    cudaMalloc(&dev_lights, num_lights * sizeof(Geom));
    cudaMemcpy(dev_lights, lights.data(), lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);
    #endif

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    #if CACHE_FIRST_BOUNCE
    cudaFree(dev_first_bounce);
    #endif

    #if DIRECT_LIGHTING
    cudaFree(dev_lights);
    #endif

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

        #if CACHE_FIRST_BOUNCE
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
        );
        #else
        // Antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, x + y * cam.resolution.x, 0);
        thrust::uniform_real_distribution<float> jitter(-0.5, 0.5);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitter(rng))
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitter(rng))
			);
        #endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

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
    , int num_materials
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
        bool changeMat = false;
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
            else if (geom.type == HOLLOW)
            {
                bool mat = true;
                t = hollowShapeIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, mat);
                if (!mat) { changeMat = true; }
            }
            else if (geom.type == TWIST)
            {
                bool mat = true;
                t = twistIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, mat);
                if (!mat) { changeMat = true; }
            }

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
            if (changeMat && (geoms[hit_geom_index].type == HOLLOW || geoms[hit_geom_index].type == TWIST))
            {
                int mID = geoms[hit_geom_index].materialid + 1;
                if (mID >= num_materials) { mID = 0; }
                intersections[path_index].materialId = mID;
            }
			intersections[path_index].surfaceNormal = normal;
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

      // If the material indicates that the object was a light, "light" the ray
      if (material.emittance > 0.0f) {
        pathSegments[idx].color *= (materialColor * material.emittance);
      }
      // Otherwise, do some pseudo-lighting computation. This is actually more
      // like what you would expect from shading in a rasterizer like OpenGL.
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

__global__ void shadeScene(
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
        if (pathSegments[idx].remainingBounces <= 0) { return; }

        if (intersection.t > 0.0f) { // if the intersection exists...
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            else {
                glm::vec3 intersecPoint = getPointOnRay(pathSegments[idx].ray, intersection.t);
                scatterRay(pathSegments[idx], intersecPoint, intersection.surfaceNormal, material, rng);
                pathSegments[idx].remainingBounces--;
            }
        }
        else {
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

__global__ void shadeSceneDirectLighting(
    int iter
    , int num_paths
    , ShadeableIntersection * shadeableIntersections
    , PathSegment * pathSegments
    , Material * materials
    , Geom * lights
    , int num_lights
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (pathSegments[idx].remainingBounces <= 0) { return; }

        if (intersection.t > 0.0f && num_lights > 0) { // if the intersection exists...
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            Geom* chosenLight = nullptr;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                float dir = glm::dot(intersection.surfaceNormal, glm::vec3(0, -1, 0));
                if (chosenLight)
                {
                    if (intersection.materialId == chosenLight->materialid && dir > 0) {
                        pathSegments[idx].color *= (materialColor * material.emittance);
                    }
                    else {
                        pathSegments[idx].color *= glm::vec3(0.0f);
                    }
                }
                else {
                    pathSegments[idx].color *= (materialColor * material.emittance);
                }
                pathSegments[idx].remainingBounces = 0;
            }
            else if (pathSegments[idx].remainingBounces == 1)
            {
                // this should be the ray to the light so if we didnt hit the light
                // it means the pixel is in shadow
                pathSegments[idx].color *= glm::vec3(0.0f);
                pathSegments[idx].remainingBounces = 0;
            }
            else {
                glm::vec3 intersecPoint = getPointOnRay(pathSegments[idx].ray, intersection.t);

                // compute f for this material
                if (material.hasReflective || material.hasRefractive){
                    // in direct lighting we dont get reflective and refractive properties
                    pathSegments[idx].color *= glm::vec3(0.0f);
                }
                else{
                    // get the color of the diffuse surface
                    pathSegments[idx].color *= materialColor;
                }

                // Sets ray direction toward some position on some light in the scene
                chosenLight = directRayToLight(pathSegments[idx], intersecPoint, lights, num_lights, rng);

                // add in absDot
                pathSegments[idx].color *= glm::abs(glm::dot(pathSegments[idx].ray.direction, intersection.surfaceNormal));

                // next ray will be to light
                pathSegments[idx].remainingBounces = 1;
            }
        }
        else {
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
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

struct sortByMat
{
    __host__ __device__
    bool operator()(const ShadeableIntersection &i1, 
            const ShadeableIntersection &i2)
    {
        return i1.materialId < i2.materialId;
    }
};

struct isTerm
{
    __host__ __device__
    bool operator()(const PathSegment &ps)
    {
        return ps.remainingBounces > 0;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
float pathtrace(uchar4 *pbo, int frame, int iter) {
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

    #if SORT_BY_MATERIAL
    thrust::device_ptr<PathSegment> dev_thrust_paths(dev_paths);
    thrust::device_ptr<ShadeableIntersection> dev_thrust_intersections(dev_intersections);
    #endif

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
    //   * Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    #if TIME_ITER
    timer().startCpuTimer();
    #endif

	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

  bool iterationComplete = false;
	while (!iterationComplete) {

	    // clean shading chunks
	    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        #if CACHE_FIRST_BOUNCE
        if (iter == 1 || depth > 0)
        {
            computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
                depth
                , num_paths
                , dev_paths
                , dev_geoms
                , hst_scene->geoms.size()
                , dev_intersections,
                num_materials);
            checkCUDAError("trace one bounce");
        }
        else
        {
            cudaMemcpy(dev_intersections, dev_first_bounce, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        if (iter == 1 && depth == 0)
        {
            cudaMemcpy(dev_first_bounce, dev_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
        }
        #else
	    computeIntersections <<<numblocksPathSegmentTracing, blockSize1d>>> (
		    depth
		    , num_paths
		    , dev_paths
		    , dev_geoms
		    , hst_scene->geoms.size()
		    , dev_intersections
            , num_materials);
	    checkCUDAError("trace one bounce");
        #endif

        cudaDeviceSynchronize();
        depth++;

        #if SORT_BY_MATERIAL 
        // Sort paths and intersections based on their material id to make 
        // computations with similar conditional branching contiguous in memory
        thrust::sort_by_key(dev_thrust_intersections, 
            dev_thrust_intersections + num_paths, 
            dev_thrust_paths,
            sortByMat());
        #endif

        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        #if DIRECT_LIGHTING 
        shadeSceneDirectLighting << <numblocksPathSegmentTracing, blockSize1d >> > (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials,
            dev_lights,
            num_lights);
        checkCUDAError("shadeScene");
        #else
        shadeScene<<<numblocksPathSegmentTracing, blockSize1d>>> (
            iter,
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials);
        checkCUDAError("shadeScene");
        #endif

        #if STREAM_COMPACTION 
        // Stream compaction
        PathSegment* new_end = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, isTerm());
        num_paths = new_end - dev_paths;

        iterationComplete = (depth == traceDepth || num_paths == 0);
        #else   
        iterationComplete = (depth == traceDepth);
        #endif
	}

    #if TIME_ITER
        timer().endCpuTimer();
    #endif

    num_paths = dev_path_end - dev_paths;

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
    #if TIME_ITER
        return timer().getCpuElapsedTimeForPreviousOperation();
    #else
        return -1.f
    #endif
}
