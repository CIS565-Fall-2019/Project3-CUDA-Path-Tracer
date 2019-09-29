#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/partition.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/rotate_vector.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1

#define CACHE_FIRST_BOUNCE 0
#define SORT_BY_MATERIAL 1
#define ANTI_ALIASING 1

// only have one of these at a time
#define BOKEH_CIRCLE 0
#define BOKEH_SQUARE 0
#define BOKEH_DIAMOND 1
#define BOKEH_TRIANGLE 0

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

static Scene *hst_scene = NULL;
static glm::vec3 *dev_image = NULL;
static Geom *dev_geoms = NULL;
static Triangle *dev_triangles = NULL;
static Material *dev_materials = NULL;
static PathSegment *dev_paths = NULL;
static ShadeableIntersection *dev_intersections = NULL;
static ShadeableIntersection *dev_cached_intersections = NULL;

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

  	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
  	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_triangles, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_triangles, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_cached_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cached_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_triangles);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
  	cudaFree(dev_cached_intersections);

    checkCUDAError("pathtraceFree");
}

__host__ __device__ bool fequals(float a, float b) {
	float diff = a - b;
	return (diff < EPSILON) && (-diff < EPSILON);
}

__host__ __device__ glm::vec3 squareToDiskConcentric(glm::vec2 sample) {
	float a = 2.f * sample.x - 1.f;
	float b = 2.f * sample.y - 1.f;
	float r, phi;

	if (a > -b) {
		if (a > b) {
			r = a;
			phi = (PI / 4.f) * (b / a);
		}
		else {
			r = b;
			phi = (PI / 4.f) * (2 - (a / b));
		}
	}
	else {
		if (a < b) {
			r = -a;
			phi = (PI / 4.f) * (4.f + (b / a));
		}
		else {
			r = -b;
			if (!fequals(b, 0.f)) {
				phi = (PI / 4.f) * (6.f - (a / b));
			}
			else {
				phi = 0.f;
			}
		}
	}

	float x = r * cos(phi);
	float y = r * sin(phi);
	return glm::vec3(x, y, 0.f);
}

// generate uniform random point in triangle
// https://www.cs.princeton.edu/~funk/tog02.pdf
__host__ __device__ glm::vec3 pointInTriangle(glm::vec2 a, glm::vec2 b, glm::vec2 c, glm::vec2 rand) {
	return glm::vec3((1.f - sqrt(rand.x)) * a + sqrt(rand.x) * (1.f - rand.y) * b + rand.y * sqrt(rand.x) * c, 0.f);
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
		
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
		thrust::uniform_real_distribution<float> u01(0, 1);

		float jitterX = 0.f;
		float jitterY = 0.f;
#if ANTI_ALIASING
		// implement antialiasing by jittering the ray
		jitterX = u01(rng);
		jitterY = u01(rng);
#endif // #if ANTI_ALIASING

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitterX)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitterY)
		);

		// depth of field via thin lens approximation
		// sample point on lens
		glm::vec2 xi = glm::vec2(u01(rng), u01(rng));
		glm::vec3 lensPoint(0.f);

#if BOKEH_CIRCLE
		lensPoint = cam.lensRadius * squareToDiskConcentric(xi);

#elif BOKEH_SQUARE
		lensPoint = cam.lensRadius * glm::vec3(xi, 0.f);

#elif BOKEH_DIAMOND
		// same as square but rotated 45 deg
		glm::vec2 p = glm::rotate(xi, 45.f);
		lensPoint = cam.lensRadius * glm::vec3(p, 0.f);

#elif BOKEH_TRIANGLE
		// pick 3 points of triangle
		//glm::vec2 a = glm::vec2(1.f, 0.f);
		glm::vec2 a = glm::vec2(0.5, 0.f);
		//glm::vec2 b = glm::vec2(-0.5f, 0.866f);
		glm::vec2 b = glm::vec2(-0.5f, 0.f);
		glm::vec2 c = glm::vec2(0.f, 0.866f);

		lensPoint = pointInTriangle(a, b, c, xi) * cam.lensRadius;
#endif
		// compute point on plane of focus
		glm::vec3 focusPoint = segment.ray.origin + segment.ray.direction * (cam.focalDistance / glm::abs(segment.ray.direction.z));

		// update ray
		segment.ray.origin += lensPoint.x * cam.right + lensPoint.y * cam.up;
		segment.ray.direction = glm::normalize(focusPoint - segment.ray.origin);

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(int depth, int num_paths, PathSegment *pathSegments, 
	Geom *geoms, int geoms_size, Triangle *triangles, ShadeableIntersection *intersections) {
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
			else if (geom.type == MESH) {
				t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside, triangles);
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
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].pos = intersect_point;
		}
	}
}

__global__ void shadeMaterial(int iter, int num_paths, ShadeableIntersection *shadeableIntersections,
	PathSegment *pathSegments, Material *materials) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		PathSegment& pathSegment = pathSegments[idx];

		if (intersection.t > 0.0f) { // if the intersection exists...
			// set up the RNG
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

			Material material = materials[intersection.materialId];

			// If the material indicates that the object was a light, "light" the ray
			if (material.emittance > 0.0f) {
				pathSegment.color *= (material.color * material.emittance);
				pathSegment.remainingBounces = 0; // np bounces after hitting light
			}
			else {
				// calculate accumulated color and new bounced ray
				scatterRay(pathSegment, intersection.pos, intersection.surfaceNormal, material, rng);
				pathSegment.remainingBounces--;
			}
		}
		else {
			// If there was no intersection, color the ray black.
			// Lots of renderers use 4 channel color, RGBA, where A = alpha, often
			// used for opacity, in which case they can indicate "no opacity".
			// This can be useful for post-processing and image compositing.
			pathSegment.color = glm::vec3(0.0f);
			pathSegment.remainingBounces = 0;
		}
	}
}


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 *image, PathSegment *iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

struct isNotTerminated {
	__host__ __device__ bool operator()(const PathSegment &p) {
		return (p.remainingBounces > 0);
	}
};

struct sortMaterialId {
	__host__ __device__ bool operator()(const ShadeableIntersection &i1, const ShadeableIntersection &i2) {
		return (i1.materialId > i2.materialId);
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

	generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;
	int remaining_paths = num_paths;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (remaining_paths + blockSize1d - 1) / blockSize1d;
		
#if CACHE_FIRST_BOUNCE
		if (depth == 0) {
			if (iter == 1) {
				// if first bounce of first iteration, cache intersections
				computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(depth, remaining_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_triangles, dev_intersections);
				cudaMemcpy(dev_cached_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else {
				// if first bounce of later iterations, pull from cache
				cudaMemcpy(dev_intersections, dev_cached_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		}
		else {
#endif // #if CACHE_FIRST_BOUNCE 
			computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>>(depth, remaining_paths, dev_paths, dev_geoms, hst_scene->geoms.size(), dev_triangles, dev_intersections);
#if CACHE_FIRST_BOUNCE
		}
#endif // #if CACHE_FIRST_BOUNCE 

		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
		depth++;

#if SORT_BY_MATERIAL
		// before we shade, sort path segments and intersections by material id
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + remaining_paths, dev_paths, sortMaterialId());
#endif // #if SORT_BY_MATERIAL

		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by evaluating the BSDF.
		shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(iter, remaining_paths, dev_intersections, dev_paths, dev_materials);

		iterationComplete = (depth > traceDepth);

		// skip stream compaction if we can
		if (!iterationComplete) {
			// run stream compaction to remove terminated rays
			dev_path_end = thrust::partition(thrust::device, dev_paths, dev_paths + remaining_paths, isNotTerminated());
			remaining_paths = dev_path_end - dev_paths;

			// if all rays are terminated, iteration is complete
			iterationComplete = (remaining_paths <= 0);
		}
	}

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
