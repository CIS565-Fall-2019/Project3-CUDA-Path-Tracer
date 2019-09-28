#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <texture_indirect_functions.h>

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
    fprintf(stderr, ": %d: %s: %s\n", line, msg, cudaGetErrorString(err));
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

static cudaTextureObject_t texobj0, texobj1, texobj2, texobj3, texobj4, texobj5, texobj6, texobj7;

typedef struct textureWrapper {
	cudaTextureObject_t contents[8];
} TextureWrapper;

static TextureWrapper texObjects = { texobj0, texobj1, texobj2, texobj3,
									 texobj4, texobj5, texobj6, texobj7 };


static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Triangle* dev_tris = NULL;
static Material * dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static ShadeableIntersection* dev_intersections_first = NULL;
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

	cudaMalloc(&dev_tris, scene->triangles.size() * sizeof(Triangle));
	cudaMemcpy(dev_tris, scene->triangles.data(), scene->triangles.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
  	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

	cudaMalloc(&dev_intersections_first, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_intersections_first, 0, pixelcount * sizeof(ShadeableIntersection));

	for (int i = 0; i < scene->textures.size(); i++) {
		scene->textures[i].putOntoDevice(i);
	}//for
    // TODO: initialize any extra device memeory you need

    checkCUDAError("pathtraceInit");

}

void pathtraceFree(Scene* scene) {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
	cudaFree(dev_tris);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);
	cudaFree(dev_intersections_first);

	for (int i = 0; i < scene->textures.size(); i++) {
		scene->textures[i].freeFromDevice(i);
	}//for

	//TODO: free textures off device?

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
		segment.curIOR = 1.0;
		

		float xfloat = (float)x;
		float yfloat = (float)y;

#if ANTIALIASING
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
		thrust::uniform_real_distribution<float> u01(-0.5, 0.5);
		float xFac = u01(rng);
		float yFac = u01(rng);
		xfloat += xFac;
		yfloat += yFac;
#endif

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * (xfloat - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * (yfloat - (float)cam.resolution.y * 0.5f)
			);


		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
		int depth,
		int num_paths,
		PathSegment * pathSegments,
		Geom * geoms,
		Triangle* tris,
		int geoms_size,
		ShadeableIntersection * intersections){

	int path_index = blockIdx.x * blockDim.x + threadIdx.x;
	if (path_index >= num_paths) return;

	PathSegment& pathSegment = pathSegments[path_index];

	float t;
	gvec3 intersect_point;
	gvec3 normal;
	float t_min = FLT_MAX;
	int hit_geom_index = -1;
	int hit_tri_index = -1;
	bool outside = true;

	bool tmp_outside = true;
	gvec3 tmp_intersect;
	gvec3 tmp_normal;
	int tmp_tri_index;
	float2 tmp_uv = { -1.0, -1.0 };
	float2 min_uv = { -1.0, -1.0 };

	// naive parse through global geoms

	for (int i = 0; i < geoms_size; i++) {
		Geom& geom = geoms[i];
		tmp_uv = { -1.0, -1.0 };

		if (geom.type == CUBE) {
			t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside, &tmp_uv);
		}
		else if (geom.type == SPHERE) {
			t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside, &tmp_uv);
		}
		else if (geom.type == MESH) {
			t = meshIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, tmp_outside, tris, &tmp_tri_index, &tmp_uv);
		}
		/*
		else if (geom.type == TRIANGLE) {
			t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal);
		}
		*/

		// Compute the minimum t from the intersection tests to determine what
		// scene geometry object was hit first.
		if (t > 0.0f && t_min > t) {
			t_min = t;
			hit_geom_index = i;
			intersect_point = tmp_intersect;
			normal = tmp_normal;
			outside = tmp_outside;
			min_uv = tmp_uv;
			if (geom.type == MESH) {
				hit_tri_index = tmp_tri_index;
			}
			else {
				hit_tri_index = -1;
			}
		}
	}//for each geom

	if (hit_geom_index == -1) {
		intersections[path_index].t = -1.0f;
	}
	else {
		//The ray hits something
		intersections[path_index].t = t_min;
		intersections[path_index].surfaceNormal = normal;
		intersections[path_index].leaving = !outside;
		intersections[path_index].uv = min_uv;
		if (hit_tri_index > -1) {
			int myMaterial = tris[hit_tri_index].materialid;
			intersections[path_index].materialId = myMaterial;
		}//if
		else {
			intersections[path_index].materialId = geoms[hit_geom_index].materialid;
		}//else
	}
}

__global__ void shadeRealMaterial(
	int iter,
	int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials, TextureWrapper twrap) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	cudaTextureObject_t textureReference;

	if (idx >= num_paths) return;

	ShadeableIntersection intersection = shadeableIntersections[idx];
	PathSegment* incoming = &pathSegments[idx];
	if (intersection.t > 0.0f) { // if the intersection exists...
	  // Set up the RNG

		thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
		thrust::uniform_real_distribution<float> u01(0, 1);

		Material material = materials[intersection.materialId];
		gvec3 materialColor = material.color;

		if (material.textureId >= 0) {
			textureReference = twrap.contents[material.textureId];
		}

		// If the material indicates that the object was a light, "light" the ray

		if (material.emittance > 0.0f) {

#if TEX_EMISSIVE
			if (material.textureMask & TEXTURE_EMISSIVE) {
				float4 emissiveText = tex2DLayered<float4>(textureReference, intersection.uv.x, intersection.uv.y, TEXTURE_LAYER_EMISSIVE);
				gvec3 emissiveColor = gvec3(emissiveText.x, emissiveText.y, emissiveText.z);
				if (glm::length(emissiveColor) > 0.04) {
					incoming->color *= emissiveColor * material.emittance;
					incoming->remainingBounces = 0;//stop bouncing here!
					return;
				}//if we're emitting light
			}//checking for emissive
			else {
#endif
				incoming->color *= (materialColor * material.emittance);
				incoming->remainingBounces = 0;//stop bouncing here!
				return;
#if TEX_EMISSIVE
			}
#endif
		}

		incoming->remainingBounces--;

		scatterRay(*incoming, 
					getPointOnRayEp(incoming->ray, intersection.t), 
					intersection.surfaceNormal, 
					material, 
					intersection.leaving, 
					intersection.uv, 
					textureReference, rng);


	}//if we have an intersection
	else {
		incoming->color = gvec3(0.0f);
		incoming->remainingBounces = 0;
	}//no hit
}//shadeRealMaterial


// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths){
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += iterationPath.color;
	}
}

/*thrust construct for telling whether we're out of bounces*/
struct hasRemainingBounces {
	__host__ __device__ bool operator()(const PathSegment x) {
		return x.remainingBounces != 0;
	}
};

/*thrust construct for sorting by material id*/
struct materialIdLess {
	__host__ __device__ bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
		if (a.t < 0 && b.t > 0) return true;//might be extraneous
		if (b.t < 0 && a.t > 0) return false;//might be extraneous
		return(a.materialId < b.materialId);
	}
};

/**
 Wrapper for the __global__ call that sets up the kernel calls and does a ton
 of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;


	//makes our initial path segments in dev_paths; contains the ray, a color, a pixelIndex, and bounce count for each
	generateRayFromCamera <<<blocksPerGrid2d, blockSize2d >>> (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = dev_path_end - dev_paths;//ok, clever use of pointer math...

	int total_paths = num_paths;
	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if CACHING_FIRST && !ANTIALIASING
		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		if (depth == 0 && iter == 1) {
			computeIntersections <<<numblocksPathSegmentTracing, blockSize1d >>> (
				depth,
				num_paths,
				dev_paths,
				dev_geoms,
				dev_tris,
				hst_scene->geoms.size(),
				dev_intersections_first);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
			cudaMemcpy(dev_intersections, dev_intersections_first, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}//if the first set of intersections
		else if (depth == 0) {
			cudaMemcpy(dev_intersections, dev_intersections_first, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
		}//use the cached intersections
		else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth,
				num_paths,
				dev_paths,
				dev_geoms,
				dev_tris,
				hst_scene->geoms.size(),
				dev_intersections);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
#else
		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth,
			num_paths,
			dev_paths,
			dev_geoms,
			dev_tris,
			hst_scene->geoms.size(),
			dev_intersections);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
#endif

		depth++;

#if SORTING_MATERIAL
		thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, materialIdLess());
#endif

		shadeRealMaterial <<<numblocksPathSegmentTracing, blockSize1d >>> (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials,
			texObjects);
		checkCUDAError("shadeRealMaterial");
		cudaDeviceSynchronize();

		PathSegment* newEnd = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, hasRemainingBounces());
		num_paths = newEnd - dev_paths;


		if (num_paths == 0) iterationComplete = true;
		//iterationComplete = true; // TODO: should be based off stream compaction results.
	}//while !iterationComplete

	// Assemble this iteration and apply it to the image
	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather <<<numBlocksPixels, blockSize1d >>> (total_paths, dev_image, dev_paths);
	checkCUDAError("finalGather");

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO <<<blocksPerGrid2d, blockSize2d >>> (pbo, cam.resolution, iter, dev_image);

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}
cudaArray* Texture::putOntoDevice(int textureIndex) {
	cudaChannelFormatDesc f4 = cudaCreateChannelDesc<float4>();
	cudaExtent extents = make_cudaExtent(width, height, 4);
	cudaMalloc3DArray(&cu_3darray, &f4, extents, cudaArrayLayered);
	cudaError_t err = cudaGetLastError();


	float4* h_data = (float4*)malloc(width * height * 4 * sizeof(float4));
	fillIntoF4Array(h_data);

	cudaMemcpy3DParms myparms = { 0 };
	myparms.srcPos = make_cudaPos(0, 0, 0);
	myparms.dstPos = make_cudaPos(0, 0, 0);
	myparms.srcPtr = make_cudaPitchedPtr(h_data, width * sizeof(float4), width, height);
	myparms.dstArray = cu_3darray;
	myparms.extent = extents;
	myparms.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&myparms);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error on the 3d memcpy! Err %d\n", err);
		exit(-1);
	}
	//check cuda error

	cudaResourceDesc    texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = cu_3darray;
	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = true;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.addressMode[2] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&(texObjects.contents[textureIndex]), &texRes, &texDescr, NULL);
	//cudaCreateTextureObject(&texobj0, &texRes, &texDescr, NULL);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error on the creating texture objects! Err %d\n", err);
		exit(-1);
	}

	free(h_data);//no need to keep it locally anymore

	return cu_3darray;
}//putOntoDevice

void Texture::freeFromDevice(int textureIndex) {
	cudaDestroyTextureObject(texObjects.contents[textureIndex]);
	cudaFreeArray(cu_3darray);
}//freeFromDevice