#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/partition.h>

#include <../../stream_compaction/common.h>
#include <../../stream_compaction/common.cu>
#include <../../stream_compaction/efficient.h>
#include <../../stream_compaction/efficient.cu>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#define ERRORCHECK 1
#define is_sort 1
#define CACHE_FIRST_BOUNCE 1
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

namespace Path_Tracer {
	//gpu timer
	using StreamCompaction::Common::PerformanceTimer;
	PerformanceTimer& timer() {
		static PerformanceTimer timer;
		return timer;
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
			color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
			color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
			color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

			// Each thread writes one pixel location in the texture (textel)
			pbo[index].w = 0;
			pbo[index].x = color.x;
			pbo[index].y = color.y;
			pbo[index].z = color.z;
		}
	}

	static Scene *hst_scene = NULL;//scene pointer
	static glm::vec3 *dev_image = NULL;//image evctor
	static Geom *dev_geoms = NULL;//geo vector
	static Material *dev_materials = NULL;//material vector
	static PathSegment *dev_paths = NULL;//path vector
	static ShadeableIntersection *dev_intersections = NULL;//inter vector
	// TODO: static variables for device memory, any extra info you need, etc
	// ...

	int *dev_bool = NULL;//bool array for ray stream compaction
	int *host_bool = NULL;
	int *host_aftercompact = NULL;
	int *dev_ind = NULL;//indices array for ray stream compaction
	int *host_ind = NULL;//indices array for ray stream compaction

	PathSegment *dev_remain = NULL;
	PathSegment *dev_paths_finish = NULL;

	static ShadeableIntersection *dev_firstbounce = NULL;
	static ShadeableIntersection *dev_firstpath = NULL;

	int pixel2 = 0;

	static Triangle * dev_tri = NULL;

	//direct light
#if DIRECT_LIGHT
	static Geom * dev_lights = NULL;
	int li_count = 0;
	int geo_count = 0;
#endif // DIRECT_LIGHT

	void pathtraceInit(Scene *scene) {
		hst_scene = scene;
		const Camera &cam = hst_scene->state.camera;
		//the pixel num
		const int pixelcount = cam.resolution.x * cam.resolution.y;
		//image memory
		cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
		//cudaMemset(void *devPtr, int value, size_t count);
		cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

		//all the path generate from the pixel
		//one pixel one line
		cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
		//gems.size is the size of the vector
		cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
		//give them the data
		cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

		cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
		cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

		//intersection
		cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// TODO: initialize any extra device memeory you need
		//stream compact
		pixel2 = (int)pow(2.0, ilog2ceil(pixelcount));


		cudaMalloc(&dev_ind, pixelcount * sizeof(int));
		cudaMalloc(&dev_bool, pixelcount * sizeof(int));

		host_bool = new int[pixelcount];
		memset(host_bool, -1, pixelcount);
		host_ind = new int[pixelcount];
		memset(host_ind, -1, pixelcount);
		host_aftercompact = new int[pixelcount];
		memset(host_ind, -1, pixelcount);

		//cudaMemset(dev_bool, -1, pixelcount * sizeof(int));
		checkCUDAError("memset error!");

		cudaMalloc(&dev_remain, pixelcount * sizeof(PathSegment));
		checkCUDAError("malloc path remain error");
		cudaMalloc(&dev_paths_finish, pixelcount * sizeof(PathSegment));
		checkCUDAError("malloc path finish error");

		//cache first bounce
		cudaMalloc(&dev_firstpath, pixelcount * sizeof(PathSegment));
		checkCUDAError("malloc first path error");
		cudaMemset(dev_firstpath, 0, pixelcount * sizeof(PathSegment));
		checkCUDAError("memset first path error");
		cudaMalloc(&dev_firstbounce, pixelcount * sizeof(ShadeableIntersection));
		checkCUDAError("malloc first bounce error");
		cudaMemset(dev_firstbounce, 0, pixelcount * sizeof(ShadeableIntersection));
		checkCUDAError("memset first bounce error");
		checkCUDAError("pathtraceInit");

#if DIRECT_LIGHT
		geo_count = scene->geoms.size();
		li_count = scene->lights.size();
		cudaMalloc(&dev_lights, scene->lights.size() * sizeof(Geom));
		cudaMemcpy(dev_lights, scene->lights.data(), scene->lights.size() * sizeof(Geom), cudaMemcpyHostToDevice);
		checkCUDAError("malloc dev lights error");
#endif 

#if !TRI_2D_Array 
		//Triangle
		cudaMalloc(&dev_tri, scene->tris.size() * sizeof(Triangle));
		checkCUDAError("malloc dev tri error");
		cudaMemcpy(dev_tri, scene->tris.data(), scene->tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
		checkCUDAError("memset dev tri error");
#else
		int count_2d = 0;
		for (int i = 0; i < scene->mesh_tris.size(); i++) {
			count_2d += scene->mesh_tris[i].size() * sizeof(Triangle);
		}
		cudaMalloc(&dev_tri, count_2d);
		checkCUDAError("malloc dev tri 2d error");
		//need to check
		cudaMemcpy(dev_tri, scene->tris.data(), count_2d, cudaMemcpyHostToDevice);
		checkCUDAError("memset dev tri 2d error");
#endif

	}

	void pathtraceFree() {
		cudaFree(dev_image);  // no-op if dev_image is null
		cudaFree(dev_paths);
		cudaFree(dev_geoms);
		cudaFree(dev_materials);
		cudaFree(dev_intersections);

		// TODO: clean up any extra device memory you created
		cudaFree(dev_ind);
		cudaFree(dev_remain);
		cudaFree(dev_bool);

		delete(host_bool);
		delete(host_ind);
		delete(host_aftercompact);

		//cudaFree(dev_aftercompact);
		cudaFree(dev_paths_finish);

		cudaFree(dev_firstpath);
		cudaFree(dev_firstbounce);

#if DIRECT_LIGHT
		cudaFree(dev_lights);
#endif

		cudaFree(dev_tri);

		checkCUDAError("pathtraceFree");
	}
	//-1-1 circle
	__host__ __device__ glm::vec2 squareToDiskConcentric(glm::vec2 xi) {
		float r = xi.x;
		float theta = 2.0f * PI * xi.y;
		return glm::vec2(r * cos(theta), r * sin(theta));
	}

	/**
	* Generate PathSegments with rays from the camera through the screen into the
	* scene, which is the first bounce of rays.
	*
	* Antialiasing - add rays for sub-pixel sampling
	* motion blur - jitter rays "in time"
	* lens effect - jitter ray origin positions based on a lens
	*/
	__global__ void generateRayFromCamera(Camera cam, int iter,
		int traceDepth, PathSegment* pathSegments) {
		int x = (blockIdx.x * blockDim.x) + threadIdx.x;
		int y = (blockIdx.y * blockDim.y) + threadIdx.y;

		if (x < cam.resolution.x && y < cam.resolution.y) {
			int index = x + (y * cam.resolution.x);
			//find this pathsegment
			PathSegment & segment = pathSegments[index];
			//from camera
			segment.ray.origin = cam.position;

			//start from white color
#if DIRECT_LIGHT
			segment.color = glm::vec3(0.0f, 0.0f, 0.0f);
#else
			segment.color = glm::vec3(1.0f, 1.0f, 1.0f);
#endif

			///////////////////// TODO: implement antialiasing by jittering the ray
#if ANTI_ALIASING
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, traceDepth);
			thrust::uniform_real_distribution<float> u01(0.f, 1.0f);
			float Jitter1 = u01(rng), Jitter2 = u01(rng);
			if (x > cam.resolution.x) {
				Jitter1 = -Jitter1;
			}
			if (y > cam.resolution.y) {
				Jitter2 = -Jitter2;
			}

			glm::vec3 dir = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + Jitter1)
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + Jitter2)
			);
#else
			glm::vec3 dir = glm::normalize(cam.view
				- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
				- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);
#endif
			/////////////////////lens camera
			if (cam.focaldistance > -1.0 && cam.lenradius > -1.0) {//is lens camera
				thrust::default_random_engine rng2 = makeSeededRandomEngine(iter, index, traceDepth);
				thrust::uniform_real_distribution<float> u02(0.f, 1.0f);
				glm::vec2 xi = glm::vec2(u02(rng2), u02(rng2));

				glm::vec2 pLens = glm::vec2(cam.lenradius * squareToDiskConcentric(xi));
				glm::vec3 pFocus = cam.position + dir * (cam.focaldistance / AbsDot(dir, cam.view));

				Point3f o = cam.position + pLens.x * cam.right + pLens.y * cam.up;
				segment.ray.direction = glm::normalize(pFocus - o);
				segment.ray.origin = o;
			}
			else {
				segment.ray.direction = dir;
			}

			//////////////////// view is the camera.lookAt - camera.position
			//segment.ray.direction = glm::normalize(cam.view
			//	- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + Jitter1)
			//	- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + Jitter2)
			//	);
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
		, int num_paths // total amount of path
		, PathSegment * pathSegments
		, Geom * geoms // vector of geometry
		, int geoms_size //count of geometry
		, ShadeableIntersection * intersections //vector of intersection
		, Triangle * tri
	)
	{
		int path_index = blockIdx.x * blockDim.x + threadIdx.x;

		if (path_index < num_paths) {
			//get path segment
			PathSegment pathSegment = pathSegments[path_index];

			float t;
			glm::vec3 intersect_point;
			glm::vec3 normal;
			float t_min = FLT_MAX;//infinite
			int hit_geom_index = -1;//the index of the hit geometry
			bool outside = true;//

			glm::vec3 tmp_intersect;
			glm::vec3 tmp_normal;

			// naive parse through global geoms
			//for each geo
			for (int i = 0; i < geoms_size; i++) {
				Geom & geom = geoms[i];//get it

				if (geom.type == CUBE) {
					//get global space normal and global space intersection 
					t = boxIntersectionTest(geom, pathSegment.ray,
						tmp_intersect, tmp_normal, outside);
				}
				else if (geom.type == SPHERE) {
					t = sphereIntersectionTest(geom, pathSegment.ray,
						tmp_intersect, tmp_normal, outside);
				}
				else if (geom.type == MESH) {
					t = meshIntersectionTest(geom, tri, pathSegment.ray, tmp_normal);
				}

				// Compute the minimum t from the intersection tests to determine what
				// scene geometry object was hit first.
				if (t > 0.0f && t_min > t) {
					t_min = t;//update t
					hit_geom_index = i;
					intersect_point = tmp_intersect;//glo
					normal = tmp_normal;//global
				}
			}

			if (hit_geom_index == -1) {
				intersections[path_index].t = -1.0f;//no intersect-> t =-1
			}
			else {
				//The ray hits something
				intersections[path_index].t = t_min;
				intersections[path_index].materialId = geoms[hit_geom_index].materialid;
				intersections[path_index].surfaceNormal = normal;//global
				intersections[path_index].outside = outside;
				//intersections[path_index].point = intersect_point;
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
	__global__ void shadeFakeMaterial(
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
			}
			else {
				pathSegments[idx].color = glm::vec3(0.0f);
			}
		}
	}

	//if there is no bounce, 0, if still have, 1
	__global__ void kernisRayTerminal(int n, int *bol, const PathSegment *p) {
		int index = threadIdx.x + (blockIdx.x * blockDim.x);
		if (index < n) {
			if (p[index].remainingBounces <= 0) {
				bol[index] = 0;
			}
			else {
				bol[index] = 1;
			}
		}
	}



	__global__ void kernbsdfShader(
		int iter
		, int num_paths
		, ShadeableIntersection * shadeableIntersections
		, PathSegment * pathSegments
		, Material * materials
#if DIRECT_LIGHT
		, Geom * lights
		, Geom * geom
		, int light_count//light count
		, int geo_count//geo count
		, Triangle *tri
#endif
	) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < num_paths) {
			//get intersection
			ShadeableIntersection intersection = shadeableIntersections[index];
			if (intersection.t > 0.0f) {
				thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);//crash pathSegments->remainingBounces);
				PathSegment p = pathSegments[index];
				Material m = materials[intersection.materialId];
				glm::vec3 interp = p.ray.origin + p.ray.direction * intersection.t;//get inter point
				//void scatterRay(PathSegment & pathSegment, glm::vec3 intersect,
				//glm::vec3 normal, const Material &m, thrust::default_random_engine &rng)
				//note: here pathsseg will be change! give the pointer!
				scatterRay2(pathSegments[index], interp, intersection.surfaceNormal,
					m,
					rng
#if	DIRECT_LIGHT
					, lights
					, light_count
					, geom
					, geo_count
					, materials
					, tri
#endif	
				);
			}
			else {//no intersection
				pathSegments[index].color = glm::vec3(0.0);
				pathSegments[index].remainingBounces = -1;
			}
		}
	}

	__global__ void kerngetMaterial(int nPaths, int* matind,
		const ShadeableIntersection* inter) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nPaths) {
			matind[index] = inter[index].materialId;
		}
	}

	// Add the current iteration's output to the overall image
	__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < nPaths) {
			PathSegment iterationPath = iterationPaths[index];
			image[iterationPath.pixelIndex] += iterationPath.color;//update path color into pixel
		}
	}

	//update remain ray
	__global__ void  kernUpdateRay(int n, const PathSegment *dev_path,
		PathSegment *new_path, int *bol, int *ind) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < n) {
			if (bol[index] == 1) {
				new_path[ind[index]] = dev_path[index];
			}
		}
	}

	__global__ void kernstoreFinishedPath(int n, const PathSegment *dev_paths, PathSegment *finish) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		if (index < n) {
			if (dev_paths[index].remainingBounces <= 0) {
				//set into origin place
				int originind = dev_paths[index].pixelIndex;
				finish[originind] = dev_paths[index];
			}
		}
	}

	//http://thrust.github.io/doc/group__sorting_gaec4e3610a36062ee3e3d16607ce5ad80.html#gaec4e3610a36062ee3e3d16607ce5ad80
	//https://stackoverflow.com/questions/9037906/fast-cuda-thrust-custom-comparison-operator
	struct material_cmp
	{
		__host__ __device__ bool operator()(const ShadeableIntersection &lhs, const ShadeableIntersection &rhs) const {
			return lhs.materialId < rhs.materialId;
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
		utilityCore::PerformanceTimer timer1;
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

		////////////////////initial the ray first from the pixel and camera/////////////////////
		generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
		checkCUDAError("generate camera ray");

		//depth 0-> tracedepth
		int depth = 0;
		PathSegment* dev_path_end = dev_paths + pixelcount;
		int num_paths = dev_path_end - dev_paths;//pixel count

		// --- PathSegment Tracing Stage ---
		////////////////////Shoot ray into scene, bounce between objects, push shading chunks/////////////////////
		bool iterationComplete = false;
		int whiletime = 0;
		timer1.startGpuTimer();
		while (!iterationComplete && depth <= traceDepth) {
			//Set the default trace flag to true
			bool traceRayIntersection = true;
			std::cout << "whiletime: " << whiletime << std::endl;
			// tracing
			dim3 blocksPerGrid1d = (num_paths + blockSize1d - 1) / blockSize1d;
			// clean shading chunks
			cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if CACHE_FIRST_BOUNCE
			//get the stored intersection
			if (depth == 0 && iter > 1) {//first time and iter more than 1
				cudaMemcpy(dev_intersections, dev_firstbounce, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
				cudaMemcpy(dev_paths, dev_firstpath, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
				traceRayIntersection = false;
			}
#endif
			//get intersection
			if (traceRayIntersection) {
				computeIntersections << <blocksPerGrid1d, blockSize1d >> > (
					depth
					, num_paths
					, dev_paths
					, dev_geoms
					, hst_scene->geoms.size()
					, dev_intersections,
					NULL
					);
				checkCUDAError("computed intersections");
				cudaDeviceSynchronize();
			}

			//store the first intersection in first iteration for subsequent iterations
#if CACHE_FIRST_BOUNCE	
			if (depth == 0 && iter == 1) {
				cudaMemcpy(dev_firstbounce, dev_intersections, num_paths * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
				cudaMemcpy(dev_firstpath, dev_paths, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
			}
#endif

			// --- Shading Stage ---

			if (is_sort) {
				thrust::device_ptr<ShadeableIntersection> dev_thrust_keys(dev_intersections);
				thrust::device_ptr<PathSegment> dev_thrust_values(dev_paths);
				//thrust::sort_by_key（d_keys.begin ，d_keys.end（），d_values.begin（）,cmp）
				thrust::sort_by_key(thrust::device, dev_thrust_keys, dev_thrust_keys + num_paths, dev_thrust_values, material_cmp());
			}
			depth++;

			
			//timer1.startGpuTimer();
			//handle material! 
			kernbsdfShader <<<blocksPerGrid1d, blockSize1d>>> (
				iter,
				num_paths,
				dev_intersections,
				dev_paths,
				dev_materials
#if DIRECT_LIGHT
				, dev_lights
				, dev_geoms
				, li_count//light count
				, geo_count//geo count
				, dev_tri
#endif
				);
			checkCUDAError("bsdf error!");
			cudaDeviceSynchronize();
			//timer1.endGpuTimer();
			//utilityCore::printElapsedTime(timer1.getGpuElapsedTimeForPreviousOperation(), "shading");			

			//store finish path
			dim3 blocksPerGrid1d_f = (num_paths + blockSize1d - 1) / blockSize1d;
			kernstoreFinishedPath << <blocksPerGrid1d_f, blockSize1d >> > (num_paths, dev_paths, dev_paths_finish);
			checkCUDAError("store error!");
			cudaDeviceSynchronize();

			/////////////////////////////stream compaction for ray /////////////////////////////
			//change ray path to bool array, prepare for the stream compact

			//timer1.startGpuTimer();
			kernisRayTerminal << <blocksPerGrid1d_f, blockSize1d >> > (num_paths, dev_bool, dev_paths);
			cudaDeviceSynchronize();
			//timer1.endGpuTimer();
			//utilityCore::printElapsedTime(timer1.getGpuElapsedTimeForPreviousOperation(), "to bool");
			/*int *bl = new int[pixelcount];
			 cudaMemcpy(bl, dev_bool, pixelcount * sizeof(int), cudaMemcpyDeviceToHost);
			 //std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
			 int num_o = 0, num_z=0;
			 for (int a = 0; a < num_paths; a++) {
				   if (bl[a] != 1) {
					   num_z++;
				   }
				   else {
					   num_o++;
				   }
			   }
			 if (depth == 1) {
				 std::cout <<"total path num: " << num_paths << std::endl;
			 }
			 std::cout <<"terminal num: "<< num_z << std::endl;
			 std::cout << "non terminal num: " << num_o << std::endl;
			 std::cout << "		|		"<< std::endl;*/

			 //get the index
			cudaMemcpy(host_bool, dev_bool, pixelcount * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("host_ind copy error!");
			//timer1.startGpuTimer();
			StreamCompaction::Efficient::scan(num_paths, host_ind, host_bool);
			checkCUDAError("scan error!");
			//timer1.endGpuTimer();
			//utilityCore::printElapsedTime(timer1.getGpuElapsedTimeForPreviousOperation(), "scan");

			cudaMemcpy(dev_ind, host_ind, num_paths * sizeof(int), cudaMemcpyHostToDevice);
			//timer1.startGpuTimer();
			//update dev_remain
			kernUpdateRay << <blocksPerGrid1d_f, blockSize1d >> > (num_paths, dev_paths, dev_remain, dev_bool, dev_ind);
			checkCUDAError("update ray error!");
			cudaDeviceSynchronize();
			//timer1.endGpuTimer();
			//utilityCore::printElapsedTime(timer1.getGpuElapsedTimeForPreviousOperation(), "compact");

			//update num_paths
			num_paths = StreamCompaction::Efficient::compact(num_paths, host_aftercompact, host_bool);
			//std::cout << "update path: " << num_paths << std::endl;

			//update dev_path data
			cudaMemset(dev_paths, 0, pixelcount * sizeof(PathSegment));
			cudaMemcpy(dev_paths, dev_remain, num_paths * sizeof(PathSegment), cudaMemcpyDeviceToDevice);

			if (num_paths == 0 || depth >= traceDepth) {
				//if no num path, complete!
				cudaMemcpy(dev_paths, dev_paths_finish, pixelcount * sizeof(PathSegment), cudaMemcpyDeviceToDevice);
				iterationComplete = true;
			}
			whiletime++;
		}
		timer1.endGpuTimer();
		utilityCore::printElapsedTime(timer1.getGpuElapsedTimeForPreviousOperation(), "compact");

		// Assemble this iteration and apply it to the image
		dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
		//finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);//num_path is 0
		finalGather << <numBlocksPixels, blockSize1d >> > (pixelcount, dev_image, dev_paths);

		///////////////////////////////////////////////////////////////////////////
		// Send results to OpenGL buffer for rendering
		sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

		// Retrieve image from GPU
		cudaMemcpy(hst_scene->state.image.data(), dev_image, pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

		checkCUDAError("pathtrace");
	}
}