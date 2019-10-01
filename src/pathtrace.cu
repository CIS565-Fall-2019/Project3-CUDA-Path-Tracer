#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/partition.h>
#include <thrust/device_ptr.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtx/intersect.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"


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
static int faceSize = 0;
static glm::vec3 * dev_image = NULL;
#if GLTF
static Triangle * dev_geoms = NULL;
static MyMaterial * dev_materials = NULL;
static unsigned char * dev_texture = NULL;
	#if BBOX
static float * dev_bbox = NULL;
	#endif
#else
static Geom * dev_geoms = NULL;
static Material * dev_materials = NULL;
#endif
static PathSegment * dev_paths = NULL;
static ShadeableIntersection * dev_intersections = NULL;
#if CACHEFIRST && !MOTIONBLUR && !DEPTHOFFIELD
static ShadeableIntersection * dev_cache_intersections = NULL;
#endif


void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

  	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

#if GLTF
	#if BBOX
	std::vector<float> bbox;
	float idx = 0.0f;
	#endif
	std::vector<Triangle> faces;
	for (const auto mesh : scene->meshes) {
	#if BBOX
		idx += mesh.faces.size() / 3.0f;
		bbox.push_back(idx);
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 3; j++) {
				bbox.push_back(mesh.bbox[i][j]);
			}
		}
	#endif
		if (mesh.facevarying_uvs.size() > 0) {
			for (int i = 0; i < mesh.faces.size(); i++) {
				Triangle t;
				t.materialid = mesh.material_ids[i];
				t.normal = glm::vec3(mesh.facevarying_normals[3 * i],
					mesh.facevarying_normals[3 * i + 1],
					mesh.facevarying_normals[3 * i + 2]);
				int vertexId = mesh.faces[i];
				t.v1 = glm::vec3(mesh.vertices[3 * vertexId], mesh.vertices[3 * vertexId + 1], mesh.vertices[3 * vertexId + 2]);
				t.uv1 = glm::vec2(mesh.facevarying_uvs[2 * i], mesh.facevarying_uvs[2 * i + 1]);
				vertexId = mesh.faces[++i];
				t.v2 = glm::vec3(mesh.vertices[3 * vertexId], mesh.vertices[3 * vertexId + 1], mesh.vertices[3 * vertexId + 2]);
				t.uv2 = glm::vec2(mesh.facevarying_uvs[2 * i], mesh.facevarying_uvs[2 * i + 1]);
				vertexId = mesh.faces[++i];
				t.v3 = glm::vec3(mesh.vertices[3 * vertexId], mesh.vertices[3 * vertexId + 1], mesh.vertices[3 * vertexId + 2]);
				t.uv3 = glm::vec2(mesh.facevarying_uvs[2 * i], mesh.facevarying_uvs[2 * i + 1]);
				faces.push_back(t);
			}
		}
		else {
			for (int i = 0; i < mesh.faces.size(); i++) {
				Triangle t;
				t.materialid = mesh.material_ids[i];
				t.normal = glm::vec3(mesh.facevarying_normals[3 * i],
					mesh.facevarying_normals[3 * i + 1],
					mesh.facevarying_normals[3 * i + 2]);
				int vertexId = mesh.faces[i];
				t.v1 = glm::vec3(mesh.vertices[3 * vertexId], mesh.vertices[3 * vertexId + 1], mesh.vertices[3 * vertexId + 2]);
				t.uv1 = glm::vec2(0);
				vertexId = mesh.faces[++i];
				t.v2 = glm::vec3(mesh.vertices[3 * vertexId], mesh.vertices[3 * vertexId + 1], mesh.vertices[3 * vertexId + 2]);
				t.uv2 = glm::vec2(0);
				vertexId = mesh.faces[++i];
				t.v3 = glm::vec3(mesh.vertices[3 * vertexId], mesh.vertices[3 * vertexId + 1], mesh.vertices[3 * vertexId + 2]);
				t.uv3 = glm::vec2(0);
				faces.push_back(t);
			}
		}
	}
	faceSize = faces.size();
	cudaMalloc(&dev_geoms, faces.size() * sizeof(Triangle));
	cudaMemcpy(dev_geoms, faces.data(), faces.size() * sizeof(Triangle), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->mymaterials.size() * sizeof(MyMaterial));
	cudaMemcpy(dev_materials, scene->mymaterials.data(), scene->mymaterials.size() * sizeof(MyMaterial), cudaMemcpyHostToDevice);
	// todo:: multiple textures
	Texture tx = scene->textures.at(0);
	int txnum = tx.height * tx.width * tx.components;
	cudaMalloc(&dev_texture, txnum * sizeof(unsigned char));
	cudaMemcpy(dev_texture, scene->textures.at(0).image, txnum * sizeof(unsigned char), cudaMemcpyHostToDevice);

	#if BBOX
	cudaMalloc(&dev_bbox, scene->meshes.size() * 7 * sizeof(float));
	cudaMemcpy(dev_bbox, bbox.data(), scene->meshes.size() * 7 * sizeof(float), cudaMemcpyHostToDevice);
	#endif

#else
	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
#endif

  	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
  	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

#if CACHEFIRST && !MOTIONBLUR && !DEPTHOFFIELD
	cudaMalloc(&dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection));
	cudaMemset(dev_cache_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

    checkCUDAError("pathtraceInit");
}

void resetGeoms() {
	cudaMemcpy(dev_geoms, hst_scene->geoms.data(), hst_scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
}

void pathtraceFree() {
    cudaFree(dev_image);  // no-op if dev_image is null
  	cudaFree(dev_paths);
  	cudaFree(dev_geoms);
  	cudaFree(dev_materials);
  	cudaFree(dev_intersections);

#if CACHEFIRST && !MOTIONBLUR && !DEPTHOFFIELD
	cudaFree(dev_cache_intersections);
#endif

#if GLTF
	cudaFree(dev_texture);
	#if BBOX
	cudaFree(dev_bbox);
	#endif
#endif

    checkCUDAError("pathtraceFree");
}

__device__ glm::vec3 squareToDiskConcentric(const glm::vec2 &sample) {
	float phi, r, u, v;
	float a = 2 * sample.x - 1;
	float b = 2 * sample.y - 1;
	if (a > -b) {
		if (a > b) {
			r = a;
			phi = (PI / 4.f) * (b / a);
		}
		else {
			r = b;
			phi = (PI / 4.f) * (2.f - (a / b));
		}
	}
	else {
		if (a < b) {
			r = -a;
			phi = (PI / 4.f) * (4.f + (b / a));
		}
		else {
			r = -b;
			if (b != 0) {
				phi = (PI / 4.f) * (6.f - (a / b));
			}
			else {
				phi = 0;
			}
		}
	}
	u = r * std::cos(phi);
	v = r * std::sin(phi);

	return glm::vec3(u, v, 0.f);
}


__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment & segment = pathSegments[index];

		segment.ray.origin = cam.position;
		segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
			);

#if DEPTHOFFIELD
		thrust::uniform_real_distribution<float> u01(0, 1);
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, segment.remainingBounces);
		glm::vec3 bias = squareToDiskConcentric(glm::vec2(u01(rng), u01(rng)));
		float t = glm::abs(FOCALDIS / glm::dot(segment.ray.direction, cam.view));
		segment.ray.origin += bias * LENSR;
		segment.ray.direction = glm::normalize(t * segment.ray.direction - bias * LENSR);
#endif

		segment.pixelIndex = index;
		segment.remainingBounces = traceDepth;
	}
}

__global__ void computeIntersections(
	int depth
	, int num_paths
	, PathSegment * pathSegments
#if GLTF
	, Triangle *geoms,
	unsigned char *texture,
	int txWidth,
	int txHeight
	#if BBOX
	, float* bbox
	#endif
#else
	, Geom * geoms
#endif
	, int geoms_size
	, ShadeableIntersection * intersections
	)
{
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		float t;
		glm::vec3 intersect_data;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		bool outside = true;

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		// naive parse through global geoms
#if GLTF
	#if BBOX
		int i = 0;
		for(int j = 0;; j++) {
			Ray r = pathSegment.ray;
			if (!checkBBox(glm::vec3(bbox[7 * j + 1], bbox[7 * j + 2], bbox[7 * j + 3]), 
				glm::vec3(bbox[7 * j + 4], bbox[7 * j + 5], bbox[7 * j + 6]), r.origin, r.direction)) {
				if (bbox[7 * j] == geoms_size) {
					break;
				}
				else {
					i = bbox[7 * j];
					continue;
				}
			}
			for (; i < bbox[7 * j]; i++) {
				Triangle &tri = geoms[i];
				if (glm::intersectRayTriangle(r.origin,
					r.direction, tri.v1, tri.v2, tri.v3, tmp_intersect)) {
					if (t_min > tmp_intersect.z)
					{
						t_min = tmp_intersect.z;
						hit_geom_index = i;
						intersect_data = tmp_intersect;
						normal = tri.normal;
					}
				}
			}
			if (bbox[7 * j] == geoms_size) {
				break;
			}
		}
	#else
		for (int i = 0; i < geoms_size; i++)
		{
			Triangle &tri = geoms[i];
			Ray r = pathSegment.ray;
			if (glm::intersectRayTriangle(r.origin,
				r.direction, tri.v1, tri.v2, tri.v3, tmp_intersect)) {
				if (t_min > tmp_intersect.z)
				{
					t_min = tmp_intersect.z;
					hit_geom_index = i;
					intersect_data = tmp_intersect;
					normal = tri.normal;
				}
			}
		}
	#endif
#else
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
			
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				normal = tmp_normal;
			}
		}
#endif
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
#if GLTF
			Triangle &tri = geoms[hit_geom_index];
			glm::vec2 uv = tri.uv1 * intersect_data.x + tri.uv2 * intersect_data.y + tri.uv3 * (1 - intersect_data.y - intersect_data.x);
			int xi = glm::floor(uv.x * txWidth);
			int yi = glm::floor(uv.y * txHeight);
			float xf = uv.x * txWidth - xi;
			float yf = uv.y * txHeight - yi;

			int idx = (xi + yi * txWidth) * 4;
			glm::vec3 bl = glm::vec3(texture[idx], texture[idx + 1], texture[idx + 2]);

			idx = (xi + 1 + yi * txWidth) * 4;
			glm::vec3 br = glm::vec3(texture[idx], texture[idx + 1], texture[idx + 2]);

			idx = (xi + (yi + 1) * txWidth) * 4;
			glm::vec3 ul = glm::vec3(texture[idx], texture[idx + 1], texture[idx + 2]);

			idx = (xi + 1 + (yi + 1) * txWidth) * 4;
			glm::vec3 ur = glm::vec3(texture[idx], texture[idx + 1], texture[idx + 2]);

			intersections[path_index].texColor = xf * yf * ur + xf * (1 - yf) * br + (1 - xf) * yf * ul + (1 - xf) * (1 - yf) * bl;
#endif
		}
	}
}

__global__ void shadeMaterial (
	int iter, 
	int num_paths, 
	ShadeableIntersection * shadeableIntersections, 
	PathSegment * pathSegments,
#if GLTF
	MyMaterial * materials
#else
	Material * materials
#endif
)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    ShadeableIntersection intersection = shadeableIntersections[idx];
    if (intersection.t > 0.0f) {
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

#if GLTF
	  MyMaterial material = materials[intersection.materialId];
	  // If the material indicates that the object was a light, "light" the ray
	  //if (material.emittance > 0.0f) {
	//	  pathSegments[idx].color *= (materialColor * material.emittance);
	//	  pathSegments[idx].remainingBounces = -1;
	//  }
	  // Otherwise, do some pseudo-lighting computation. This is actually more
	  // like what you would expect from shading in a rasterizer like OpenGL.
	  // TODO: replace this! you should be able to start with basically a one-liner
	//  else {
		  pathSegments[idx].remainingBounces--;
		  if (pathSegments[idx].remainingBounces < 0) {
			  pathSegments[idx].color = glm::vec3(0);
		  }
		  else {
			  pathSegments[idx].color *= intersection.texColor / 255.0f;
			  scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t),
				  intersection.surfaceNormal, material, rng);
		  }
	//  }
	  // If there was no intersection, color the ray black.
	  // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
	  // used for opacity, in which case they can indicate "no opacity".
	  // This can be useful for post-processing and image compositing.
	}
	else {
		glm::vec3 nn;
		skyboxTest(pathSegments->ray, nn);
		pathSegments[idx].color *= glm::clamp((1.5f * (1.0f - nn.z) * glm::vec3(0.6f, 0.95f, 1.1f) + 13.0f * glm::pow(glm::dot(nn, glm::vec3(0.44f, -0.4f, 0.8f)), 30)), 0.0f, 8.0f);
		pathSegments[idx].remainingBounces = -1;
	}
  }
#else
      Material material = materials[intersection.materialId];
      glm::vec3 materialColor = material.color;

      if (material.emittance > 0.0f) {
		  pathSegments[idx].color *= (materialColor * material.emittance);
		  pathSegments[idx].remainingBounces = -1;
      }
      else {
		  pathSegments[idx].remainingBounces--;
		  if (pathSegments[idx].remainingBounces < 0) {
			  pathSegments[idx].color = glm::vec3(0.0f);
		  }
		  else {
			  scatterRay(pathSegments[idx], getPointOnRay(pathSegments[idx].ray, intersection.t),
				  intersection.surfaceNormal, material, rng);
		  }
      }
    
    } else {
		pathSegments[idx].color = glm::vec3(0.0f);
		pathSegments[idx].remainingBounces = -1;
    }
  }
#endif
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

struct is_ended {
	__host__ __device__
		bool operator() (const PathSegment &ps) {
		return ps.remainingBounces >= 0;
	}
};

struct matCompare {
	__host__ __device__ 	
		bool operator()(const ShadeableIntersection &a, const ShadeableIntersection &b) {
		return a.materialId < b.materialId;
	}
};

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

	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	// cout << cam.position.x << " "<<  cam.position.y << " " << cam.position.z <<endl;
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = pixelcount;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks
#if SORTBYMAT
	thrust::device_ptr<ShadeableIntersection> dv_intersections(dev_intersections);
	thrust::device_ptr<PathSegment> dv_paths(dev_paths);
#endif

	bool iterationComplete = false;
	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

		// tracing
		dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHEFIRST && !MOTIONBLUR && !DEPTHOFFIELD
		// Handling intersections for first bounce
		if (depth == 0) {
			if (iter == 1) {
				computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
					depth
					, num_paths
					, dev_paths
					, dev_geoms
	#if GLTF
					, dev_texture
					, hst_scene->textures.at(0).width
					, hst_scene->textures.at(0).height
		#if BBOX
					, dev_bbox
		#endif
					, faceSize
	#else
					, hst_scene->geoms.size()
	#endif
					, dev_intersections
					);
				checkCUDAError("trace one bounce");
				cudaDeviceSynchronize();
				cudaMemcpy(dev_cache_intersections, dev_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
			else {
				cudaMemcpy(dev_intersections, dev_cache_intersections, pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice);
			}
		} else {
			computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, num_paths
				, dev_paths
				, dev_geoms
	#if GLTF
				, dev_texture
				, hst_scene->textures.at(0).width
				, hst_scene->textures.at(0).height
		#if BBOX
				, dev_bbox
		#endif
				, faceSize
	#else
				, hst_scene->geoms.size()
#endif
				, dev_intersections
				);
			checkCUDAError("trace one bounce");
			cudaDeviceSynchronize();
		}
#else
		computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, num_paths
			, dev_paths
			, dev_geoms
	#if GLTF
			, dev_texture
			, hst_scene->textures.at(0).width
			, hst_scene->textures.at(0).height
		#if BBOX
			, dev_bbox
		#endif
			, faceSize
	#else
			, hst_scene->geoms.size()
#endif
			, dev_intersections
			);
		checkCUDAError("trace one bounce");
		cudaDeviceSynchronize();
#endif
		depth++;

#if SORTBYMAT
		thrust::sort_by_key(dv_intersections, dv_intersections + num_paths,
			dv_paths, matCompare());
#endif

		shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>> (
			iter,
			num_paths,
			dev_intersections,
			dev_paths,
			dev_materials
		);
		PathSegment* pivot_index = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, is_ended());
		num_paths = pivot_index - dev_paths;
		if (num_paths < 1 || depth > traceDepth) {
			iterationComplete = true;
		}
	}

  // Assemble this iteration and apply it to the image
  dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
	finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);
    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
            pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
