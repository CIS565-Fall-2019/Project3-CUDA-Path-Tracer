#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "utilities.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
	TRIANGLE,
	MESH,
};

struct Ray {
    gvec3 origin;
    gvec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;
    gvec3 translation;
    gvec3 rotation;
    gvec3 scale;
    gmat4 transform;
    gmat4 inverseTransform;
    gmat4 invTranspose;
	int triangleIndex;
	int triangleCount;
};//conceit that this is either a primitive or a primitive acting as a bounding box to contain a boatload of triangles

struct Triangle {
	int materialid;
	gvec3 vert0;
	gvec3 vert1;
	gvec3 vert2;
	//gvec3 normal;
	gvec3 norm0;//for interpolation
	gvec3 norm1;//for interpolation
	gvec3 norm2;//for interpolation
	float2 uv0;
	float2 uv1;
	float2 uv2;
};

struct Material {
    gvec3 color;
    struct {
        float exponent;
        gvec3 color;
    } specular;
    float hasReflective;//currently using as "proportion of specular that is precisely mirror-like"
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
	int8_t textureId;//index of texture to draw from
	uint8_t textureMask;//presence/absence of texture data to override other attributes
};

struct Camera {
    ivec2 resolution;
    gvec3 position;
    gvec3 lookAt;
    gvec3 view;
    gvec3 up;
    gvec3 right;
    gvec2 fov;
    gvec2 pixelLength;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    gvec3_v image;
    std::string imageName;
};

struct PathSegment {
	Ray ray;
	gvec3 color;
	int pixelIndex;
	int remainingBounces;
	float curIOR;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  gvec3 surfaceNormal;
  float2 uv;
  int materialId;
  bool leaving;
};

//vector typedefs
typedef std::vector<Ray>            Ray_v;
typedef std::vector<Geom>           Geom_v;
typedef std::vector<Material>       Material_v;
typedef std::vector<Camera>         Camera_v;
typedef std::vector<PathSegment>    PathSegment_v;
typedef std::vector<Triangle>		Triangle_v;



