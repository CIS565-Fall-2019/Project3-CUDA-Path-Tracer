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
    gvec3 translation;//TODO: also make this vert0
    gvec3 rotation;//TODO: also make this vert1
    gvec3 scale;//TODO: also make this vert2
    gmat4 transform;
    gmat4 inverseTransform;
    gmat4 invTranspose;
	gvec3 vert0;
	gvec3 vert1;
	gvec3 vert2;
	gvec3 normal;
};

struct Material {
    gvec3 color;
    struct {
        float exponent;
        gvec3 color;
    } specular;
    float hasReflective;//use this as proportion-of-mirror-like term?
    float hasRefractive;
    float indexOfRefraction;
    float emittance;
	/* Eventually, may want to think in these terms
	gvec3 diffuse;
	gvec3 specular;
	gvec3 transmittance;
	gvec3 emission;
	*/
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
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  gvec3 surfaceNormal;
  int materialId;
};

//vector typedefs
typedef std::vector<Ray>            Ray_v;
typedef std::vector<Geom>           Geom_v;
typedef std::vector<Material>       Material_v;
typedef std::vector<Camera>         Camera_v;
typedef std::vector<PathSegment>    PathSegment_v;



