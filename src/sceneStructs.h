#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

enum GeomType {
    SPHERE,
    CUBE,
	MESH
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Geom {
    enum GeomType type;
    int materialid;

    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;

    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

	int Tri_start_Idx;
	int Tri_end_Idx;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;//index of refraction for Fresnel effects
    float emittance;//the emittance strength of the material. 
	//Material is a light source if emittance > 0.
};

struct Camera {
    glm::ivec2 resolution;// width, height
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
	float focaldistance;
	float lenradius;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;//how many iterations to refine the image
    int traceDepth;//depth of bounce
    std::vector<glm::vec3> image;
    std::string imageName;//name 
};

struct PathSegment {
	Ray ray;
	glm::vec3 color;
	int pixelIndex;
	int remainingBounces;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  glm::vec3 surfaceNormal;
  int materialId;
  bool outside;
  glm::vec3 point;//intersect point
};

struct Triangle {
	int index;
	glm::vec3 vertices[3];
	glm::vec3 normals[3];
	//glm::vec2 uvs[3];
};

struct Light {
	int index;
	glm::vec3 vertices[3];
	glm::vec3 normals[3];
	//glm::vec2 uvs[3];
};

