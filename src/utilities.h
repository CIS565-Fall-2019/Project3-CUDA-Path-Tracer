#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>

// Performance and Feature Switches

#define SORTING_MATERIAL 1//pretty sure this fucks performance
#define CACHING_FIRST 1
#define ANTIALIASING 1
#define USING_OIDN 1
#define ANY_REFRACTIVE 1

#define TEX_COLOR 1
#define TEX_EMISSIVE 0
#define TEX_ROUGH 0
#define TEX_NORM 0

// Useful math symbols

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

//soft culling of glm namespace
typedef glm::vec2 gvec2;
typedef glm::vec3 gvec3;
typedef glm::vec4 gvec4;
typedef glm::ivec2 ivec2;
typedef glm::ivec3 ivec3;
typedef glm::ivec4 ivec4;
typedef glm::mat3 gmat3;
typedef glm::mat4 gmat4;

//vector typedefs
typedef std::vector<int> int_v;
typedef std::vector<float> float_v;
typedef std::vector<gvec3> gvec3_v;
typedef std::vector<gvec4> gvec4_v;
typedef std::vector<ivec3> ivec3_v;
typedef std::vector<ivec4> ivec4_v;
typedef std::vector<std::string> string_v;

struct DebugVector {
	float x;
	float y;
	float z;
};

/**
Will be useful for going through our textures
*/
typedef struct f4vec {
	float r;
	float g;
	float b;
	float a;
} f4vec;


//Preprocessor functions that you really shouldn't trust, but they might do what we want
//note: only work for gvec3 types

//Preprocessor macro for dot product
#define DOTP(a, b) (a.x * b.x + a.y * b.y + a.z * b.z)
//Preprocessor macro for cross product
#define CROSSP(a, b) (gvec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x))
//Preprocessor macro for ray reflection of a (incoming) over n (normal)
#define REFLECT(a, n) (a - (2 * DOTP(a, n)) * n)

#define MAT3I (gmat3(1, 0, 0, 0, 1, 0, 0, 0, 1))
#define MAT4I (gmat4(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1))

//some stupid bits of math
/** Returns a normalized version of the vector **/
extern inline gvec3 normalized(const gvec3 input);
/** Normalizes the vector in place **/
extern inline void normalize(gvec3* input);
/** Calculates the magnitude of the vector **/
extern inline float magnitude(const gvec3 input);
/** Calculates the magnitude of the vector **/
extern inline float magnitude(const gvec4 input);

namespace utilityCore {
    extern float clamp(float f, float min, float max);
    extern bool replaceString(std::string& str, const std::string& from, const std::string& to);
    extern glm::vec3 clampRGB(glm::vec3 color);
    extern bool epsilonCheck(float a, float b);
    extern std::vector<std::string> tokenizeString(std::string str);
    extern glm::mat4 buildTransformationMatrix(glm::vec3 translation, glm::vec3 rotation, glm::vec3 scale);
    extern std::string convertIntToString(int number);
    extern std::istream& safeGetline(std::istream& is, std::string& t); //Thanks to http://stackoverflow.com/a/6089413
}
