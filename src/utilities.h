#pragma once

#include "glm/glm.hpp"
#include <algorithm>
#include <istream>
#include <ostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define PI                3.1415926535897932384626422832795028841971f
#define TWO_PI            6.2831853071795864769252867665590057683943f
#define SQRT_OF_ONE_THIRD 0.5773502691896257645091487805019574556476f
#define EPSILON           0.00001f

//soft culling of glm namespace
typedef glm::vec3 gvec3;
typedef glm::vec4 gvec4;
typedef glm::ivec3 ivec3;//likely unused
typedef glm::ivec4 ivec4;//likely unused

//vector typedefs
typedef std::vector<int> int_v;
typedef std::vector<float> float_v;
typedef std::vector<gvec3> gvec3_v;
typedef std::vector<gvec4> gvec4_v;
typedef std::vector<ivec3> ivec3_v;
typedef std::vector<ivec4> ivec4_v;
typedef std::vector<std::string> string_v;

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
