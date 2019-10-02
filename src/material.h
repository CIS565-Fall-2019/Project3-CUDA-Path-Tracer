#ifndef EXAMPLE_MATERIAL_H_
#define EXAMPLE_MATERIAL_H_

#include <cstdlib>

#ifdef __clang__
#pragma clang diagnostic push
#if __has_warning("-Wzero-as-null-pointer-constant")
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#endif
#endif

struct MyMaterial {
  float diffuse[3];
  float specular[3];
  int id;
  int texid;
  float metallicF;
  float roughnessF;

  MyMaterial() {
    diffuse[0] = 0.5;
    diffuse[1] = 0.5;
    diffuse[2] = 0.5;
    specular[0] = 0.5;
    specular[1] = 0.5;
    specular[2] = 0.5;
    id = -1;
	texid = -1;
	metallicF = 0.5;
	roughnessF = 0.5;
  }
};

struct Texture {
  int width;
  int height;
  int components;
  int _pad_;
  unsigned char* image;

  Texture() {
    width = -1;
    height = -1;
    components = -1;
    image = NULL;
  }
};

#endif  // EXAMPLE_MATERIAL_H_
