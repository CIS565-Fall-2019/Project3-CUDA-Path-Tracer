#ifndef TEXTURE_H
#define TEXTURE_H

#include "utilities.h"
#include <vector>
#include "gltf-loader.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>


#define TEXTURE_BASECOLOR 0x01
#define TEXTURE_EMISSIVE 0x02
#define TEXTURE_METALLICROUGHNESS 0x04
#define TEXTURE_NORMAL 0x08
#define TEXTURE_ANY 0x0F
#define TEXTURE_LAYER_BASECOLOR 0
#define TEXTURE_LAYER_EMISSIVE 1
#define TEXTURE_LAYER_METALLICROUGHNESS 2
#define TEXTURE_LAYER_NORMAL 3

class Texture {
private:
	const static std::string baseColorString;
	const static std::string emissiveString;
	const static std::string metallicRoughnessString;
	const static std::string normalString;

	void fillIntoF4Array(float4* dst);

public:
	cudaArray* cu_3darray;
	std::vector<f4vec> baseColor = std::vector<f4vec>();
	std::vector<f4vec> emissive = std::vector<f4vec>();
	std::vector<f4vec> metallicRoughness = std::vector<f4vec>();
	std::vector<f4vec> normal = std::vector<f4vec>();
	int width = -1;
	int height = -1;

	uint8_t texturePresenceMask = 0x00;

	uint8_t createFromGltfVector(std::vector<example::Texture> inputs);

	cudaArray* putOntoDevice(int textureIndex);

	void freeFromDevice();

};//Texture

typedef std::vector<Texture> Texture_v;

#endif