#include "texture.h"

//Note: including the period because these come right before the extension
const std::string Texture::baseColorString = std::string("baseColor.");
const std::string Texture::emissiveString = std::string("emissive.");
const std::string Texture::metallicRoughnessString = std::string("metallicRoughness.");
const std::string Texture::normalString = std::string("normal.");

uint8_t Texture::createFromGltfVector(std::vector<example::Texture> inputs) {
	
	for (example::Texture gtext : inputs) {
		width = gtext.width;
		height = gtext.height;//really hope these are the same for each

		if (gtext.name.find(baseColorString) != std::string::npos) {
			texturePresenceMask |= TEXTURE_BASECOLOR;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					int index = ((i * width) + j) * 4;
					baseColor.push_back({
						(gtext.image[index + 0] / 255.0f),
						(gtext.image[index + 1] / 255.0f),
						(gtext.image[index + 2] / 255.0f),
						(gtext.image[index + 3] / 255.0f)
						});
				}//for
			}//for

		}//base color
		else if (gtext.name.find(emissiveString) != std::string::npos) {
			texturePresenceMask |= TEXTURE_EMISSIVE;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					int index = ((i * width) + j) * 4;
					emissive.push_back({
						(gtext.image[index + 0] / 255.0f),
						(gtext.image[index + 1] / 255.0f),
						(gtext.image[index + 2] / 255.0f),
						(gtext.image[index + 3] / 255.0f)
						});
				}//for
			}//for
		}//emissive
		else if (gtext.name.find(metallicRoughnessString) != std::string::npos) {
			texturePresenceMask |= TEXTURE_METALLICROUGHNESS;//likely unused
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					int index = ((i * width) + j) * 4;
					metallicRoughness.push_back({
						(gtext.image[index + 0] / 255.0f),
						(gtext.image[index + 1] / 255.0f),
						(gtext.image[index + 2] / 255.0f),
						(gtext.image[index + 3] / 255.0f)
						});
				}//for
			}//for
		}//metallic roughness
		else if (gtext.name.find(normalString) != std::string::npos) {
			texturePresenceMask |= TEXTURE_NORMAL;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					int index = ((i * width) + j) * 4;
					normal.push_back({
						(float)((gtext.image[index + 0] / 255.0) * 2.0 - 1.0),
						(float)((gtext.image[index + 1] / 255.0) * 2.0 - 1.0),
						(float)((gtext.image[index + 2] / 255.0) * 2.0 - 1.0),
						(float)((gtext.image[index + 3] / 255.0) * 2.0 - 1.0)
						});
				}//for
			}//for
		}//normal

	}//for each input texture

	return texturePresenceMask;

}//createFromGltfVector


void Texture::fillIntoF4Array(float4* dst) {
	if (texturePresenceMask & TEXTURE_BASECOLOR) memcpy(&dst[0 * width * height], baseColor.data(), width * height * sizeof(f4vec));
	if (texturePresenceMask & TEXTURE_EMISSIVE) memcpy(&dst[1 * width * height], emissive.data(), width * height * sizeof(f4vec));
	if (texturePresenceMask & TEXTURE_METALLICROUGHNESS) memcpy(&dst[2 * width * height], metallicRoughness.data(), width * height * sizeof(f4vec));
	if (texturePresenceMask & TEXTURE_NORMAL) memcpy(&dst[3 * width * height], normal.data(), width * height * sizeof(f4vec));
}

void Texture::freeFromDevice() {
	cudaFreeArray(cu_3darray);
}//freeFromDevice