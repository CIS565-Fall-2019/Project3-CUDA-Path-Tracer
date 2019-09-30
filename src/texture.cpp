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

void Texture::makeProdeduralTexture(gvec3 color1, gvec3 color2, float bumpiness) {
	texturePresenceMask = TEXTURE_BASECOLOR | TEXTURE_NORMAL;//has color and normal elements

	width = 256;
	height = 256;

	baseColor = std::vector<f4vec>();
	normal = std::vector<f4vec>();

	f4vec c1 = { color1.x, color1.y, color1.z, 1.0f };
	f4vec c2 = { color2.x, color2.y, color2.z, 1.0f };
	//Put in color data
	for (int i = 0; i < height; i++) {
		uint8_t doFlip = ((i / 32) % 2) == 1;
		for (int j = 0; j < width; j++) {
			uint8_t choose1 = ((j / 32) % 2) == 1;
			uint8_t pickVal = (doFlip << 1) | choose1;
			switch (pickVal) {
			case 0x00:
			case 0x03:
				baseColor.push_back({c1.r, c1.g, c1.b, c1.a});
				break;
			case 0x02:
			case 0x01:
				baseColor.push_back({ c2.r, c2.g, c2.b, c2.a });
				break;
			}//switch
		}//for j
	}//for i



	//put in bump data
	for (int i = 0; i < height; i++) {
		int heightval = (i % 64) - 31;//ranges from -31 to 32
		float offsetY = sinf(heightval / 32.0 * PI);//from -1 to 1
		offsetY *= bumpiness * 0.3f;
		gvec3 yVec = gvec3(0, 1, 0) * offsetY;
		for (int j = 0; j < width; j++) {
			int widthval = (j % 64) - 31;//ranges from -31 to 32
			float offsetX = sinf(widthval / 32.0 * PI);
			offsetX *= bumpiness * 0.3f;
			gvec3 xVec = gvec3(1, 0, 0) * offsetX;
			gvec3 normalVec = glm::normalize(gvec3(0, 0, 1) + yVec + xVec);
			f4vec bumpVal = { normalVec.x, normalVec.y, normalVec.z, 1.0f };
			normal.push_back(bumpVal);
		}//for j
	}//for i


}