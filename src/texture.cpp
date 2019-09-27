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
				for (int j = 0; j < width * 4; j += 4) {
					baseColor.push_back({
						(gtext.image[i * width + j + 0] / 255.0f),
						(gtext.image[i * width + j + 1] / 255.0f),
						(gtext.image[i * width + j + 2] / 255.0f),
						(gtext.image[i * width + j + 3] / 255.0f)
						});
				}//for
			}//for

		}//base color
		else if (gtext.name.find(emissiveString) != std::string::npos) {
			texturePresenceMask |= TEXTURE_EMISSIVE;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width * 4; j += 4) {
					emissive.push_back({
						(gtext.image[i * width + j + 0] / 255.0f),
						(gtext.image[i * width + j + 1] / 255.0f),
						(gtext.image[i * width + j + 2] / 255.0f),
						(gtext.image[i * width + j + 3] / 255.0f)
						});
				}//for
			}//for
		}//emissive
		else if (gtext.name.find(metallicRoughnessString) != std::string::npos) {
			texturePresenceMask |= TEXTURE_METALLICROUGHNESS;//likely unused
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width * 4; j += 4) {
					metallicRoughness.push_back({
						(gtext.image[i * width + j + 0] / 255.0f),
						(gtext.image[i * width + j + 1] / 255.0f),
						(gtext.image[i * width + j + 2] / 255.0f),
						(gtext.image[i * width + j + 3] / 255.0f)
						});
				}//for
			}//for
		}//metallic roughness
		else if (gtext.name.find(normalString) != std::string::npos) {
			texturePresenceMask |= TEXTURE_NORMAL;
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width * 4; j += 4) {
					normal.push_back({
						(float)((gtext.image[i * width + j + 0] / 255.0) * 2.0 - 1.0),
						(float)((gtext.image[i * width + j + 1] / 255.0) * 2.0 - 1.0),
						(float)((gtext.image[i * width + j + 2] / 255.0) * 2.0 - 1.0),
						(float)((gtext.image[i * width + j + 3] / 255.0) * 2.0 - 1.0)
						});
				}//for
			}//for
		}//normal

	}//for each input texture

	return texturePresenceMask;

}//createFromGltfVector


void Texture::fillIntoF4Array(f4vec* dst) {
	memcpy(&dst[0 * width * height], baseColor.data(), width * height * sizeof(f4vec));
	memcpy(&dst[1 * width * height], emissive.data(), width * height * sizeof(f4vec));
	memcpy(&dst[2 * width * height], metallicRoughness.data(), width * height * sizeof(f4vec));
	memcpy(&dst[3 * width * height], normal.data(), width * height * sizeof(f4vec));
}

cudaArray* Texture::putOntoDevice(int textureIndex) {
	cudaChannelFormatDesc f4 = cudaCreateChannelDesc<float4>();
	cudaExtent extents = make_cudaExtent(width, height, 4);
	cudaMalloc3DArray(&cu_3darray, &f4, extents, cudaArrayLayered);
	cudaError_t err = cudaGetLastError();
	

	f4vec* h_data = (f4vec*)malloc(width * height * 4 * sizeof(f4vec));
	fillIntoF4Array(h_data);

	cudaMemcpy3DParms myparms = { 0 };
	myparms.srcPos = make_cudaPos(0, 0, 0);
	myparms.dstPos = make_cudaPos(0, 0, 0);
	myparms.srcPtr = make_cudaPitchedPtr(h_data, width * sizeof(f4vec), width, height);
	myparms.dstArray = cu_3darray;
	myparms.extent = extents;
	myparms.kind = cudaMemcpyHostToDevice;
	cudaMemcpy3D(&myparms);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error on the 3d memcpy! Err %d\n", err);
		exit(-1);
	}
	//check cuda error

	cudaResourceDesc    texRes;
	memset(&texRes, 0, sizeof(cudaResourceDesc));
	texRes.resType = cudaResourceTypeArray;
	texRes.res.array.array = cu_3darray;
	cudaTextureDesc texDescr;
	memset(&texDescr, 0, sizeof(cudaTextureDesc));
	texDescr.normalizedCoords = false;
	texDescr.filterMode = cudaFilterModeLinear;
	texDescr.addressMode[0] = cudaAddressModeClamp;
	texDescr.addressMode[1] = cudaAddressModeClamp;
	texDescr.readMode = cudaReadModeElementType;

	cudaCreateTextureObject(&texObjects[textureIndex], &texRes, &texDescr, NULL);
	err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Error on the creating texture objects! Err %d\n", err);
		exit(-1);
	}
	


	free(h_data);//no need to keep it locally anymore

	return cu_3darray;
}//putOntoDevice

void Texture::freeFromDevice() {
	cudaFreeArray(cu_3darray);
}//freeFromDevice