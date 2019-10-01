#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "material.h"
#include "mesh.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename, bool gltf_flag);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;

	std::vector<Mesh<float>> meshes;
	std::vector<MyMaterial> mymaterials;
	std::vector<Texture> textures;

    RenderState state;

	bool Scene::loadGLTF(const std::string &filename, float scale,
		std::vector<Mesh<float>> &meshes,
		std::vector<MyMaterial> &mymaterials,
		std::vector<Texture> &textures);
};
