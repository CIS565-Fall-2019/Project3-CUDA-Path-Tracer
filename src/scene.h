#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

//#include "tiny_gltf.h"
#include "gltf-loader.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();

	// GLTF Parsing Functions
	std::vector<Geom> readGltfFile(const Geom & meshGeom, const string & file);
	vector<Geom> gltfMeshToTriangles(const Geom & parentMesh, const example::Mesh<float> & mesh);

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
