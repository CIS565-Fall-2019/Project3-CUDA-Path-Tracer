#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
	void LoadOBJ(const string &filepath);
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
	std::vector<Geom> lights;

	std::vector<Triangle> tris;

#if TRI_2D_Array 
	std::vector<std::vector<Triangle>> mesh_tris;
#endif //  TRI_2D_Array 

    RenderState state;
};
