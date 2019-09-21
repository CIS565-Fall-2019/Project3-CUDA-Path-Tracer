#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_obj_loader.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    Geom_v geoms;
    Material_v materials;
    RenderState state;

	//TODO: just feed in the transformation matrix, so we can get rotation in as well
	Geom_v readFromMesh(string filename, int materialid, gmat4 transform = MAT4I);//default to the identity matrix
};
