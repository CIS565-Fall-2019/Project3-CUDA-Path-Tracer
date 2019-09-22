#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include <filesystem>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_inverse.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_obj_loader.h"

using namespace std;
namespace fs = std::experimental::filesystem;

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
	Triangle_v triangles;
    Material_v materials;
    RenderState state;

	Geom_v readFromMesh(string filename, int materialid, gmat4 transform = MAT4I);//default to the identity matrix

	/**
	Creates a Material object from the read-in material
	*/
	Material materialFromObj(tinyobj::material_t mat);
	/**
	Constructs a Geom object with all the relevant bounding-box parameters 
	*/
	Geom Scene::geomFromShape(tinyobj::shape_t shape, tinyobj::attrib_t attrib, std::vector<tinyobj::material_t> materials);
	/**
	Pulls out the relevant information to make a triangle from the given face index
	*/
	Triangle triangleFromIndex(int index, vector<tinyobj::index_t> indices, vector<int> material_ids,
								tinyobj::attrib_t attrib,
								int defaultMaterialId, gmat4 transform);
};
