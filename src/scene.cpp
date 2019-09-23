#include <iostream>
#include "scene.h"
#include "tiny_obj_loader.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/normal.hpp>

//#define SKIPFACES 4//skip all mesh faces save one out of this many

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
	triangles = Triangle_v();

    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            } else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
        }
    }

}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());

	cout << "Loading Geom " << id << "..." << endl;
	Geom newGeom;
	string line;

	//load object type
	utilityCore::safeGetline(fp_in, line);
	if (!line.empty() && fp_in.good()) {
		if (strcmp(line.c_str(), "sphere") == 0) {
			cout << "Creating new sphere..." << endl;
			newGeom.type = SPHERE;
		}
		else if (strcmp(line.c_str(), "cube") == 0) {
			cout << "Creating new cube..." << endl;
			newGeom.type = CUBE;
		}
		else if (strcmp(line.c_str(), "triangle") == 0) {
			cout << "Creating new triangle..." << endl;
			newGeom.type = TRIANGLE;
		}
		else if (strcmp(line.c_str(), "mesh") == 0) {
			cout << "Creating new mesh..." << endl;
			newGeom.type = MESH;
		}

	}

	//link material
	utilityCore::safeGetline(fp_in, line);
	if (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilityCore::tokenizeString(line);
		newGeom.materialid = atoi(tokens[1].c_str());
		cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
	}

	if (newGeom.type == SPHERE || newGeom.type == CUBE) {
		//load transformations
		utilityCore::safeGetline(fp_in, line);
		while (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);

			//load tranformations
			if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
				newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
				newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
				newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}

			utilityCore::safeGetline(fp_in, line);
		}
		newGeom.transform = utilityCore::buildTransformationMatrix(
			newGeom.translation, newGeom.rotation, newGeom.scale);
		newGeom.inverseTransform = glm::inverse(newGeom.transform);
		newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
	}//cube or sphere
	else if (newGeom.type == MESH) {
		string filename;
		utilityCore::safeGetline(fp_in, line);
		while (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);

			//load tranformations
			if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
				newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
				newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
				newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
				filename = tokens[1];
			}

			utilityCore::safeGetline(fp_in, line);
		}
		newGeom.transform = utilityCore::buildTransformationMatrix(
			newGeom.translation, newGeom.rotation, newGeom.scale);
		newGeom.inverseTransform = glm::inverse(newGeom.transform);
		newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

		Geom_v meshList = readFromMesh(filename, newGeom.materialid, newGeom.transform);
		geoms.insert(geoms.end(), meshList.begin(), meshList.end());
	}//mesh

	if (newGeom.type != MESH)
		geoms.push_back(newGeom);
	return 1;
    
}

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState &state = this->state;
    Camera &camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        } else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
							, 2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    } else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color( atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()) );
                newMaterial.color = color;
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

Geom_v Scene::readFromMesh(string filename, int materialid, gmat4 transform) {
//bool LoadObj(attrib_t * attrib, std::vector<shape_t> * shapes,
//	std::vector<material_t> * materials, std::string * warn,
//	std::string * err, const char* filename, const char* mtl_basedir,
//	bool trianglulate, bool default_vcols_fallback);
	fs::path destination = fs::path(filename);
	fs::path parent = destination.parent_path();


	std::string warn;
	std::string err;
	const char* meshfile = filename.c_str();
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, meshfile, parent.string().c_str());

	//printf("Return result: %d\n", ret);
	//printf("Warning: %s\n", warn.c_str());
	//printf("Error: %s\n", err.c_str());

	Geom_v retval = Geom_v();
	
	for (tinyobj::shape_t shape : shapes) {

		Geom newGeom = geomFromShape(shape, attrib, materials, materialid, transform);
		retval.push_back(newGeom);

	}//for each shape


	return retval;
}

Geom Scene::geomFromShape(tinyobj::shape_t shape, tinyobj::attrib_t attrib, 
							std::vector<tinyobj::material_t> materials, int materialid, 
							gmat4 transform) {
	int startingTriangle = triangles.size();
	Geom newGeom = Geom();
	newGeom.type = MESH;

	//make the triangles

	string name = shape.name;//maybe useful?
	tinyobj::mesh_t mesh = shape.mesh;
	vector<tinyobj::index_t> indices = mesh.indices;
	int totalIndices = indices.size() / 3;
	for (int i = 0; i < indices.size(); i += 3) {
#ifdef SKIPFACES
		if ((i / 3) % SKIPFACES != 0) continue;
#endif
		Triangle tri = triangleFromIndex(i / 3, indices, mesh.material_ids, attrib, materialid, transform);

		triangles.push_back(tri);

		if ((i / 3) % 100 == 0) {
			printf("\tProcessing triangle %d of %d\n", i / 3, totalIndices);
		}//if

	}//for each face

	int endingTriangle = triangles.size();

	//compute bounding box (could maybe parallelize this... maybe not worth it)
	float minX = INFINITY; float minY = INFINITY; float minZ = INFINITY;
	float maxX = -INFINITY; float maxY = -INFINITY; float maxZ = -INFINITY;
	for (int i = startingTriangle; i < endingTriangle; i++) {
		gvec3 vert0 = triangles[i].vert0;
		gvec3 vert1 = triangles[i].vert1;
		gvec3 vert2 = triangles[i].vert2;
		minX = vert0.x < minX ? vert0.x : minX; minX = vert1.x < minX ? vert1.x : minX; minX = vert2.x < minX ? vert2.x : minX;		
		minY = vert0.y < minY ? vert0.y : minY; minY = vert1.y < minY ? vert1.y : minY; minY = vert2.y < minY ? vert2.y : minY;
		minZ = vert0.z < minZ ? vert0.z : minZ; minZ = vert1.z < minZ ? vert1.z : minZ; minZ = vert2.z < minZ ? vert2.z : minZ;
		maxX = vert0.x > maxX ? vert0.x : maxX; maxX = vert1.x > maxX ? vert1.x : maxX; maxX = vert2.x > maxX ? vert2.x : maxX;
		maxY = vert0.y > maxY ? vert0.y : maxY; maxY = vert1.y > maxY ? vert1.y : maxY; maxY = vert2.y > maxY ? vert2.y : maxY;
		maxZ = vert0.z > maxZ ? vert0.z : maxZ; maxZ = vert1.z > maxZ ? vert1.z : maxZ; maxZ = vert2.z > maxZ ? vert2.z : maxZ;
	}//for

	float xWidth = maxX - minX;
	float yWidth = maxY - minY;
	float zWidth = maxZ - minZ;
	float xCenter = xWidth / 2 + minX;
	float yCenter = yWidth / 2 + minY;
	float zCenter = zWidth / 2 + minZ;
	newGeom.scale = gvec3(xWidth, yWidth, zWidth);
	newGeom.translation = gvec3(xCenter, yCenter, zCenter);
	newGeom.rotation = gvec3(0, 0, 0);

	newGeom.transform = utilityCore::buildTransformationMatrix(
		newGeom.translation, newGeom.rotation, newGeom.scale);
	newGeom.inverseTransform = glm::inverse(newGeom.transform);
	newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
	
	newGeom.triangleIndex = startingTriangle;
	newGeom.triangleCount = endingTriangle - startingTriangle;

	return newGeom;
}//geomFromShape

//TODO: connect better to materials
Triangle Scene::triangleFromIndex(int index, vector<tinyobj::index_t> indices, vector<int> material_ids,
						   tinyobj::attrib_t attrib, 
						   int defaultMaterialId, gmat4 transform) {
	Triangle retval;

	//COLLECT ALL THE DATA

	//note: treating index as the index for the triangle as a whole
	//as a result, indexing into 3 * index
	tinyobj::index_t index0 = indices[3 * index + 0];//index for our first vertex
	tinyobj::index_t index1 = indices[3 * index + 1];
	tinyobj::index_t index2 = indices[3 * index + 2];

	gvec3 vert0 = gvec3(attrib.vertices[3 * index0.vertex_index + 0],
						attrib.vertices[3 * index0.vertex_index + 1],
						attrib.vertices[3 * index0.vertex_index + 2]);
	gvec3 vert1 = gvec3(attrib.vertices[3 * index1.vertex_index + 0],
						attrib.vertices[3 * index1.vertex_index + 1],
						attrib.vertices[3 * index1.vertex_index + 2]);
	gvec3 vert2 = gvec3(attrib.vertices[3 * index2.vertex_index + 0],
						attrib.vertices[3 * index2.vertex_index + 1],
						attrib.vertices[3 * index2.vertex_index + 2]);

	gvec3 norm, norm0, norm1, norm2;
	if (index0.normal_index > 0) {

		norm0 = gvec3(attrib.normals[3 * index0.normal_index + 0],
					  attrib.normals[3 * index0.normal_index + 1],
					  attrib.normals[3 * index0.normal_index + 2]);
		norm1 = gvec3(attrib.normals[3 * index1.normal_index + 0],
					  attrib.normals[3 * index1.normal_index + 1],
					  attrib.normals[3 * index1.normal_index + 2]);
		norm2 = gvec3(attrib.normals[3 * index2.normal_index + 0],
					  attrib.normals[3 * index2.normal_index + 1],
					  attrib.normals[3 * index2.normal_index + 2]);

		norm = normalized(norm0 + norm1 + norm2);
	}//if we have normal indexes
	else {
		gvec3 edge0 = vert1 - vert0;
		gvec3 edge1 = vert2 - vert0;
		norm = normalized(CROSSP(edge0, edge1));
		norm0 = norm;
		norm1 = norm;
		norm2 = norm;
	}//else (hoping for clockwise winding)

	//HERE IS WHERE WE WOULD FUCK WITH MATERIALS
	int materialId = defaultMaterialId;
	if (material_ids.size() > index) {
		if (material_ids[index] > 0) {
			materialId = material_ids[index];//would need a variant mapping
		}
	}//if

	//at this point, normal vector IS A normal (might be 

	//TRANSFORM
	vert0 = gvec3(transform * gvec4(vert0, 1.0));
	vert1 = gvec3(transform * gvec4(vert1, 1.0));
	vert2 = gvec3(transform * gvec4(vert2, 1.0));

	gmat4 normTransform = glm::transpose(glm::inverse(transform));
	norm0 = normalized(gvec3(normTransform * gvec4(norm0, 1.0)));
	norm1 = normalized(gvec3(normTransform * gvec4(norm1, 1.0)));
	norm2 = normalized(gvec3(normTransform * gvec4(norm2, 1.0)));
	norm = normalized(gvec3(normTransform * gvec4(norm, 1.0)));

	//SET VALUES
	retval.materialid = materialId;
	retval.vert0 = vert0;
	retval.vert1 = vert1;
	retval.vert2 = vert2;
	retval.norm0 = norm0;
	retval.norm1 = norm1;
	retval.norm2 = norm2;
	//retval.normal = norm;//deprecated

	return retval;
}//triangleFromIndex

Material Scene::materialFromObj(tinyobj::material_t mat) {
	Material retval;
	retval.color = gvec3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
	retval.specular.color = gvec3(mat.specular[0], mat.specular[1], mat.specular[2]);
	retval.specular.exponent = mat.shininess;//guessing

	retval.indexOfRefraction = mat.ior;

	if (mat.emission[0] || mat.emission[1] || mat.emission[2]) {
		retval.color = gvec3(mat.emission[0], mat.emission[1], mat.emission[2]);
		normalize(&retval.color);
		retval.emittance = sqrtf(mat.emission[0] * mat.emission[0]
			+ mat.emission[1] * mat.emission[1]
			+ mat.emission[2] * mat.emission[2]);
	}//if a light


	return retval;
}