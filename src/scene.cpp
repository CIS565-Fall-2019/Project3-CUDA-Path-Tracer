#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"

//#define OBJ1
//#define OBJ2

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
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

#ifdef OBJ1

	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	tinyobj::attrib_t attribs;
	std::string warning;
	std::string error;
	std::string objfile = "../objs/dragon.obj";
	bool objLoad = tinyobj::LoadObj(&attribs, &shapes, &materials, &warning, &error, objfile.c_str());
	if (!warning.empty()) {
		std::cout << "WARN: " << warning << std::endl;
	}
	if (!error.empty()) {
		std::cout << "ERR: " << error << std::endl;
	}

	if (!objLoad) {
		std::cout << "failed to load obj";
		return;
	}

	Geom cubeMesh;
	cubeMesh.materialid = 4;
	cubeMesh.type = MESH;
	cubeMesh.rotation = glm::vec3(30, -10, -15);
	cubeMesh.scale = glm::vec3(3, 3, 3);
	cubeMesh.translation = glm::vec3(0, 4, 0);

	cubeMesh.transform = utilityCore::buildTransformationMatrix(
		cubeMesh.translation, cubeMesh.rotation, cubeMesh.scale);
	cubeMesh.inverseTransform = glm::inverse(cubeMesh.transform);
	cubeMesh.invTranspose = glm::inverseTranspose(cubeMesh.transform);

	cubeMesh.minXYZ = glm::vec3(FLT_MAX);
	cubeMesh.maxXYZ = glm::vec3(-FLT_MAX);




	// Get out the vertices and normals from attribs
	std::vector<float> &positions = attribs.vertices;
	std::vector<float> &normals = attribs.normals;

	// Iterate over every shape in the obj
	for (tinyobj::shape_t shape : shapes) {
		// Get the indices of the points in each shape
		std::vector<tinyobj::index_t> &currIndices = shape.mesh.indices;
		// Make sure number of indices is a multiple of 3 for triangulation
		if (currIndices.size() % 3 != 0) {
			std::cout << "not triangles" << std::endl;
			return;
		}
		cubeMesh.lastTriangle += currIndices.size() / 3;


		// Go over every triangle and add the triangle
		for (int i = 0; i < currIndices.size(); i += 3) {

			tinyobj::index_t indexP1 = currIndices.at(i);
			tinyobj::index_t indexP2 = currIndices.at(i + 1);
			tinyobj::index_t indexP3 = currIndices.at(i + 2);

			Triangle currTri;

			currTri.p1 = glm::vec3(positions.at(3 * indexP1.vertex_index), 
								   positions.at(3 * indexP1.vertex_index + 1),
								   positions.at(3 * indexP1.vertex_index + 2));
			currTri.p2 = glm::vec3(positions.at(3 * indexP2.vertex_index), 
				                   positions.at(3 * indexP2.vertex_index + 1), 
				                   positions.at(3 * indexP2.vertex_index + 2));
			currTri.p3 = glm::vec3(positions.at(3 * indexP3.vertex_index), 
				                   positions.at(3 * indexP3.vertex_index + 1), 
				                   positions.at(3 * indexP3.vertex_index + 2));

			currTri.n2 = glm::vec3(normals.at(3 * indexP2.normal_index), 
                                   normals.at(3 * indexP2.normal_index + 1), 
				                   normals.at(3 * indexP2.normal_index + 2));
			currTri.n3 = glm::vec3(normals.at(3 * indexP3.normal_index), 
                                   normals.at(3 * indexP3.normal_index + 1), 
				                   normals.at(3 * indexP3.normal_index + 2));
			currTri.n1 = glm::vec3(normals.at(3 * indexP1.normal_index), 
							       normals.at(3 * indexP1.normal_index + 1), 
								   normals.at(3 * indexP1.normal_index + 2));

			cubeMesh.minXYZ.x = min(currTri.p1.x, 
								min(currTri.p2.x, 
								min(currTri.p3.x, 
									cubeMesh.minXYZ.x)));

			cubeMesh.minXYZ.y = min(currTri.p1.y,
								min(currTri.p2.y,
								min(currTri.p3.y,
							        cubeMesh.minXYZ.y)));

			cubeMesh.minXYZ.z = min(currTri.p1.z,
								min(currTri.p2.z,
								min(currTri.p3.z,
									cubeMesh.minXYZ.z)));

			cubeMesh.maxXYZ.x = max(currTri.p1.x, 
								max(currTri.p2.x, 
								max(currTri.p3.x, 
									cubeMesh.maxXYZ.x)));

			cubeMesh.maxXYZ.y = max(currTri.p1.y,
								max(currTri.p2.y,
								max(currTri.p3.y,
							        cubeMesh.maxXYZ.y)));

			cubeMesh.maxXYZ.z = max(currTri.p1.z,
				max(currTri.p2.z,
					max(currTri.p3.z,
						cubeMesh.maxXYZ.z)));


		

			this->triangles.push_back(currTri);


		}
	}
	std::cout << this->triangles.size();
	cubeMesh.lastTriangle = this->triangles.size();
	this->geoms.push_back(cubeMesh);

#endif // OBJ1

#ifdef OBJ2

	std::vector<tinyobj::shape_t> shapescello;
	std::vector<tinyobj::material_t> materialscello;
	tinyobj::attrib_t attribscello;
	std::string warningcello;
	std::string errorcello;
	std::string objfilecello = "../objs/cello.obj";
	bool objloadcello = tinyobj::loadobj(&attribscello, &shapescello, &materialscello, &warningcello, &errorcello, objfilecello.c_str());
	if (!warningcello.empty()) {
		std::cout << "warn: " << warningcello << std::endl;
	}
	if (!errorcello.empty()) {
		std::cout << "err: " << errorcello << std::endl;
	}

	if (!objloadcello) {
		std::cout << "failed to load obj";
		return;
	}


	geom cellomesh;
	cellomesh.materialid = 4;
	cellomesh.type = mesh;
	cellomesh.rotation = glm::vec3(-20, -30, -20);
	cellomesh.scale = glm::vec3(2, 2, 2);
	cellomesh.translation = glm::vec3(1.4, 1.3, 6.1);

	cellomesh.transform = utilitycore::buildtransformationmatrix(
		cellomesh.translation, cellomesh.rotation, cellomesh.scale);
	cellomesh.inversetransform = glm::inverse(cellomesh.transform);
	cellomesh.invtranspose = glm::inversetranspose(cellomesh.transform);

	cellomesh.minxyz = glm::vec3(flt_max);
	cellomesh.maxxyz = glm::vec3(-flt_max);




	// get out the vertices and normals from attribs
	std::vector<float> &positionscello = attribscello.vertices;
	std::vector<float> &normalscello = attribscello.normals;
	std::cout << positionscello.size() << std::endl;
	cellomesh.firsttriangle = cubemesh.lasttriangle;
	cellomesh.lasttriangle = cellomesh.firsttriangle;

	// iterate over every shape in the obj
	for (tinyobj::shape_t shape : shapescello) {
		// get the indices of the points in each shape
		std::vector<tinyobj::index_t> &currindices = shape.mesh.indices;
		// make sure number of indices is a multiple of 3 for triangulation
		if (currindices.size() % 3 != 0) {
			std::cout << "not triangles" << std::endl;
			return;
		}
		cellomesh.lasttriangle += currindices.size() / 3;


		// go over every triangle and add the triangle
		for (int i = 0; i < currindices.size(); i += 3) {
			tinyobj::index_t indexp1 = currindices.at(i);
			tinyobj::index_t indexp2 = currindices.at(i + 1);
			tinyobj::index_t indexp3 = currindices.at(i + 2);

			triangle currtri;

			currtri.p1 = glm::vec3(positionscello.at(3 * indexp1.vertex_index),
				positionscello.at(3 * indexp1.vertex_index + 1),
				positionscello.at(3 * indexp1.vertex_index + 2));
			currtri.p2 = glm::vec3(positionscello.at(3 * indexp2.vertex_index),
				positionscello.at(3 * indexp2.vertex_index + 1),
				positionscello.at(3 * indexp2.vertex_index + 2));
			currtri.p3 = glm::vec3(positionscello.at(3 * indexp3.vertex_index),
				positionscello.at(3 * indexp3.vertex_index + 1),
				positionscello.at(3 * indexp3.vertex_index + 2));

			currtri.n2 = glm::vec3(normalscello.at(3 * indexp2.normal_index),
				normalscello.at(3 * indexp2.normal_index + 1),
				normalscello.at(3 * indexp2.normal_index + 2));
			currtri.n3 = glm::vec3(normalscello.at(3 * indexp3.normal_index),
				normalscello.at(3 * indexp3.normal_index + 1),
				normalscello.at(3 * indexp3.normal_index + 2));
			currtri.n1 = glm::vec3(normalscello.at(3 * indexp1.normal_index),
				normalscello.at(3 * indexp1.normal_index + 1),
				normalscello.at(3 * indexp1.normal_index + 2));

			cellomesh.minxyz.x = min(currtri.p1.x,
				min(currtri.p2.x,
					min(currtri.p3.x,
						cellomesh.minxyz.x)));

			cellomesh.minxyz.y = min(currtri.p1.y,
				min(currtri.p2.y,
					min(currtri.p3.y,
						cellomesh.minxyz.y)));

			cellomesh.minxyz.z = min(currtri.p1.z,
				min(currtri.p2.z,
					min(currtri.p3.z,
						cellomesh.minxyz.z)));

			cellomesh.maxxyz.x = max(currtri.p1.x,
				max(currtri.p2.x,
					max(currtri.p3.x,
						cellomesh.maxxyz.x)));

			cellomesh.maxxyz.y = max(currtri.p1.y,
				max(currtri.p2.y,
					max(currtri.p3.y,
						cellomesh.maxxyz.y)));

			cellomesh.maxxyz.z = max(currtri.p1.z,
				max(currtri.p2.z,
					max(currtri.p3.z,
						cellomesh.maxxyz.z)));




			this->triangles.push_back(currtri);

		}
	}
	cellomesh.lasttriangle = this->triangles.size();
	this->geoms.push_back(cellomesh);
#endif // OBJ2

}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        return 1;
    }
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
