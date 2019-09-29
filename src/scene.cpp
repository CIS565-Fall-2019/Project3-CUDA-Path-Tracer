#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_obj_loader.h"

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
//	std::vector<tinyobj::shape_t> shapes;
//	std::vector<tinyobj::material_t> materials;
//	tinyobj::attrib_t attribs;
//	std::string warning;
//	std::string error;
//	std::string objfile = "../objs/alpaca.obj";
//	bool objLoad = tinyobj::LoadObj(&attribs, &shapes, &materials, &warning, &error, objfile.c_str());
//	if (!warning.empty()) {
//		std::cout << "WARN: " << warning << std::endl;
//	}
//	if (!error.empty()) {
//		std::cout << "ERR: " << error << std::endl;
//	}
//
//	if (!objLoad) {
//		std::cout << "failed to load obj";
//		return;
//	}
//
//	Geom cubeMesh;
//	cubeMesh.materialid = 4;
//	cubeMesh.type = MESH;
//	cubeMesh.rotation = glm::vec3(0, 40, 0);
//	cubeMesh.scale = glm::vec3(4, 4, 4);
//	cubeMesh.translation = glm::vec3(0, 0, 2);
//
//	cubeMesh.transform = utilityCore::buildTransformationMatrix(
//		cubeMesh.translation, cubeMesh.rotation, cubeMesh.scale);
//	cubeMesh.inverseTransform = glm::inverse(cubeMesh.transform);
//	cubeMesh.invTranspose = glm::inverseTranspose(cubeMesh.transform);
//
//	cubeMesh.minXYZ = glm::vec3(FLT_MAX);
//	cubeMesh.maxXYZ = glm::vec3(-FLT_MAX);
//
//
//
//
//	// Get out the vertices and normals from attribs
//	std::vector<float> &positions = attribs.vertices;
//	std::vector<float> &normals = attribs.normals;
//	std::cout << positions.size() << std::endl;
//
//	// Iterate over every shape in the obj
//	for (tinyobj::shape_t shape : shapes) {
//		// Get the indices of the points in each shape
//		std::vector<tinyobj::index_t> &currIndices = shape.mesh.indices;
//		// Make sure number of indices is a multiple of 3 for triangulation
//		if (currIndices.size() % 3 != 0) {
//			std::cout << "not triangles" << std::endl;
//			return;
//		}
//		cubeMesh.lastTriangle += currIndices.size() / 3;
//
//
//		// Go over every triangle and add the triangle
//		for (int i = 0; i < currIndices.size(); i += 3) {
//			//std::cout << "TRIANGLE " << i << std::endl;
//
//			tinyobj::index_t indexP1 = currIndices.at(i);
//			tinyobj::index_t indexP2 = currIndices.at(i + 1);
//			tinyobj::index_t indexP3 = currIndices.at(i + 2);
//
//			/*std::cout << indexP1.vertex_index << std::endl;
//			std::cout << indexP2.vertex_index << std::endl;
//			std::cout << indexP3.vertex_index << std::endl;
//*/
//			Triangle currTri;
//
//			currTri.p1 = glm::vec3(positions.at(3 * indexP1.vertex_index), 
//								   positions.at(3 * indexP1.vertex_index + 1),
//								   positions.at(3 * indexP1.vertex_index + 2));
//			currTri.p2 = glm::vec3(positions.at(3 * indexP2.vertex_index), 
//				                   positions.at(3 * indexP2.vertex_index + 1), 
//				                   positions.at(3 * indexP2.vertex_index + 2));
//			currTri.p3 = glm::vec3(positions.at(3 * indexP3.vertex_index), 
//				                   positions.at(3 * indexP3.vertex_index + 1), 
//				                   positions.at(3 * indexP3.vertex_index + 2));
//
//			currTri.n2 = glm::vec3(normals.at(3 * indexP2.normal_index), 
//                                   normals.at(3 * indexP2.normal_index + 1), 
//				                   normals.at(3 * indexP2.normal_index + 2));
//			currTri.n3 = glm::vec3(normals.at(3 * indexP3.normal_index), 
//                                   normals.at(3 * indexP3.normal_index + 1), 
//				                   normals.at(3 * indexP3.normal_index + 2));
//			currTri.n1 = glm::vec3(normals.at(3 * indexP1.normal_index), 
//							       normals.at(3 * indexP1.normal_index + 1), 
//								   normals.at(3 * indexP1.normal_index + 2));
//
//			cubeMesh.minXYZ.x = min(currTri.p1.x, 
//								min(currTri.p2.x, 
//								min(currTri.p3.x, 
//									cubeMesh.minXYZ.x)));
//
//			cubeMesh.minXYZ.y = min(currTri.p1.y,
//								min(currTri.p2.y,
//								min(currTri.p3.y,
//							        cubeMesh.minXYZ.y)));
//
//			cubeMesh.minXYZ.z = min(currTri.p1.z,
//								min(currTri.p2.z,
//								min(currTri.p3.z,
//									cubeMesh.minXYZ.z)));
//
//			cubeMesh.maxXYZ.x = max(currTri.p1.x, 
//								max(currTri.p2.x, 
//								max(currTri.p3.x, 
//									cubeMesh.maxXYZ.x)));
//
//			cubeMesh.maxXYZ.y = max(currTri.p1.y,
//								max(currTri.p2.y,
//								max(currTri.p3.y,
//							        cubeMesh.maxXYZ.y)));
//
//			cubeMesh.maxXYZ.z = max(currTri.p1.z,
//				max(currTri.p2.z,
//					max(currTri.p3.z,
//						cubeMesh.maxXYZ.z)));
//
//
//		
//
//			this->triangles.push_back(currTri);
//			
//			/*std::cout << "(" << currTri.p1.x << ", " << currTri.p1.y << ", " << currTri.p1.z << ")" << std::endl;
//			std::cout << "(" << currTri.p2.x << ", " << currTri.p2.y << ", " << currTri.p2.z << ")" << std::endl;
//			std::cout << "(" << currTri.p3.x << ", " << currTri.p3.y << ", " << currTri.p3.z << ")" << std::endl;*/
//
//			/*std::cout << "(" << currTri.n1.x << ", " << currTri.n1.y << ", " << currTri.n1.z << ")" << std::endl;
//			std::cout << "(" << currTri.n2.x << ", " << currTri.n2.y << ", " << currTri.n2.z << ")" << std::endl;
//			std::cout << "(" << currTri.n3.x << ", " << currTri.n3.y << ", " << currTri.n3.z << ")" << std::endl*/;
//
//
//
//		}
//	}
//	cubeMesh.lastTriangle = this->triangles.size();
//	this->geoms.push_back(cubeMesh);

//	std::vector<tinyobj::shape_t> shapesCello;
//	std::vector<tinyobj::material_t> materialsCello;
//	tinyobj::attrib_t attribsCello;
//	std::string warningCello;
//	std::string errorCello;
//	std::string objfileCello = "../objs/cello.obj";
//	bool objLoadCello = tinyobj::LoadObj(&attribsCello, &shapesCello, &materialsCello, &warningCello, &errorCello, objfileCello.c_str());
//	if (!warningCello.empty()) {
//		std::cout << "WARN: " << warningCello << std::endl;
//	}
//	if (!errorCello.empty()) {
//		std::cout << "ERR: " << errorCello << std::endl;
//	}
//
//	if (!objLoadCello) {
//		std::cout << "failed to load obj";
//		return;
//	}
//
//
//	Geom celloMesh;
//	celloMesh.materialid = 4;
//	celloMesh.type = MESH;
//	celloMesh.rotation = glm::vec3(-20, -30, -20);
//	celloMesh.scale = glm::vec3(2, 2, 2);
//	celloMesh.translation = glm::vec3(1.4, 1.3, 6.1);
//
//	celloMesh.transform = utilityCore::buildTransformationMatrix(
//		celloMesh.translation, celloMesh.rotation, celloMesh.scale);
//	celloMesh.inverseTransform = glm::inverse(celloMesh.transform);
//	celloMesh.invTranspose = glm::inverseTranspose(celloMesh.transform);
//
//	celloMesh.minXYZ = glm::vec3(FLT_MAX);
//	celloMesh.maxXYZ = glm::vec3(-FLT_MAX);
//
//
//
//
//	// Get out the vertices and normals from attribs
//	std::vector<float> &positionsCello = attribsCello.vertices;
//	std::vector<float> &normalsCello = attribsCello.normals;
//	std::cout << positionsCello.size() << std::endl;
//	celloMesh.firstTriangle = cubeMesh.lastTriangle;
//	celloMesh.lastTriangle = celloMesh.firstTriangle;
//
//	// Iterate over every shape in the obj
//	for (tinyobj::shape_t shape : shapesCello) {
//		// Get the indices of the points in each shape
//		std::vector<tinyobj::index_t> &currIndices = shape.mesh.indices;
//		// Make sure number of indices is a multiple of 3 for triangulation
//		if (currIndices.size() % 3 != 0) {
//			std::cout << "not triangles" << std::endl;
//			return;
//		}
//		celloMesh.lastTriangle += currIndices.size() / 3;
//
//
//		// Go over every triangle and add the triangle
//		for (int i = 0; i < currIndices.size(); i += 3) {
//			//std::cout << "TRIANGLE " << i << std::endl;
//
//			tinyobj::index_t indexP1 = currIndices.at(i);
//			tinyobj::index_t indexP2 = currIndices.at(i + 1);
//			tinyobj::index_t indexP3 = currIndices.at(i + 2);
//
//			/*std::cout << indexP1.vertex_index << std::endl;
//			std::cout << indexP2.vertex_index << std::endl;
//			std::cout << indexP3.vertex_index << std::endl;
//*/
//			Triangle currTri;
//
//			currTri.p1 = glm::vec3(positionsCello.at(3 * indexP1.vertex_index),
//				positionsCello.at(3 * indexP1.vertex_index + 1),
//				positionsCello.at(3 * indexP1.vertex_index + 2));
//			currTri.p2 = glm::vec3(positionsCello.at(3 * indexP2.vertex_index),
//				positionsCello.at(3 * indexP2.vertex_index + 1),
//				positionsCello.at(3 * indexP2.vertex_index + 2));
//			currTri.p3 = glm::vec3(positionsCello.at(3 * indexP3.vertex_index),
//				positionsCello.at(3 * indexP3.vertex_index + 1),
//				positionsCello.at(3 * indexP3.vertex_index + 2));
//
//			currTri.n2 = glm::vec3(normalsCello.at(3 * indexP2.normal_index),
//				normalsCello.at(3 * indexP2.normal_index + 1),
//				normalsCello.at(3 * indexP2.normal_index + 2));
//			currTri.n3 = glm::vec3(normalsCello.at(3 * indexP3.normal_index),
//				normalsCello.at(3 * indexP3.normal_index + 1),
//				normalsCello.at(3 * indexP3.normal_index + 2));
//			currTri.n1 = glm::vec3(normalsCello.at(3 * indexP1.normal_index),
//				normalsCello.at(3 * indexP1.normal_index + 1),
//				normalsCello.at(3 * indexP1.normal_index + 2));
//
//			celloMesh.minXYZ.x = min(currTri.p1.x,
//				min(currTri.p2.x,
//					min(currTri.p3.x,
//						celloMesh.minXYZ.x)));
//
//			celloMesh.minXYZ.y = min(currTri.p1.y,
//				min(currTri.p2.y,
//					min(currTri.p3.y,
//						celloMesh.minXYZ.y)));
//
//			celloMesh.minXYZ.z = min(currTri.p1.z,
//				min(currTri.p2.z,
//					min(currTri.p3.z,
//						celloMesh.minXYZ.z)));
//
//			celloMesh.maxXYZ.x = max(currTri.p1.x,
//				max(currTri.p2.x,
//					max(currTri.p3.x,
//						celloMesh.maxXYZ.x)));
//
//			celloMesh.maxXYZ.y = max(currTri.p1.y,
//				max(currTri.p2.y,
//					max(currTri.p3.y,
//						celloMesh.maxXYZ.y)));
//
//			celloMesh.maxXYZ.z = max(currTri.p1.z,
//				max(currTri.p2.z,
//					max(currTri.p3.z,
//						celloMesh.maxXYZ.z)));
//
//
//
//
//			this->triangles.push_back(currTri);
//
//			/*std::cout << "(" << currTri.p1.x << ", " << currTri.p1.y << ", " << currTri.p1.z << ")" << std::endl;
//			std::cout << "(" << currTri.p2.x << ", " << currTri.p2.y << ", " << currTri.p2.z << ")" << std::endl;
//			std::cout << "(" << currTri.p3.x << ", " << currTri.p3.y << ", " << currTri.p3.z << ")" << std::endl;*/
//
//			/*std::cout << "(" << currTri.n1.x << ", " << currTri.n1.y << ", " << currTri.n1.z << ")" << std::endl;
//			std::cout << "(" << currTri.n2.x << ", " << currTri.n2.y << ", " << currTri.n2.z << ")" << std::endl;
//			std::cout << "(" << currTri.n3.x << ", " << currTri.n3.y << ", " << currTri.n3.z << ")" << std::endl*/;
//
//
//
//		}
//	}
//	celloMesh.lastTriangle = this->triangles.size();
//	this->geoms.push_back(celloMesh);
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
