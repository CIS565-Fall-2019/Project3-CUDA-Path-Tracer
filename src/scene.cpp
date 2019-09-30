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

	//std::vector<string> meshfile;

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
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    } else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

		string filename;
		bool hasmesh = false;


        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(tokens[0].c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            } else if (strcmp(tokens[0].c_str(), "mesh") == 0) {//mesh
				cout << "Creating new mesh..." << endl;
				newGeom.type = MESH;
				filename = tokens[1].c_str();
				if (!hasmesh) {
					hasmesh = true;
				}
			}
        }

		//load the obj
		if (hasmesh) {
			LoadOBJ(filename);
		}

		if (newGeom.type != MESH) {
			newGeom.Tri_start_Idx = -1;
			newGeom.Tri_end_Idx = -1;
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

		//push back to the geometry array
        geoms.push_back(newGeom);
		//push back to the light array
		if (materials[newGeom.materialid].emittance > 0.0) {//is light
			lights.push_back(newGeom);
		}

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
			//maximum depth (number of times the path will bounce)
            state.traceDepth = atoi(tokens[1].c_str());
        } else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

	bool islens_cam = false;//juage whether lenscamera

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {//good-> is th file stream ok?
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        } else if (strcmp(tokens[0].c_str(), "FOCAL") == 0) {
			islens_cam = true;
			camera.focaldistance = atof(tokens[1].c_str());
		} else if (strcmp(tokens[0].c_str(), "LREDIUS") == 0) {
			camera.lenradius = atof(tokens[1].c_str());
		}
        utilityCore::safeGetline(fp_in, line);
    }

	//not lenscamera
	if (!islens_cam) {
		camera.focaldistance = -1.0;
		camera.lenradius = -1.0;
	}

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));//tan alpha
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;//tan beta
    float fovx = (atan(xscaled) * 180) / PI;//beta angle
    camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	//1/viewlength
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x
							, 2 * yscaled / (float)camera.resolution.y);
	//view is the camera.lookAt - camera.position
    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());// the 1st?,2nd? 3rd? material in the scene?
    if (id != materials.size()) {//material size is plus slowly when adding
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
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {//rgb color
				//atof change the string to the float
                glm::vec3 color(atof(tokens[1].c_str()), 
					atof(tokens[2].c_str()), 
					atof(tokens[3].c_str()) );
                newMaterial.color = color;//color is rgb color
            } else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {//specular
                newMaterial.specular.exponent = atof(tokens[1].c_str());//zhi shu
            } else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {//spec
                glm::vec3 specColor(atof(tokens[1].c_str()), //spec color
					atof(tokens[2].c_str()), 
					atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            } else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());//is reflect?
            } else if (strcmp(tokens[0].c_str(), "REFR") == 0) {//is refract?
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {//eta
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            } else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {//a number, if emit?
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

int t_count = 0;

void Scene::LoadOBJ(const string &filepath) {
	std::vector<tinyobj::shape_t> shapes; 
	std::vector<tinyobj::material_t> materials;
	const char* c_s = filepath.c_str();

	std::string errors = tinyobj::LoadObj(shapes, materials, c_s);
	std::cout << errors << std::endl;

	if (errors.size() == 0) {
		//Read the information from the vector of shape_ts
		for (unsigned int i = 0; i < shapes.size(); i++) {
			std::vector<float> &positions = shapes[i].mesh.positions;
			std::vector<float> &normals = shapes[i].mesh.normals;
			//std::vector<float> &uvs = shapes[i].mesh.texcoords;
			std::vector<unsigned int> &indices = shapes[i].mesh.indices;

			for (unsigned int j = 0; j < indices.size(); j += 3) {
				glm::vec3 p1(positions[indices[j] * 3], positions[indices[j] * 3 + 1], positions[indices[j] * 3 + 2]);
				glm::vec3 p2(positions[indices[j + 1] * 3], positions[indices[j + 1] * 3 + 1], positions[indices[j + 1] * 3 + 2]);
				glm::vec3 p3(positions[indices[j + 2] * 3], positions[indices[j + 2] * 3 + 1], positions[indices[j + 2] * 3 + 2]);

				Triangle t;
				t.index = t_count++;
				t.vertices[0] = p1;
				t.vertices[1] = p2;
				t.vertices[2] = p3;

				if (normals.size() > 0) {
					glm::vec3 n1(normals[indices[j] * 3], normals[indices[j] * 3 + 1], normals[indices[j] * 3 + 2]);
					glm::vec3 n2(normals[indices[j + 1] * 3], normals[indices[j + 1] * 3 + 1], normals[indices[j + 1] * 3 + 2]);
					glm::vec3 n3(normals[indices[j + 2] * 3], normals[indices[j + 2] * 3 + 1], normals[indices[j + 2] * 3 + 2]);
					t.normals[0] = n1;
					t.normals[1] = n2;
					t.normals[2] = n3;
				}
				//this->faces.append(t);
				tris.push_back(t);
			}
		}		
		std::cout << "" << std::endl;
		//TODO: .mtl file loading
	}
	else {
		//An error loading the OBJ occurred!
		std::cout << errors << std::endl;
	}
}

