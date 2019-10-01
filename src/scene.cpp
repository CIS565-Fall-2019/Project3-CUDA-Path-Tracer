#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tinyobj/tiny_obj_loader.h"
#include <iostream>

Scene::Scene(string filename) {
	triCount = 0;
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
            } else if (strcmp(line.c_str(), "mesh") == 0) {
				string line2;
				utilityCore::safeGetline(fp_in, line2);
				newGeom.firstTriangle = triCount;
				if (!line2.empty() && fp_in.good()) {
					std::vector<tinyobj::shape_t> shapes;
					std::vector<tinyobj::material_t> materials;
					string output = tinyobj::LoadObj(shapes, materials, line2.c_str());
					if (output != "") cout << output << endl;

					//Read the information from the vector of shape_ts
					for (unsigned int i = 0; i < shapes.size(); i++)
					{
						std::vector<float> &positions = shapes[i].mesh.positions;
						std::vector<float> &normals = shapes[i].mesh.normals;
						std::vector<float> &uvs = shapes[i].mesh.texcoords;
						std::vector<unsigned int> &indices = shapes[i].mesh.indices;
						for (unsigned int j = 0; j < indices.size(); j += 3)
						{
							glm::vec3 p1(positions[indices[j] * 3], positions[indices[j] * 3 + 1], positions[indices[j] * 3 + 2]);
							glm::vec3 p2(positions[indices[j + 1] * 3], positions[indices[j + 1] * 3 + 1], positions[indices[j + 1] * 3 + 2]);
							glm::vec3 p3(positions[indices[j + 2] * 3], positions[indices[j + 2] * 3 + 1], positions[indices[j + 2] * 3 + 2]);

							for (int k = 0; k < 3; k++) {
								if (newGeom.bottomCornerBound[k] > p1[k]) newGeom.bottomCornerBound[k] = p1[k];
								if (newGeom.topCornerBound[k] < p1[k]) newGeom.topCornerBound[k] = p1[k];

								if (newGeom.bottomCornerBound[k] > p2[k]) newGeom.bottomCornerBound[k] = p2[k];
								if (newGeom.topCornerBound[k] < p2[k]) newGeom.topCornerBound[k] = p2[k];

								if (newGeom.bottomCornerBound[k] > p3[k]) newGeom.bottomCornerBound[k] = p3[k];
								if (newGeom.topCornerBound[k] < p3[k]) newGeom.topCornerBound[k] = p3[k];
							}

							Triangle t = Triangle();
							t.p1 = p1;
							t.p2 = p2;
							t.p3 = p3;
							if (normals.size() > 0)
							{
								glm::vec3 n1(normals[indices[j] * 3], normals[indices[j] * 3 + 1], normals[indices[j] * 3 + 2]);
								glm::vec3 n2(normals[indices[j + 1] * 3], normals[indices[j + 1] * 3 + 1], normals[indices[j + 1] * 3 + 2]);
								glm::vec3 n3(normals[indices[j + 2] * 3], normals[indices[j + 2] * 3 + 1], normals[indices[j + 2] * 3 + 2]);
								t.n1 = n1;
								t.n2 = n2;
								t.n3 = n3;
							}
							if (uvs.size() > 0)
							{
								glm::vec2 t1(uvs[indices[j] * 2], uvs[indices[j] * 2 + 1]);
								glm::vec2 t2(uvs[indices[j + 1] * 2], uvs[indices[j + 1] * 2 + 1]);
								glm::vec2 t3(uvs[indices[j + 2] * 2], uvs[indices[j + 2] * 2 + 1]);
								t.uv1 = t1;
								t.uv2 = t2;
								t.uv3 = t3;
							}
							
							triCount++;
							tris.push_back(t);
						}
					}
				}
				cout << "Creating new mesh..." << endl;
				newGeom.type = MESH;
				newGeom.lastTriangle = triCount;
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
	camera.focusDist = 10.5f;
	camera.lensRadius = 0.f;

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
		} else if (strcmp(tokens[0].c_str(), "LENSRADIUS") == 0) {
			camera.lensRadius = atof(tokens[1].c_str());
		} else if (strcmp(tokens[0].c_str(), "FOCUSDIST") == 0) {
			camera.focusDist = atof(tokens[1].c_str());
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
		newMaterial.specular.strength = 0;
		newMaterial.hasDiffuseMap = 0;
		newMaterial.hasSpecExpMap = 0;
		newMaterial.hasSpecStrMap = 0;

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

		string line;
		utilityCore::safeGetline(fp_in, line);
		while (!line.empty() && fp_in.good()) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "SPECSTREN") == 0) {
				newMaterial.specular.strength = atof(tokens[1].c_str());
			} else if (strcmp(tokens[0].c_str(), "RGBMAP") == 0) {
				newMaterial.hasDiffuseMap = 1;
				string filename = tokens[1];
				FILE * image = std::fopen(filename.c_str(), "r");
				int label;
				int dimensions;
				fscanf(image, "%d", &label);
				fscanf(image, "%d", &dimensions);
				float *colors = new float[dimensions];
				for (int j = 0; j < dimensions; j++) {
					int color;
					fscanf(image, "%d", &color);
					colors[j] = color;
				}
			}

			utilityCore::safeGetline(fp_in, line);
		}

        materials.push_back(newMaterial);
        return 1;
    }
}
