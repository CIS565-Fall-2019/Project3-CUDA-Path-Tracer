#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "tiny_gltf.h"

Scene::Scene(string filename) : currTriCount(0) {
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

int Scene::loadMesh(string filename, Geom& geom) {
	cout << "Reading mesh from " << filename << " ..." << endl;
	tinygltf::Model model;
	tinygltf::TinyGLTF loader;
	string error;
	string warning;

	bool ret = loader.LoadASCIIFromFile(&model, &error, &warning, filename);

	if (!warning.empty()) {
		cout << "Warning: " << warning.c_str() << endl;
	}

	if (!error.empty()) {
		cout << "Error: " << error.c_str() << endl;
	}

	if (!ret) {
		cout << "Failed to parse glTF " <<  filename << endl;
		return 0;
	}

	for (int i = 0; i < model.meshes.size(); i++) {
		tinygltf::Mesh& mesh = model.meshes[i];
		for (int j = 0; j < mesh.primitives.size(); j++) {
			tinygltf::Primitive& prim = mesh.primitives[i];

			// positions
			const tinygltf::Accessor& posAccessor = model.accessors[prim.attributes["POSITION"]];
			const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
			const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];
			const float* positions = reinterpret_cast<const float*>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);

			// get per-component min and max pos for bounding box of geom
			geom.minPos = glm::vec3(posAccessor.minValues[0], posAccessor.minValues[1], posAccessor.minValues[2]);
			geom.maxPos = glm::vec3(posAccessor.maxValues[0], posAccessor.maxValues[1], posAccessor.maxValues[2]);

			// normals
			const tinygltf::Accessor& norAccessor = model.accessors[prim.attributes["NORMAL"]];
			const tinygltf::BufferView& norBufferView = model.bufferViews[norAccessor.bufferView];
			const tinygltf::Buffer& norBuffer = model.buffers[norBufferView.buffer];
			const float* normals = reinterpret_cast<const float*>(&norBuffer.data[norBufferView.byteOffset + norAccessor.byteOffset]);

			// uvs
			const tinygltf::Accessor& uvAccessor = model.accessors[prim.attributes["TEXCOORD_0"]];
			const tinygltf::BufferView& uvBufferView = model.bufferViews[uvAccessor.bufferView];
			const tinygltf::Buffer& uvBuffer = model.buffers[uvBufferView.buffer];
			const float* uvs = reinterpret_cast<const float*>(&uvBuffer.data[uvBufferView.byteOffset + uvAccessor.byteOffset]);
			
			// indices
			geom.trianglesStart = currTriCount;
			const tinygltf::Accessor& indicesAccessor = model.accessors[prim.indices];
			const tinygltf::BufferView& indicesBufferView = model.bufferViews[indicesAccessor.bufferView];
			const tinygltf::Buffer& indicesBuffer = model.buffers[indicesBufferView.buffer];

			// TODO: this is terrible
			switch (indicesAccessor.componentType) {
			case TINYGLTF_COMPONENT_TYPE_BYTE:
				break;
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
				break;
			case TINYGLTF_COMPONENT_TYPE_SHORT:
				break;
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
				break;
			case TINYGLTF_COMPONENT_TYPE_INT:
				break;
			case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
				break;
			case TINYGLTF_COMPONENT_TYPE_FLOAT:
				cout << "hey" << endl;
				break;
			}

			const unsigned short* indices = reinterpret_cast<const unsigned short*>(&indicesBuffer.data[indicesBufferView.byteOffset + indicesAccessor.byteOffset]);
			for (size_t i = 0; i < indicesAccessor.count; i+=3) {
				Triangle tri;
				int index0 = indices[i];
				int index1 = indices[i + 1];
				int index2 = indices[i + 2];

				// positions
				glm::vec3 pos0 = glm::vec3(positions[(index0 * 3) + 0], positions[(index0 * 3) + 1], positions[(index0 * 3) + 2]);
				glm::vec3 pos1 = glm::vec3(positions[(index1 * 3) + 0], positions[(index1 * 3) + 1], positions[(index1 * 3) + 2]);
				glm::vec3 pos2 = glm::vec3(positions[(index2 * 3) + 0], positions[(index2 * 3) + 1], positions[(index2 * 3) + 2]);
				tri.positions[0] = pos0;
				tri.positions[1] = pos1;
				tri.positions[2] = pos2;

				// normals
				tri.normal = glm::normalize(glm::cross(pos1 - pos0, pos2 - pos1));

				// uvs
				glm::vec2 uv1 = glm::vec2(positions[(index0 * 2) + 0], positions[(index0 * 2) + 1]);
				glm::vec2 uv2 = glm::vec2(positions[(index1 * 2) + 0], positions[(index1 * 2) + 1]);
				glm::vec2 uv3 = glm::vec2(positions[(index2 * 2) + 0], positions[(index2 * 2) + 1]);
				tri.uvs[0] = uv1;
				tri.uvs[1] = uv2;
				tri.uvs[2] = uv3;

				triangles.push_back(tri);
				currTriCount++;
		
			}
			geom.trianglesEnd = currTriCount;
				

		}
	}



	return 1;
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

        // load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            } else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
			} else if (strcmp(line.c_str(), "mesh") == 0) {
				cout << "Creating new mesh..." << endl;
				newGeom.type = MESH;
			}
        }

        // link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        // load transformations
		int count = 0;
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good() && count < 3) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            // load transformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            } else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

			count++;
            utilityCore::safeGetline(fp_in, line);
        }

		if (!line.empty() && fp_in.good() && newGeom.type == MESH) {
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "FILENAME") == 0) {
				string filename = tokens[1].c_str();
				loadMesh(filename, newGeom);
			}
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
		} else if (strcmp(tokens[0].c_str(), "FOCALDISTANCE") == 0) {
			camera.focalDistance = atoi(tokens[1].c_str());
		} else if (strcmp(tokens[0].c_str(), "LENSRADIUS") == 0) {
			camera.lensRadius = atoi(tokens[1].c_str());
		}

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

	camera.right = glm::normalize(glm::cross(camera.view, camera.up));
	camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x, 2 * yscaled / (float)camera.resolution.y);

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
