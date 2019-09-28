#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/normal.hpp>


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
			}
			else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
				loadGeom(tokens[1]);
				cout << " " << endl;
			}
			else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
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
	}
	else {
		cout << "Loading Geom " << id << "..." << endl;
		Geom newGeom;
		string line;
		string meshFile = "";

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
				cout << "Creating new cube..." << endl;
				newGeom.type = TRIANGLE;
			}
			else if (strcmp(line.c_str(), "mesh") == 0) {
				cout << "Creating new cube..." << endl;
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
			else if (strcmp(tokens[0].c_str(), "VELO") == 0) {
				newGeom.velocity = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
			}
			else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
				meshFile = tokens[1];
			}

			utilityCore::safeGetline(fp_in, line);
		}

		newGeom.transform = utilityCore::buildTransformationMatrix(newGeom.translation, newGeom.rotation, newGeom.scale);
		newGeom.inverseTransform = glm::inverse(newGeom.transform);
		newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

		if (newGeom.type == MESH) {
			assert(meshFile != "", "Missing Mesh file argument for object id " + objectId);
			
			// Read from the gltf file.
			std::vector<Geom> meshTriangles = readGltfFile(newGeom, meshFile);
			geoms.insert(geoms.end(), meshTriangles.begin(), meshTriangles.end());
		}
		else {
			// If a primitive shape, take it onto the end.
			geoms.push_back(newGeom);
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
		}
		else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
			fovy = atof(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
			state.iterations = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
			state.traceDepth = atoi(tokens[1].c_str());
		}
		else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
			state.imageName = tokens[1];
		}
	}

	string line;
	utilityCore::safeGetline(fp_in, line);
	while (!line.empty() && fp_in.good()) {
		vector<string> tokens = utilityCore::tokenizeString(line);
		if (strcmp(tokens[0].c_str(), "EYE") == 0) {
			camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
			camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
		}
		else if (strcmp(tokens[0].c_str(), "UP") == 0) {
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

std::vector<Geom> Scene::readGltfFile(const Geom & parentGeom, const string & file)
{
	bool status = false;
	vector<Geom> out;

	vector<example::Mesh<float>> gltfMeshes;
	vector<example::Material>    gltfMaterials;
	vector<example::Texture>     gltfTextures;

	cout << "JOHN: Loading file..." << endl;
	if (!example::LoadGLTF(file, 1.0, &gltfMeshes, &gltfMaterials, &gltfTextures)) {
		cout << "Failed to load GLTF! File was " << file << endl;
		abort();
	}
	cout << "JOHN: Loaded!" << endl;

	// OK, we parsed the file thanks to gltf. Now to make sense of the mesh.
	for (const auto& mesh : gltfMeshes) {
		cout << "JOHN: Parsing Mesh" << endl;
		// Parse each mesh for all triangles.
		vector<Geom> g = gltfMeshToTriangles(parentGeom, mesh);
		out.insert(out.end(), g.begin(), g.end());
	}
	
	// TODO: Care about materials. For now, use the material from the config file.

	return out;
}

vector<Geom> Scene::gltfMeshToTriangles(const Geom & parentGeom, const example::Mesh<float>& mesh)
{
	// Get a mesh from gltf and turn it into a mesh of triangles we can parse
	vector<Geom> triangles;

	for (int i = 0; i < mesh.faces.size() / 3; i++) {
		Geom tri;
		tri.type = TRIANGLE;

		int v1idx = i * 3 + 0;
		int v2idx = i * 3 + 1;
		int v3idx = i * 3 + 2;

		// Convert our idexes into faces into vertex indicies
		v1idx = mesh.faces[v1idx];
		v2idx = mesh.faces[v2idx];
		v3idx = mesh.faces[v3idx];

		// Now get the real vertex info
		tri.v1 = glm::vec3(
			mesh.vertices[3 * v1idx + 0],
			mesh.vertices[3 * v1idx + 1],
			mesh.vertices[3 * v1idx + 2]
		);

		tri.v2 = glm::vec3(
			mesh.vertices[3 * v2idx + 0],
			mesh.vertices[3 * v2idx + 1],
			mesh.vertices[3 * v2idx + 2]
		);

		tri.v3 = glm::vec3(
			mesh.vertices[3 * v3idx + 0],
			mesh.vertices[3 * v3idx + 1],
			mesh.vertices[3 * v3idx + 2]
		);

		tri.norm = glm::triangleNormal(tri.v1, tri.v2, tri.v3);

		// Inherit properties of parent
		tri.translation = parentGeom.translation;
		tri.transform = parentGeom.transform;
		tri.scale = parentGeom.scale;
		tri.rotation = parentGeom.rotation;
		tri.invTranspose = parentGeom.invTranspose;
		tri.inverseTransform = parentGeom.inverseTransform;
		tri.velocity = parentGeom.velocity;

		tri.materialid = parentGeom.materialid;

		triangles.push_back(tri);
	}

	return triangles;
}

int Scene::loadMaterial(string materialid) {
	int id = atoi(materialid.c_str());
	if (id != materials.size()) {
		cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
		return -1;
	}
	else {
		cout << "Loading Material " << id << "..." << endl;
		Material newMaterial;

		//load static properties
		for (int i = 0; i < 7; i++) {
			string line;
			utilityCore::safeGetline(fp_in, line);
			vector<string> tokens = utilityCore::tokenizeString(line);
			if (strcmp(tokens[0].c_str(), "RGB") == 0) {
				glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.color = color;
			}
			else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
				newMaterial.specular.exponent = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
				glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
				newMaterial.specular.color = specColor;
			}
			else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
				newMaterial.hasReflective = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
				newMaterial.hasRefractive = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
				newMaterial.indexOfRefraction = atof(tokens[1].c_str());
			}
			else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
				newMaterial.emittance = atof(tokens[1].c_str());
			}
		}
		materials.push_back(newMaterial);
		return 1;
	}
}
