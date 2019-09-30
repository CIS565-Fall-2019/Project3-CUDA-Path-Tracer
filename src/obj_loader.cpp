#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"
#include <iostream>
#include "obj_loader.h"
using namespace tinyobj;
using namespace std;
void load_obj(string input, vector<Geom>* triangle) {
  	vector<shape_t> shapes;
	vector<material_t> materials;
    attrib_t attribute;
	string error;
	string warning;
	if (!tinyobj::LoadObj(&attribute, &shapes, &materials, &warning, &error, input.c_str())){
		cerr << "Obj loading failed";
		cerr << error << std::endl;
	}
     
	
	for (int i = 0; i < shapes.size(); i++) {
		int offset = 0;
		for (int j = 0; j < shapes[i].mesh.num_face_vertices.size(); j++) {
			int num_faces = shapes[i].mesh.num_face_vertices[j];

		    vector<glm::vec3> vertices, normals;
			if (num_faces != 3) {
				cerr << "ERror: need triagnle mesh" << endl;
				exit(1);
			}
			for (int k = 0; k < 3; k++) {
				index_t index1 = shapes[i].mesh.indices[offset + k];
				vertices.push_back(glm::vec3(attribute.vertices[3 * index1.vertex_index + 0],
					attribute.vertices[3 * index1.vertex_index + 1],
					attribute.vertices[3 * index1.vertex_index + 2]));
				normals.push_back(glm::vec3(attribute.normals[3 * index1.normal_index + 0],
					attribute.normals[3 * index1.normal_index + 1],
					attribute.normals[3 * index1.normal_index + 2]));
			}
			offset += 3;

			Geom t; 
			t.type = TRI;
			t.vertex[0] = vertices[0];
			t.vertex[1] = vertices[1];
			t.vertex[2] = vertices[2];
			t.normal[0] = normals[0];
			t.normal[1] = normals[1];
			t.normal[2] = normals[2];
			triangle->push_back(t);
		}
	}
}
