#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "glslUtility.hpp"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <OpenImageDenoise/oidn.hpp>
#include <array>

#include "sceneStructs.h"
#include "image.h"
#include "pathtrace.h"
#include "utilities.h"
#include "scene.h"

using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------


namespace oidn {

	class ImageBuffer
	{
	private:
		std::vector<float> data;
		int width;
		int height;
		int channels;

	public:
		ImageBuffer()
			: width(0),
			height(0),
			channels(0) {}

		ImageBuffer(int width, int height, int channels)
			: data(width * height * channels),
			width(width),
			height(height),
			channels(channels) {}

		operator bool() const
		{
			return data.data() != nullptr;
		}

		const float& operator [](size_t i) const { return data[i]; }
		float& operator [](size_t i) { return data[i]; }

		int getWidth() const { return width; }
		int getHeight() const { return height; }
		std::array<int, 2> getSize() const { return { width, height }; }
		int getChannels() const { return channels; }

		const float* getData() const { return data.data(); }
		float* getData() { return data.data(); }
		int getDataSize() { return int(data.size()); }
	};

} // namespace oidn

extern Scene* scene;
extern int iteration;

extern int width;
extern int height;

void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
