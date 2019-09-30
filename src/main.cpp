#include "main.h"
#include "preview.h"
#include <cstring>
#include<chrono>
#include<intrin.h>
#include <OpenImageDenoise/oidn.hpp>
#include <string>
#include <vector>
#include <array>

#define DENOISE 0
static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene *scene;
RenderState *renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
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

	ImageBuffer loadImage(const std::string& filename);
	void saveImage(const std::string& filename, const ImageBuffer& image);

}
int main(int argc, char** argv) {
    startTimeString = currentTimeString();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

    const char *sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    // GLFW main loop
    mainLoop();

    return 0;
}
float ReverseFloat(const float inFloat)
{
	float retVal;
	char *floatToConvert = (char*)& inFloat;
	char *returnFloat = (char*)& retVal;

	// swap the bytes into a temporary buffer
	returnFloat[0] = floatToConvert[3];
	returnFloat[1] = floatToConvert[2];
	returnFloat[2] = floatToConvert[1];
	returnFloat[3] = floatToConvert[0];

	return retVal;
}
void denoise() {
	oidn::ImageBuffer im(width, height, 3);
	float samples = iteration;
	image img(width, height);
	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index];
			im[((height - 1 - y)*width + x) * 3 + 0] = pix.x;
			im[((height - 1 - y)*width + x) * 3 + 1] = pix.y;
			im[((height - 1 - y)*width + x) * 3 + 2] = pix.z;
		}
	}
	oidn::ImageBuffer output(width, height, 3);
	oidn::DeviceRef device = oidn::newDevice();
	device.commit();
	oidn::FilterRef filter = device.newFilter("RT");
	filter.setImage("color", im.getData(), oidn::Format::Float3, width, height);
	filter.setImage("output", output.getData(), oidn::Format::Float3, width, height);
	filter.commit();
	filter.execute();
	// Write the pixels
	for (int h = 0; h < height; ++h){
		for (int w = 0; w < width; ++w){
			const float r = output[((height - 1 - h)*width + w) * 3 + 0];
			const float g = output[((height - 1 - h)*width + w) * 3 + 1];
			const float b = output[((height - 1 - h)*width + w) * 3 + 2];
		//	std::cout << r << " " << g << " " << b << std::endl;
			glm::vec3 pix(r, g, b);
			img.setPixel(width - 1 - w, h, glm::vec3(pix) / samples);
	}
		
}
	std::string filename = "denoise_output_2";
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();
	img.savePNG(filename);

}
void saveImage() {
    float samples = iteration;
    // output image file
    image img(width, height);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {
    if (camchanged) {
        iteration = 0;
        Camera &cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
      }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene);
    }

    if (iteration < renderState->iterations) {
        uchar4 *pbo_dptr = NULL;
        iteration++;
        cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

        // execute the kernel
        int frame = 0;
		auto start = std::chrono::high_resolution_clock::now();

        pathtrace(pbo_dptr, frame, iteration);
        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    } else {
		if(DENOISE)
			denoise();
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
      switch (key) {
      case GLFW_KEY_ESCAPE:
        saveImage();
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
      case GLFW_KEY_S:
        saveImage();
        break;
      case GLFW_KEY_SPACE:
        camchanged = true;
        renderState = &scene->state;
        Camera &cam = renderState->camera;
        cam.lookAt = ogLookAt;
        break;
      }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
  if (leftMousePressed) {
    // compute new camera parameters
    phi -= (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, PI));
    camchanged = true;
  }
  else if (rightMousePressed) {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
  }
  else if (middleMousePressed) {
    renderState = &scene->state;
    Camera &cam = renderState->camera;
    glm::vec3 forward = cam.view;
    forward.y = 0.0f;
    forward = glm::normalize(forward);
    glm::vec3 right = cam.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
    cam.lookAt += (float) (ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}
