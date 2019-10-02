#include "main.h"
#include "preview.h"
#include <cstring>

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

float oriheight;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    startTimeString = currentTimeString();

    if (argc < 2) {
        printf("Usage: %s SCENEFILE.txt\n", argv[0]);
        return 1;
    }

	const char *sceneFile = argv[1];

#if GLTF
	scene = new Scene(sceneFile, true);
#else
	scene = new Scene(sceneFile, false);
#endif


    // Set up camera stuff from loaded path tracer settings
    iteration = 0;

	renderState = &(scene->state);
	Camera &cam = renderState->camera;
#if GLTF
	cam.resolution.x = 1280;
	cam.resolution.y = 720;
	//Zelda
	//cam.lookAt = glm::vec3(0, 0, 6);// glm::vec3(scene->meshes.at(0).pivot_xform[3][0], scene->meshes.at(0).pivot_xform[3][1], scene->meshes.at(0).pivot_xform[3][2]);
	//cam.position = glm::vec3(4, -60, 20);
	//person
	cam.lookAt = glm::vec3(0, 0, 1);
	cam.position = glm::vec3(0.4, -2, 1.6);

	cam.view = cam.lookAt - cam.position;
	cam.view = glm::normalize(cam.view);
	cam.up = glm::vec3(0, 0, 1);
	cam.right = glm::cross(cam.view, cam.up);
	cam.up = glm::cross(cam.right, cam.view);
	cam.fov = glm::vec2(80, 45);
	cam.pixelLength = glm::vec2(1 / (float)cam.resolution.y, 1 / (float)cam.resolution.y);
	int arraylen = cam.resolution.x * cam.resolution.y;
	renderState->image.resize(arraylen);
	std::fill((renderState->image).begin(), (renderState->image).end(), glm::vec3());
	renderState->iterations = 5000;
	renderState->traceDepth = 2;
	renderState->imageName = "demo";
#endif
    width = cam.resolution.x;
    height = cam.resolution.y;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    phi = glm::atan(view.x / view.z);
	if(view.z > 0) {
		phi += PI;
	}
    theta = glm::acos(-view.y);
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

#if MOTIONBLUR
	oriheight = scene->geoms.at(6).transform[3][1];
#endif

    // Initialize CUDA and GL components
    init();

    // GLFW main loop
    mainLoop();

    return 0;
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
#if GLTF
		glm::vec3 u = glm::vec3(0, 0, 1);// glm::normalize(cam.up);
#else
		glm::vec3 u = glm::vec3(0, 1, 0);
#endif
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        //cam.position = cameraPosition;
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
#if MOTIONBLUR
		glm::mat4 &trans = scene->geoms.at(6).transform;
		trans[3][1] = oriheight - (iteration % 20) * 0.05f;
		scene->geoms.at(6).invTranspose = glm::transpose(glm::inverse(trans));
		scene->geoms.at(6).inverseTransform = glm::inverse(trans);
		resetGeoms();
#endif
        pathtrace(pbo_dptr, frame, iteration);

        // unmap buffer object
        cudaGLUnmapBufferObject(pbo);
    } else {
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
