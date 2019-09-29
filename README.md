CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Disha Jindal: [Linkedin](https://www.linkedin.com/in/disha-jindal/)
* Tested on: Windows 10 Education, Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, NVIDIA Quadro P1000 @ 4GB (Moore 100B Lab)
## Path Tracer
## Overview
This is an implementation of CUDA-based path tracer capable of rendering globally-illuminated images very quickly. Path tracing is a computer graphics Monte Carlo method of rendering images so that we can achieve good results with tracing a finite number out of the infinite space of rays.

### Contents
* `scenes/` Example scene description files
* `img/` Renders of example scene description files
* `external/` Includes and static libraries for 3rd party libraries
* `src/` C++/CUDA source files
  - `main.cpp` : Setup and keyboard control logic
  - `pathtrace.cu` : Driver class which takes care of casting rays into the scene, testing for intersections, shading, graphics and performance optimizations and terminating a ray either after bouncing 8 times or reaching an emissive source
  - `interactions.cu` : Simulates coloring and scattering of reflective, diffusing and refractive surfaces
  - `intersections.cu`: Handles box, sphere, and mesh intersections

### Controls
 * Esc to save an image and exit.
 * S to save an image. Watch the console for the output filename.
 * Space to re-center the camera at the original scene lookAt point
 * left mouse button to rotate the camera
 * right mouse button on the vertical axis to zoom in/out
 * middle mouse button to move the LOOKAT point in the scene's X/Z plane

## Features Implemented
 * **Graphics**
   - [x] Shaders
      * Ideal Diffusion
      * Perfect Reflection
      * Refraction with fresnel effects [1/2 Additional Feature]
   - [x] Antialiasing [1/2 Additional Feature]
   - [x] Motion Blur [2/2 Additional Feature]
   - [x] 3D Object Mesh Loading and Renderning [1/2 Extra Credit]
 * **Optimizations**
   - [x] Work-efficient shared memory based Stream Compaction [2/2 Extra Credit]
   - [x] Contiguous rays by material type
   - [x] Cache First Bounce

### Ideal Diffusion
A ray after striking with a material is either reflected, refracted or diffused depending upon the material properties of the object. Diffusion is implemented using Bidirectional Scattering Distribution Function. 

### Perfect Reflection
In case of perfectly reflective surface, the new ray is calculated using `glm::reflect` function.

### Refraction with fresnel effects
Refraction is calculated using Snell's law and I have used `glm::refract` function to do this. But since most materials are not perfectly sepcular, have implmented fresnel effects using **Schlick's approximation**. Fresnel equations give the proportion of reflected and refracted light and then a random number from 0 to 1 is calculated to choose between specular reflection and refraction.

### Antialiasing
Antialiasing is a technique to diminish the jaggies/stairstep-like lines and smoothen them. This is implemented using a very simple trick that is by jittering the pixel's location. The idea is to subdivide the pixel into subpixels and choose a random supixel each time rather than always looking at the center to of the pixel. Accumulating the effect across multiple iterations, the intensity value of the pixel is the average of all these samples and creates a more continuos effect.

### Motion Blur
Motion blur is another technique which leverages this averaging effect of this implementation. To implement this, the object is moved slighlty between each iteration and the averaging of such multiple shots creats the effect of motion.

### 3D Object Modeling
Loading 3D models (Reference: https://free3d.com/) using [tinyObj](http://syoyo.github.io/tinyobjloader/) and then checking triangle intersection using `glm::intersectRayTriangle`.

## Optimizations
### Stream Compaction
A lot of rays die after a few iterations by either merging into light or the ones which do not intersect with any object. So, we can use stream compaction to limit the number of rays we are tracing and the number of threads launched at each iteration. I am using my Work-efficient stream compaction implementation across multiple blocks which uses shared memory for performance. 
#### Performance impact of stream compaction
#### Number of live/unterminated rays at each iteration

### Contiguous rays by material type
The shader implementation depends on the material with which the ray has intersected. So, If one warp has rays intersecting with different materials, it would lead to warp divergence and only the threads with one material could run at one time making it sequential in the number of materials. We can avoid this performance bottleneck by sorting the rays according to the material they are intersecting with so that rays interacting with the same material are contiguous in memory before shading and warp divergence is reduced. 

### Cache First Bounce
One first step of generating rays and finding itersection for the first bounce is same across all iterations with an exception while we are using anti aliasing. So, we could to an optimization by saving the first after the first iteration and reuse it rather than re doing it every time.

## Bloopers