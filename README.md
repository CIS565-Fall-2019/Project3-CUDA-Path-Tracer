CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Disha Jindal: [Linkedin](https://www.linkedin.com/in/disha-jindal/)
* Tested on: Windows 10 Education, Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, NVIDIA Quadro P1000 @ 4GB (Moore 100B Lab)
## Path Tracer
<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/scene3.png" width="600"/> </p>

<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/scene11.png" width="600"/> </p>

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
  - `intersections.cu` : Handles box, sphere, and mesh intersections
  - `utilities.h` : Contains some utility functions and following flags to togge the features:
     ```
      #define COMPACT_RAYS [0,1]
      #define CACHE_FIRST_BOUNCE [0,1]
      #define MATERIAL_BASED_SORT [0,1]
      #define ANTI_ALIASING [0,1]
      #define MOTION_BLUR [0,1]
     ```

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
<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/Diffusion.png" width="600"/> </p>

### Perfect Reflection
In case of perfectly reflective surface, the new ray is calculated using `glm::reflect` function.
<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/Reflection.png" width="600"/> </p>

### Refraction with fresnel effects
Refraction is calculated using Snell's law and I have used `glm::refract` function to do this. But since most materials are not perfectly sepcular, have implmented fresnel effects using **Schlick's approximation**. Fresnel equations give the proportion of reflected and refracted light and then a random number from 0 to 1 is calculated to choose between specular reflection and refraction.
<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/Refraction.png" width="600"/> </p>

### Antialiasing
Antialiasing is a technique to diminish the jaggies/stairstep-like lines and smoothen them. This is implemented using a very simple trick that is by jittering the pixel's location. The idea is to subdivide the pixel into subpixels and choose a random supixel each time rather than always looking at the center to of the pixel. Accumulating the effect across multiple iterations, the intensity value of the pixel is the average of all these samples and creates a more continuos effect.
<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/without_anti_z.png" width="300"/>  <img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/with_anti_z.png" width="312"/> </p>

### Motion Blur
Motion blur is another technique which leverages this averaging effect of this implementation. To implement this, the object is moved slighlty between each iteration and the averaging of such multiple shots creats the effect of motion.
<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/motionblur.png" width="600"/> </p>


### 3D Object Modeling
Loading 3D models (Reference: https://free3d.com/) using [tinyObj](http://syoyo.github.io/tinyobjloader/) and then checking triangle intersection using `glm::intersectRayTriangle`.

<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/droid_1.png" width="600"/> </p>
<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/3D_Android.png" width="600"/> </p>
<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/scene2.png" width="600"/> </p>

## Optimizations
### Stream Compaction
A lot of rays die after a few iterations by either merging into light or the ones which do not intersect with any object. So, we can use stream compaction to limit the number of rays we are tracing and the number of threads launched at each iteration. I am using my Work-efficient stream compaction implementation across multiple blocks which uses shared memory for performance. 

#### Performance impact of stream compaction
Following plot shows the average time per depth with and without stream compaction. Stream compaction took around 3.8 ms whereas it took 4s without it. These are the results with 8 bounces and so the performance would increase even further with more bounces and more complex scenes.

<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/Performance_SC.PNG" width="300"/> </p>

#### Number of live/unterminated rays at each iteration
Following plots shows the number of unterminated rays at each depth. Yellow bars correspond to an open scene and the red bars show corresponding closed scene with additional left and right walls. We can see that the numbe rof live rays drop at a very fast pace in the open scene compared to the closed one.

<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/SteamCompaction_open_closed.PNG" width="600"/> </p>

### Contiguous rays by material type
The shader implementation depends on the material with which the ray has intersected. So, If one warp has rays intersecting with different materials, it would lead to warp divergence and only the threads with one material could run at one time making it sequential in the number of materials. We can avoid this performance bottleneck by sorting the rays according to the material they are intersecting with so that rays interacting with the same material are contiguous in memory before shading and warp divergence is reduced. 

Following plot shows the average time per iteration with and without sorting tha paths according to the material type. There is a huge performance drop due to this. One potential reason for this is the number of materials (6 in this case) used to create the scene .Another reason is the sorting overhead. Probably the gain due to less warp divergence is not sufficient to make up for the sorting overhead. It might help in case we have a huge number of materials.

<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/Performance_MS.PNG" width="300"/> </p>

### Cache First Bounce
One first step of generating rays and finding itersection for the first bounce is same across all iterations with an exception while we are using anti aliasing. So, we could to an optimization by saving the first after the first iteration and reuse it rather than re doing it every time. 

Following plot shows average time per iteration with and without using cache. It took around 32 ms for iteration without using cache and 29 ms with cache. These number are calculated with an average across 10 iterations. The gap would increase with the complexity of the scene specifically the number of objects.
<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/Performance_Cache.PNG" width="300"/> </p>

## Bloopers
Following are some of the bloopers. First one was caused when I used an offset of 0.00001f instead of 0.0001f. The second was when I gave the reverse of eta to the refract function instead of eta.
<p align="center"><img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/Blooper1_0.00001.png" width="400"/>   <img src="https://github.com/DishaJindal/Project3-CUDA-Path-Tracer/blob/mesh-loading/img/Blooper2_inverse_eta.png" width="400"/> </p>
