CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**


* Author: Chhavi Sharma ([LinkedIn](https://www.linkedin.com/in/chhavi275/))
* Tested on: Windows 10, Intel Core(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, 
             NVIDIA Quadro P1000 4GB (MOORE100B-06)


 <p align="center">
  <img src="img/cornell_cover.png">
</p>


### Index

- [Introduction](  )
- [Implementation Details]( )
- [Basic Features]( )
- [Advance Features]( )
- [Performacne Analysis]( )
- [Extra Credit]( )


### Introduction 

Path tracing is a computer graphics method of rendering digital 3D images such that the global illumination is as close as possible to reality. Path Tracing is similar to ray tracing in which rays are cast from a virtual camera and traced through a simulated scene by random sampling to incrementally compute a final image. The random sampling process makes it possible to render some complex phenomena which are not handled in regular ray tracing such as multiple reflections.

### Implementation Details
We implement an estimation of the Bidirectional Scattering Distribution Function to compute the an estimated illumination per pixel in the image over several iterations. In reality: Rays leave light sources -> bounce around a scene and change color/intensity based on the scene’s materials -> some hit pixels in a camera/ our eyes. Our implementation simulations this phenomnenon in reverse where a ray is launched from our camera thorugh each pixel of the image, and it's subequent intersections and bounces in the scene are traced upto a certain depth to compute the final color of the pixel. 
This is implemented by computing a single bounce at each time-step for all the rays in the image parallely to get maximum throughput.  

The ray starts with an identity color which is modified multiplicatively as it hits differnet materials in the scene.
The bounce direction and colour intensity depend on various material properties and the angle of incidence. We simulate four types of materials i.e. Emissive, Diffused, Reflective, Refractive and their combinations.

### Basic Features
The following basic features are implemented:
   - Shading using [BSDF](https://en.wikipedia.org/wiki/Bidirectional_scattering_distribution_function)
      - Diffuse Reflection: Reflects all rays randomly in the normal facing semi-sphere.
      - Specular Reflection: Reflects the incoming ray about the normal where angle of incidence is equal to the angle of relection (mirror like behaviour).
      - Refraction: Allows ray to pass through the media based on the ratio of the refractive index of the two mediums [snell's law](https://en.wikipedia.org/wiki/Snell%27s_law)
      - Emissive Media: Rays in our computation terminate at these materials since they emit light.
      - Percentage combinations of properties.   
      
<p float="left">
  <img src="build/cornell_refraction.png" width="300" />
  <img src="build/cornell_reflection.png" width="300" />
  <img src="build/cornell_diffuese.png" width="300" />   
</p>      

      
<p float="left">
  <img src="build/cornell_Refract-Reflect-50-50.png" width="300" />
  <img src="build/cornell_emmisive.png" width="300" />
  <img src="build/cornell_emmisive.png" width="300" />
</p> 


