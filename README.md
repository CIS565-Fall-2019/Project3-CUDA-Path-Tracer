CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* SOMANSHU AGARWAL
  * [LinkedIn](https://www.linkedin.com/in/somanshu25)
* Tested on: Windows 10, i7-6700 @ 3.4GHz 16GB, Quadro P1000 4GB (Moore 100B Lab)

### What is Path Tracer and how it is different from Ray Tracing?

Path tracing is a realistic lighting algorithm that simulates light bouncing around a scene. it is based on Monte-Carlo based sampling technique and simulate realistic images through rendering. The path tracer is continually rendering, so the scene will start off grainy and become smoother over time. 

The following features are enabled in the path tracer:

* Basic Features:
  * BSDF evaluation for Diffusion and Perfectly Specular Surfaces
  * Sort the rays after intersection with material ID type it intersected
  * Stream Compaction using Thrust
  * Cache First Bounce
  
* Adanced Features:
  * BSDF evaluation including Refraction, Fresnel Effect using Schlick's approximation
  * BSDF evaluation for Different Percentage Comnibations for Refraction and Reflection
  * Anit-aliasing
  * Work-Efficient Stream Compaction using Shared Memory
  * Motion Blur (attempted)



