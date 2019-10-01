CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Alexis Ward
  * [LinkedIn](https://www.linkedin.com/in/alexis-ward47/), [personal website](https://www.alexis-ward.tech/)
* Tested on: Windows 10, i7-8750H CPU @ 2.20GHz 16GB, GTX 1050 Ti 

![](img/dof/0.2dof-d6.82-l0.25-5000samp.png)

# README

Path tracing is one of the most integral and commonly used methods for rendering 3D scenes. It allows a realistic portrayal of how light bounces through a scene. For each pixel of an image, we cast a ray from our camera into the scene, collide with an object and keep bouncing until we find a light source, stop intersection objects, or reach maximum bounce-depth, compiling color data along the way.

&#x1F537; There are many different features that are toggleable through the code; for all the options, read below!

## PART 1: Path Tracer Set-Up: Naive Shader, Basic Materials, and Memory Management 

`shadeNaive()`: I first created a "naive" implementation of a path tracer, dubbed as such because ray bounces for diffuse materials are purely random, and it contains no optimizations to converge the scene faster (like multiple importance sampling or any weighted random values). This shader is how the program runs on default. My kernel is based on the base code's `shadeFakeMaterial()` function, and the main difference comes in the `scatterRay()` function call and the ability to track `remainingBounces` of the currently processed path.


### Reflective Materials

![](img/Main/0-spec-5000samp.png)

The aforementioned `scatterRay()` function handles the direction of the next bounce and the properties of the current colors, both based on the given intersection's material. Perfectly specular (reflective) materials return a bounced ray based on the angle of the incoming ray, and their colors are heavily based on reflections rather than it's own inherant color (which can tint the reflections).


### Diffuse Materials

![](img/Main/0-diff-5000samp.png)

Perfectly diffuse materials can reflect light from any direction, and so the next direction for a ray bouncing off a diffuse surface is chosen at random. This random direction is sampled from a hemisphere surrounding the intersection, and the sampling method can change the output result (see cosine weighted hemisphere below).


### Stream Compaction, Partitioning by Material, and Cacheing for Speed

![](img/Main/0cornell5000samp.png) ![](img/graphcomp)

&#x1F537; Toggleable options in `pathtrace.cu`: 
* Line 21, set `TOGGLESTREAM = true` for Stream Compaction
* Line 22, set `TOGGLESORT = true` to sort paths by Material
* Line 23, set `TOGGLECACHE = true` to cache the first bounce intersections


Minimizes the chances of divergent warps; every path completing the same tasks (since many are based on material, see `scatterRay()`)

Measurements were taken on the scene pictured at left, (but for 1000 samples rather than 5000).
Diffuse Cornell, Diffuse and Specular Cornell (specular sphere), My Cornell (diffuse walls and box, two transmissive and one reflective spheres)
under each: No optimization, Stream compaction, partitioning, cache, all three



### Depth of Field

&#x1F537; **Toggleable option:** Line 26 of `pathtrace.cu`, set `DEPTHOFFIELD` to `true` if you desire this effect, `false` if you do not

    **Important** You must turn off the first bounce-caching, because this method relies on having slightly random rays coming from the camera. This is controlled by `TOGGLECACHE`



### (TODO: Your README)

*DO NOT* leave the README to the last minute! It is a crucial part of the
project, and we will not be able to grade you without a good README.

