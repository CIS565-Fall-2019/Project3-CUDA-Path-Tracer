# CUDA Path Tracer
**University of Pennsylvania, CIS 565: GPU Programming and Architecture,
Project 3 - Path Tracer**

Caroline Lachanski: [LinkedIn](https://www.linkedin.com/in/caroline-lachanski/), [personal website](http://carolinelachanski.com/)

Tested on: Windows 10, i5-6500 @ 3.20GHz 16GB, GTX 1660 (personal computer)

## Project Description

The purpose of this project was to create a GPU-parallelized path tracer using CUDA.

## Features

### Ray Termination via Stream Compaction

Each ray shot out from the camera will continue to bounce around a scene, accumulating color, until it terminates for any number of reasons, including hitting a light source, hitting no geometry, and reaching the maximum number of bounces. To avoid warp divergence by continuing to process rays that are terminated, stream compaction (using the thrust partition function) is used to reorganize all of the rays by whether they are terminated or not. Thus, following kernels can be called on only the yet unterminated rays.

### First Bounce Caching

Another potential optimization is first bounce caching. This means that on the first iteration (first sample per pixel), we cache the intersections found on the first bounce in the scene, and on subsequent iterations, reuse those first intersections. 

### Material Sorting

Each piece of geometry has a material ID corresponding to its material, and materials can be reused across different geometry. Since each material potentially follows a different code path in the scatterRay function, we can attempt to avoid warp divergence by sorting rays/intersections by their material ID. 

### glTF Mesh Loading

This path tracer also supports the rendering of arbitrary meshes using the glTF format, with help from [tiny glTF](https://github.com/syoyo/tinygltf). One toggleable performance optimziation is the use of a bounding box for each mesh (which is made easy to calculate since the glTF format already stores the per-component min and max positions). Rather than attempting to find an intersection with a mesh by immediately checking an intersection with its potentially many triangles, we first check if a ray intersects the mesh's bounding box, and return false if it does not.

### Depth of Field

We can create a physically-based depth of field effect in our renders using a thin lens approximation. Rather than the typical pinhole camera model, which has a lens with a radius of zero, we use a thin lens camera, which has a neglibile thickness but a non-zero radius. Rather than each ray from the camera starting from the same place, we jitter the ray's starting place to be somewhere on the camera lens.

### Bokeh

A fun consequence of the thin lens approximation is the creation of [bokeh](https://en.wikipedia.org/wiki/Bokeh), "blur produced in the out-of-focus parts of an image produced by a lens." The thin lens approximation typically features a circular lens, creating circular bokeh, but one can create other shapes by sampling points from a shape other than a disk.

### Various Materials

This project features diffuse reflection, perfectly specular reflection, transmission, and specular refraction using the Fresnel coefficient.

### Anti-Aliasing

The anti-aliasing used in this project is an easy-to-implement way to improve visual quality with little to no effect on performance. When shooting a ray from the camera, we jitter its starting position slightly, resulting in less jaggies.

## Analysis

## Bloopers
