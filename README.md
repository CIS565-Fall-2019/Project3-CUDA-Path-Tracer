CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Gangzheng Tong
* Tested on: Windows 10, i7-8750H @ 2.20GHz 16GB, RTX 2070 8GB (personal laptop)

### Overview

**Path tracing** is a [computer graphics](https://en.wikipedia.org/wiki/Computer_graphics "Computer graphics")  [Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method "Monte Carlo method") of [rendering](https://en.wikipedia.org/wiki/Rendering_(computer_graphics) "Rendering (computer graphics)") images of three-dimensional scenes such that the [global illumination](https://en.wikipedia.org/wiki/Global_illumination "Global illumination") is faithful to reality [Widipediea] 
In this project I implemented the basic path tracing algorithm with C++ and Cuda, taking advantage of the highly parallelial nature of GPU and achieved an interactive renderer. 

![](https://www.scratchapixel.com/images/upload/shading-intro2/shad2-globalillum1a.png?)
The general idea is to shoot a large number of rays from the camera, compute the intersections with the objects in the scene, and scatter more rays from the intersection point based on the material properties such as glassy or matte. If after a few bounces the scattered ray reaches a light source, we can shade the path segment based on the   light color and emittance,  

### Features
3 Different materials: 
* Diffuse
* Perfectly Specular
* Refraction with Frensel effects using [Schlick's approximation]([https://en.wikipedia.org/wiki/Schlick%27s_approximation](https://en.wikipedia.org/wiki/Schlick%27s_approximation))

![](img/3_materials.png)

Motion Blur
![](img/mb.png)
Depth of Field
![](img/dop.png)

**Other features:**
* Direct Light
* Sort by Material
* First Bounce Cache
* Stream Compaction for Removing Terminated Rays


### Performance Analysis


### References