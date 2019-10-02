CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**
* Jiangping Xu
  * [LinkedIn](https://www.linkedin.com/in/jiangping-xu-365b19134/)
* Tested on: Windows 10, i7-4700MQ @ 2.40GHz 8GB, GT 755M 6100MB (personal laptop)
<p align="center">
    <img src = img/link.5000samp.png>
    Model credit to
    <a href="https://sketchfab.com/3d-models/link-from-wind-waker-033b1cee62a14dbfbb65f5deb6725265">jemepousse</a>
</p>

## Features
* Stream Compaction, Materials Sorting, First Bounce Caching
* Refraction with Fresnel
* Depth of Field
* GLTF Scene Loading
    * Bounding Box and arbitrary Meshes
    * Texture and Normal Mapping
* Motion Blur

## Demos and Analysis
#### Refraction with Fresnel
<p align="center">
    <img src = img/cornell.refract.2880samp.png><br> 
    <em>index of refraction is 1.5</em><br> 
    <p align="center">
    <img src = img/cornell.reflect.2068samp.png><br> 
    <em>pure reflection</em>
    </p>
</p>

I use the Fresnel coefficients to determine the ratio of reflection and transmission. When a ray hits the refractive surface, a uniform random number between 0 and 1 is generated and compared with the Fresnel coefficient. Then the ray reflects or refract accordingly.

#### Depth of Field
<p align="center">
    <img src = img/cornell.dof.3550samp.png>
</p>

For each ray cast from the camera, a intersection point with the focal plane is calculated. After that, choose a random point on the lens as the origin of the modified ray and cast it to the same intersection point on focal plane.

#### GLTF Scene Loading

glTF format is developed by the Khronos Group for the purpose of efficient transmission and loading of 3D scenes. It contains meshes, textures, uvs, normals ... and even lighting information. I use the tinyglTF library to load the gltf files. It really took me a long time to make glTF loading work. The tinyglTF library has bugs when compling and the examples of the library are incomplete. But the good thing is once you are able to load the data, you get everything you need (textures, normals and bounding box information).

I use the bilinear interpolation to get the uv of an arbitrary point. I use the glm::intersectRayTriangle() to calculate the intersection points between rays and triangles. (For bump maping, it is basically the same as texture mapping)

<p align="center">
    <img src = img/person1.685samp.png><br> 
    2.1k Triangles, credit to
    <a href="https://sketchfab.com/3d-models/low-poly-person-bfe451f06bba4a6baa4bae9f4b0b112e"> 
Dimitriy Nikonov</a>
    <p align="center">
        <img src = img/person2.531samp.png><br> 
        Procedural Texture
    </p>
</p>

When using either a procedral texture or a predifined texture image for the above rendering, it takes about 135ms for each iteration. The time difference between the two kinds of textures is depending on the specific implementation of procedural texture.

I also wrote a function to test whether a ray intersects with a bounding box. When use bounding box, rendering a scene with 12.8k triangles (the Zelda scene at the top of this readme) need 4s per iteration, while not using bounding box needs 5.3s per iteration. 135ms


#### Motion Blur
<p align="center">
    <img src = img/cornell.motionblur.2872samp.png>
</p>
Average the results of multiple renderings with different object positions to achieve a motion blur effect.

#### Stream Compaction

Stream compaction technique is applied here to end the terminated rays.

<p align="center">
    <img src = img/NumberofRaysAlongwithbouncesIncreasing.png>
</p>

Stream compaction frees the meaningless GPU memory usage and make sure every thread is computing rays' bouncing rather than determining whether a ray is dead (and waiting if diverge in a warp). 

To compare the open cornell box and a closed cornell box, I make the walls exrtremly long so that rays are almost not able to escape from the box while the total number of geometry is the same.

<p align="center">
    <img src = img/NumberofRaysAlongwithbouncesIncreasing2.png>
</p>

The time for each iteration increases from 130ms to 484ms. As expected, far less rays are terminated and be removed from stream compaction in the "closed" cornell box.