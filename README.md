CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Saket Karve
  * [LinkedIn](https://www.linkedin.com/in/saket-karve-43930511b/), [twitter](), etc.
* Tested on:  Windows 10 Education, Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz 16GB, NVIDIA Quadro P1000 @ 4GB (Moore 100B Lab)

### Highlights

### Features implemented

- Visual artefacts
  - Shading different materials
    - Ideal diffuse and shading
    - Perfect Specular reflection
    - Refraction with Fresnel effects and total internal reflection \[EXTRA CREDIT\]
  - Stochastic Anti-aliasing \[EXTRA CREDIT\]
  - Motion Blur \[EXTRA CREDIT\]
  - Arbitrary mesh loading and redering (OBJ) \[EXTRA CREDIT\]
- Performance improvements
  - Path termination using Stream Compaction
  - Cache first bounce
  - Sort by materials
  - Stream Compaction using shared memory \[EXTRA CREDIT\]

### Shading different materials

#### Ideal Diffuse

When rays fall on an object with ideal diffuse, the ray is scattered randomnly sampled from a uniform distribution in any direction within the hemisphere centered at the point of incidence. This gives a matt finish to the object.

![]()

#### Perfect Specular reflection

When rays fall on an object with perfectly specular material, it always (with 100% probability) reflects at an angle equal to the incidence angle on the other side of the normal. This gives a shiny mirror like finish to the object. Reflectons of objects around can be seen. Since it does not allow any light to pass through, we can see a shadow on the side which is away from light.

![]()

#### Refraction

When a ray falls on a refractive surface, it penetrates inside the object making an angle with the normal determined by the refractive index of the material the rays comes from and that of the object. This angle is determined by Snell's law. 

A perfectly refractive object is rendered as below.

![]()

However, objects are not perfectly refractive. Some proportion of the rays are reflected depending on the refractive indices of the materials and the angle of incidence of the ray. The reflectance coefficient (proportion of rays which refract) is calculated by Fresnel's law. Also, for rays coming from a material with higher refractive index to one with a lower refractive index, some rays reflect perfectly if the angle of incidence is more than a perticuar value (called critical angle). This is called total internal reflection. 

The render of an object with refraction wil be as follows.

![]()

### Anti-Aliasing

Depending on the resolution, when images are rendered, the pixels show a stair-step like lines near the edges of objects.
