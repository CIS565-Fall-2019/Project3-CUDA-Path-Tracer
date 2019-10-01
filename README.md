
CUDA PATH TRACER
==================================================================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture**

Dhruv Karthik: [LinkedIn](https://www.linkedin.com/in/dhruv_karthik/)

Tested on: Windows 10 Home, Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz, 16GM, GTX 2070 - Compute Capability 7.5
____________________________________________________________________________________
![Developer](https://img.shields.io/badge/Developer-Dhruv-0f97ff.svg?style=flat) ![CUDA 10.1](https://img.shields.io/badge/CUDA-10.1-yellow.svg) ![Built](https://img.shields.io/appveyor/ci/gruntjs/grunt.svg) ![Issues](https://img.shields.io/badge/issues-none-green.svg)
____________________________________________________________________________________
<p align="center">
  <img  src="img/frontpage.png">
</p>

Table of contents
=================
   * [What is Path Tracing] (#What is path tracing?)
   * [Features Overview]
      * [BSDF Scattering: Diffuse, Specular-Reflective, Specular Transmissive]
      * [Procedural Shapes]
      * [Motion Blur]
      * [Stochastic Sampled Anti Aliasing]
  * [Optimizations ]
    * [Stream compaction to remove terminated rays]
    * [First bounce caching]
    * [Sort by Material]
   * [Questions]
   * [Performance Analysis]
   * [Credits & Acknowledgments]

## What is path tracing?
