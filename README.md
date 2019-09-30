CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

Peyman Norouzi
* [LinkedIn](https://www.linkedin.com/in/peymannorouzi)
* Tested on: Windows 10, i7-6700 @ 3.40GHz 16GB, Quadro P1000 4096MB (Moore 100B Lab)


## CUDA Path Ray Tracer:

![](img/w_MB.png)

In computer graphics, ray tracing is a rendering technique for generating photo realastic images. In this approach, we trace paths of light as they leave from a camera as pixels in an image plane and simulating the effects of them encountering with virtual objects. 


## Table of Contents:



## CUDA Ray Tracing Implementation:

I am implementing ray tracing on CUDA capable of rendering globally-illuminated images very quickly. The basic idea of the implementation can be seen below:

![](img/1280px-Ray_trace_diagram.svg.png)

When a ray leaves the camera (pixel), it can hit the objects in the enviroment and bounce, change direction or get diffused. So it is important to implement the rules that govern the behaviour/interactions between rays and materials and objects in the enviroment. A ray hitting an object can have the following behavior and outcomes: 

![](img/beh_img.png)

The behavior rules can be found below: 

![](img/Ray_Tracing_Illustration_First_Bounce.png)


### Core Implementation:

In our core implementation we will model refraction, reflection and difusion behavior of material/ray interaction. For the refraction/reflection implemnetation I will be using Snell's law with Frensel effects using [Schlick's approximation](https://en.wikipedia.org/wiki/Schlick's_approximation). In this implementation, the ray that gets fired from the camera bouces for a maximum of 8 time (Depth of 8) unless it gets diffused or hits the light source. The walls in this rednder only diffuse and the sphere in the enviroment both reflects and refracts. The result of the render is as follows:

![](img/Basic_core.png)

Now lets make the left and right walls into the same material as the sphere. The result looks pretty cool:

![](img/Basic_m.png)

lets have two objects now! lets make it red so that the whole render get a red hue! it is starting to look like a Salvador Dali painting!

![](img/Basis_2.png)

### Core Implementation + Anti-Aliasing:

we can use Stochastic Sampled Antialiasing method and add some noise to the position of our rays when they get fired from the camera. This would help us greatly for rendering edges of the objects. The result speak for themselves:

| Without Anti-Aliasing | With Anti-Aliasing |
| ------------- | ----------- |
| ![](img/wo_AA.png)  | ![](img/w_AA.png) |
| ---------------->![](img/wo_AA_Z.png)<---------------- | ---------------->![](img/w_AA_Z.png)<----------------|

As you can see we were able to greatly improve our rendering perfomance!

### Core Implementation + Anti-Aliasing + Motion Blur:

In this implementation we try to move objects in the image slowly as we are creating the render. As the objects moves, we can average samples(frames) at different times in the animation. The results look super cool:

| Without Motion Blur | With Motion Blur |
| ------------- | ----------- |
| ![](img/wo_MB.png)  | ![](img/w_MB.png) |
| ---------------->![](img/wo_MB_Z.png)<---------------- | ---------------->![](img/w_MB_Z.png)<----------------|


## Perfomance Implementation and Analysis:

In the naive approach, we track each rays motion and bouce, throughout its jounrney untill our depth requirment is met. But this is not the best and most effitient way to approach this since many rays will be terminating their journey earlier by either hitting the light source or a diffusing surface.

## Stream Compaction: 

Stream compaction would allow us to get rid of rays that have already terminted by hitting the light source or a diffusing surface. This way we can exit earlier in each iteration thus improving our performance. This is specially usefull when our depth is a larger number such as 32. We can see the performance improvment significantly as follows:


## First bounce intersections Caching:

## Material Sort:


### Cool Renders:

### Bloopers

