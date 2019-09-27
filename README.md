CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Eric Micallef
  * https://www.linkedin.com/in/eric-micallef-99291714b/
  
* Tested on: Windows 10, i5, Nvidia GTX1660 (Personal)

## Performance Analysis

The below charts are gathered from the scene shown above. In this scene we have 4 small spheres with some reflection but are mostly diffuse.

![](img/1_loop.png)


## Algorith Analysis

### Material Sorting

![](img/Material_Sorting.png)

The idea behind material sorting is that we sort our rays by material. So that all rays with with Material ID 1 are next to each other and so on. This is to help reduce thread divergence. If all threads in a warp are computing based off a certain material we would expect that they would mostly behave the same way. For example, all threads with a specular material are all most likely going to reflect. 

In practicality there is not any speedup seen from this as most of the time spent is from computation. I suspected that the scene was not ocmplex enough and added many objects with many different materials to see if some speed up could be achieved but again, I saw no benefit from material sorting which is a bit unintuitive.

### First Bounce Caching

![](img/Caching.png)

The idea behind first bounce caching is to reduce repetitive computations. As we can see from the graphs above most of the time spent is from computation. If we can save some computation time we will see a noticeable difference.

Every sample, rays are generated from the camera and enter the scene. for the first sample the rays will go to the same spot. Instead of computing this every time we can cache it on our cpu and just reuse it for the next iterations. thereby we get a savings of depth-1.

As we allow more bounces or depth to the scene the benefit of this technique will diminish.


### Stream Compaction 

![](img/Stream_Compaction.png)

The idea behind stream compaction is to remove dead rays or ( rays that are no longer bouncing ) from the scene. Dead rays are either rays that have hit a light object, have exited the scene or make no contribution to the scene. 

In our naive path tracer our kernel launches a thread per ray. After the first bounce some rays exit and after more bounces more rays will exit. Yet we will launch a thread for this ray. 

To remedy this we use stream compaction. After every bounce we can remove the rays that are no longer bouncing. Thus, launching less kernels, thus less work to be done.

This technique sees a small speedup in performance. My guess is that if I had a more complex scene where rays had to do more computational work then this speed up would be more significant.

My guess is although we may not see a speed up in run time we may see a pretty significant reduction in power from the GPU. 

### Anti-aliasing

The idea behind anti aliasing is to add some jitter or randomness to the camera rays. Instead of having the rays shoot out the same direction every time we add a slight random offset. This has the rays bounce in different possibly more interesting directions to accumlate color. What we see with the anti aliasing is edges become a bit smoother. This technique is essentially free because all we are doing is adding some randomness to the input rays. 

Unfortunately if we use anti aliasing we can not use the First Bounce Cache technique. This is because we are manipulating our first rays just a bit. So the first computation will not always be the same.

We could possibly use both techniques if we made the "random-ness" of anti aliasing more deterministic. For example, we adjust the camera jitter based off of our sample number. IF we did this though we would have to have cache more elements meaning more memory useage.

![](img/combo_alias.jpg)

The left image spehere is a render with anti aliasing. You will notice that the edges are much smoother. 

On the right is a render with no aliasing. You will notice the edges are more abrupt and jagged. 

Both of these images are at the same sample point. 

Below is the image render even from afar the jagged edges are noticeable!


![](img/cornell_no_alias.png)

No Alias

![](img/cornell_alias.png)
 
Anti Aliasing


## Refraction

From our results above we see that refracion addded a bit of overhead but not too much. This is expected as adding refraction adds a bit more branch divergence. But the time spent computing refraction is not significantly more than computing a reflective or diffuse object so we only see a slight slow down.

Below is a render with my attempt to simulate refractive water. To do this I just added a thin blue refractive material. It is not the prettiest render but it shows refraction.

![](img/water_refraction.png)

From the image above we can see the distortion on the walls and slight distortion of the tower.

In real life, water and other refractive materials have some reflection to them. When you look into a lake if it is calm enough you can see a reflection. This is where Fresnels Number can create a more realistic render. 

In the below image we add some reflective and refractive properties to the "water" and we can see how the middle area has a little glean to it.

![](img/water_refraction_p2.png)


## depth of field

From our chart we see depth of field did not add too much if any overhead to the system.
Similar to anti-aliasing this technique is done once per iteration and makes adjustments to the camera. the compute time for this is pretty minimal there by it is masked away. 

Similar to anti aliasing we can not emply our caching speed up when do depth of field. This is so because we are manipulating the camera differently each iteration to get that blur effect.


Below is a render of what you would see if you have perfect vision

![](img/no_blur.png)

Below is what you would see if you had decent eye sight or over the age of 60

![](img/blur.png)

And finally, this is what I see without my contacts

![](img/full_blur.png)


### Pictures


### References

Physically Based Rendering second and third edition
	Used this books algorithms to employ Depth of field
	Used book for refraction
	Used book for fresnels

https://www.scratchapixel.com/lessons/3d-basic-rendering/introduction-to-shading/reflection-refraction-fresnel
	in help with refraction

https://en.wikipedia.org/wiki/Snell%27s_law
	more help with refraction and fresnels

https://computergraphics.stackexchange.com/questions/2482/choosing-reflection-or-refraction-in-path-tracing
	for help with fresnels

https://github.com/ssloy/tinyraytracer/wiki/Part-1:-understandable-raytracing
	in general helped with the basics of fresnels, reflection, refraction




