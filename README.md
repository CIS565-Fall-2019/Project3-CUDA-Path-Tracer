CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Taylor Nelms
  * [LinkedIn](https://www.linkedin.com/in/taylor-k-7b2110191/), [twitter](https://twitter.com/nelms_taylor), etc.
* Tested on: Windows 10, Intel i3 Coffee Lake 4-core 3.6GHz processor, 16GB RAM, NVidia GeForce GTX1650 4GB

## Path Tracer

![Glass Zelda on a Textured Altar](progressImages/demoRenderFil.png)

A Path Tracer is a method of rendering virtual geometry onto the screen. Notably, they do so by simulating how light moves around a scene. This is in contrast to traditional rendering methods, which transform the geometry more directly from world-space to screen-space. While path tracers are slower than traditional renderers, they are able to natively perform much more impressive feats overall.

![glass, steel, and mirror ball in a checkered hellscape](progressImages/checkers_unfiltered.png)

For example, features such as *caustics* (the more intense light on the floor at the bottom-left of the image), or complex reflections and re-reflections, are easier to get with path-tracing than other methods.

This particular implementation is running on the GPU (graphics processing unit) via the CUDA framework, which allows us to parallelize the rigorous task of doing all our calculations for each pixel one at a time. This allows for significant speed-ups over CPU path tracers.

## Features

### Arbitary Mesh Loading

For an Object file in the scene description, it may be given the type “mesh.” Their transformation parameters act the same way, but you may also specify a “FILE” string. This can be the path to an `.obj` file, or a `.gltf` file (the latter of which must be in the same place as its assets).

![Bunny!](progressImages/day4Bunny.png)*classic Bunny obj file*

This loads all the file’s triangles into our data structures, and can then be rendered sensibly. As of now, no material characteristics are loaded in; however, textures are recognized for `.gltf` files, across a few metric types. All of my gltf files came from Sketchfab's automated conversion system, so certain conventions may have become baked into the code.

#### Bounding Volume

I implemented a simple axis-aligned bounding volume for my loaded meshes. That is to say, I constructed a "box" around each based on the maximum and minimum `x`, `y`, and `z` values for all of my triangle vertices, and did ray intersection tests on the box before attempting to intersect with each triangle in the mesh.

##### Performance

The image of the three guns was a good testbed for bounding box testing; the scene had a lot of triangles, but they were very densely localized. As such, cutting out a significant number of the triangle intersections was relevant.

I saw, with bounding-box culling, a time of `144s` to reach 500 iterations. Without, the same scene took `236s`. This, additionally, gives evidence that one of the more time-consuming parts of the path-tracer is intersecting each ray with the scene geometry; taking out roughly 2/3 of the intersections tests produced a 61% speedup.

![Gun Trio](progressImages/rifleTrioFil.png)

#### Textures

Using the CUDA texture memory, I was able to hook up `.gltf` files with their textures within the ray tracer. I worked almost entirely from files downloaded from [Sketchfab](https://sketchfab.com/), so their naming conventions may have ended up baked in to my implementation. (Specifically, the file naming for their texture image files is how I distinguish between different types of texture mappings.)

I acheived this by loading each of four different potential texture layers into the GPU's texture memory: color, emissivity, metallicRoughness (which also encoded an (unused) ambient occlusion parameter), and a bump texture (normal mapping). (These were what I was finding from the Sketchfab models I was downloading.) If a loaded model had a texture, I would keep track of its existence, and when applicable, pull it from memory. This was able to cut down on some amount of frustration of interpolating between values, as CUDA handled that for me. This convenience was counterbalanced by the frustration of figuring out how to navigate texture memory like CUDA wanted me to.

Here is a series of images of the same scene, with differing level of textures applied to them.

<figure>
<img src="progressImages/altar0.png" alt="No Textures"
	title="No Textures" width="600" height="450" />
 <figcaption>No Textures</figcaption>
</figure>
 
<figure>
<img src="progressImages/altarC.png" alt="Color Texture"
	title="Color Texture" width="600" height="450" />
 <figcaption>Color Texture</figcaption>
</figure>

<figure>
<img src="progressImages/altarCE.png" alt="Color and Emissivity Textures"
	title="Color and Emissivity Textures" width="600" height="450" />
 <figcaption>Color and Emissivity Textures</figcaption>
</figure>

<figure>
<img src="progressImages/altarCEM.png" alt="Color, Emissivity, and Metallic Textures"
	title="Color, Emissivity, and Metallic Textures" width="600" height="450" />
 <figcaption>Color, Emissivity, and Metallic Textures</figcaption>
</figure>

<figure>
<img src="progressImages/altarCEMN.png" alt=Color, Emissivity, Metallic, and Normal Textures"
	title="Color, Emissivity, Metallic, and Normal Textures" width="600" height="450" />
 <figcaption>Color, Emissivity, Metallic, and Normal Textures</figcaption>
</figure>

##### Performance

The above images took the following times to complete (across 2000 iterations):

| Textures | None | Color | Emissive | Metallic/Roughness | Normal |
|----------|------|-------|----------|--------------------|--------|
| Time(s)  | 171  | 172   | 173      | 189                | 198    |

It is unsurprising that the color textures and emissivity textures did not affect performance much, as their values were easy to pull directly from texture memory. I imagine the metallic/roughness textures worsened performance at least a little bit because they introduced variance in the rendering method for the mesh; instead of always being diffuse, it needed to occasionally bounce off in a specular pattern, which may (due to the roughness) have involved a power function for the non-perfect specularity. The normal maps, as well, likely hurt performance for how the variation amongst warps changed.

I also implemented a *very* simple procedural texture for use on my cube primitives (though, in theory, it could be applied to the triangle meshes as well without too much issue). It was a simple checkerboard color texture, as well as a slight set of normal variations to create a waviness in a rough grid pattern. It is, reluctantly, shown here (the back wall combines the checkerboard and some bumpiness/"waviness"):

![Checkerboard mirror](progressImages/day8checkerboard.png)

However, all it did was fill in texture memory in the same way that the loaded textures did; given that the GPU accesses both in the same way, there was no noticeable performance difference.

### Specular Sampling with Exponent

Implemented specular reflections with configurable exponent. Pictured below is a comparison of various exponential values for specularity. Notice that the very high value is effectively mirror-like; with such a highly specular object, the slight variations we get off the "mirror" direction are small enough to, effectively, not alter the ray at all. In this fashion, if we wished, we could eliminate the idea of "reflectivity" from our material description altogether.

![Shiny balls with their exponents noted](progressImages/day10shinyAnnotated.png)

Note: I used powers of three solely because they created a reasonable range of shininess across 7 samples; I have no idea if there was any computational speedup or slowdown because of this.

### Refraction

Refraction turned out to be trickier than I anticipated. Notably, it made triangle intersection tests more difficult, because I now had to check my meshes for backface triangles. (A smarter implementation than mine might only do so if the material for the mesh as a whole were refractive.) However, allowing for refraction on more complex models meant that I could display much more interesting results.

![Glass Zelda](progressImages/day8glass3.png)

You can see some of the other objects in the scene through Zelda's dress in this image; in particular, the shadow of the ball behind her.

Of course, it's also better with a more interesting background:

![Zelda with Altar](progressImages/oidn_zelda_2000nof.png)

Here, you see the emissive materials behind her come through, distorted but still clear. The "candle" light through her hand is also visible.

### Antialiasing

I implemented some simple antialiasing as well, modifying the starting camera ray within its pixel in a random fashion. The differences may be seen between these two images:


<figure>
<img src="progressImages/cornellNoAntialiased.png" alt="No Antialiasing"
	title="No Antialiasing" width="700" height="700" />
 <figcaption>No Antialiasing</figcaption>
</figure>

<figure>
<img src="progressImages/cornellAntialiased.png" alt="Antialiasing"
	title="Antialiasing" width="700" height="700" />
 <figcaption>Antialiasing</figcaption>
</figure>

You can spot the difference around the sphere, in particular; the presence of "jaggies" in the non-antialiased picture gives it away.

### Material Sorting

In order to attempt to reduce warp divergence, and better make use of the GPU resources, I implemented a pass to allow for material sorting between computing intersections and shading the materials.

For a simple scene, such as `scenes/checkersdemo.txt`, I was able to get 200 iterations deep in `20s` without material sorting. With sorting, it took `50s`.

Now, that was with only a dozen or so primitives; surely, when dealing with hundreds or thousands of triangles, the performance will be improved!

For a more complex scene, such as `scenes/teapotdemo.txt` (containing some 16,000 triangles), without sorting the materials, it took `75s` to get to 200 iterations; with sorting, it took `79s` to get to 200 iterations. Still not a performance boost, but better nonetheless.

When working with a significantly complex scene, such as `scenes/bunnydemo.txt` (containing 144,000 triangles), it took `184s` to get to just 50 iterations (pardon my impatience); with material sorting, it took `186s` total.

I can honestly conclude that it did not make much of a difference in my application; I suspect that what warp divergence I encountered came from the randomness involved in the ray-scattering function, which happened after the material sorting. Additionally, many of the objects in my scenes were spatially very localized as well, which probably cut down on the unsorted divergence.

### First-Intersection Caching

It was useful to save the first camera-ray and scene intersection across iterations. I did not have the chance to do an extensive performance analysis, but an off-the-cuff trial on `scenes/altardemo.txt` showd that with the optimization, running 1000 iterations took `98s`, as opposed to `111s` without. Given a max depth of `12` on that scene, that's a pretty reasonable speedup.

### Stream Compaction

I made use of stream compaction to get rid of the terminated paths as I went through the depth of each iteration, removing those that had hit nothing or an emitter. I used the `thrust` libraries to do so, and unfortunately, I did not get a chance to examine the performance implications of doing so. I imagine they are significant.

### Open Image Denoiser

The [OpenImageDenoiser](https://github.com/OpenImageDenoise/oidn) was a particularly interesting (albeit late) addition. It uses machine learning (read: magic) to take some of the gritty noise out of a ray-traced image, and construct it as if it were closer to being converged.

I elected to only feed the image in at the very end of a run, so as to not sully the process of accumulating light up to that point. Notably, running an image for longer improves the final image, but I was able to get smoother images from the beginning than I would have anticipated, feeding just the initial normal and albedo maps into the program. See the following comparison of image qualities of the same scene after a different number of iterations through the path tracer:



<figure>
<img src="progressImages/oidn_zelda_50nof.png" alt="50 iterations, no filter"
	title="50 iterations, no filter" width="500" height="500" />
 <figcaption>50 iterations, not filtered</figcaption>
</figure>
 
<figure>
<img src="progressImages/oidn_zelda_50fil.png" alt="50 iterations, filtered"
	title="50 iterations, no filter" width="500" height="500" />
 <figcaption>50 iterations, filtered</figcaption>
</figure>



<figure>
<img src="progressImages/oidn_zelda_200nof.png" alt="200 iterations, no filter"
	title="200 iterations, no filter" width="500" height="500" />
 <figcaption>200 iterations, not filtered</figcaption>
</figure>

<figure>
<img src="progressImages/oidn_zelda_200fil.png" alt="200 iterations, filtered"
	title="200 iterations, filtered" width="500" height="500" />
 <figcaption>200 iterations, filtered</figcaption>
</figure>


<figure>
<img src="progressImages/oidn_zelda_2000nof.png" alt="2000 iterations, no filter"
	title="0200 iterations, no filter" width="500" height="500" />
 <figcaption>2000 iterations, not filtered</figcaption>
</figure>

<figure>
<img src="progressImages/oidn_zelda_2000fil.png" alt="2000 iterations, filtered"
	title="2000 iterations, filtered" width="500" height="500" />
 <figcaption>2000 iterations, filtered</figcaption>
</figure>

As you can see, the filtering smoothed out even particularly rough images, but also eliminated some significant actual detail; only the last image was able to acheive a good amount of detail that got through the filtering process.


## Configuration Notes

### Run Options

Most of the switches for features and performance in the code are in `#define` variables at the top of `utilities.h`. Note that, if antialiasing is used, the first intersection is not cached.

### CMakeLists changes

I put the `tinyobjloader` library contents into the `external` folder, so I had to include the relevant header and source file in the project, as well as mark their locations to be included and linked.

I added the [OpenImageDenoiser](https://github.com/OpenImageDenoise/oidn) library to the `external` folder, and so added the line `include_directories(external/oidn/include)` so that the headers could be read sensibly. Additionally, I added the subdirectory `external/oidn`, and linked the `OpenImageDenoise` library to the `target_link_libraries` function. This did not end up doing all the necessary linking (see below), but it helped.

Notably, this required having Intels `tbb` installed; I acheived this by signing up for, and subsequently installing, [Intel Parallel Studio](https://software.intel.com/en-us/parallel-studio-xe). Time will tell if I made the right decision.

Additionally, I decided to compile this all with `C++17`, in case I decided to make use of the `std::filesystem` library (a slight quality of life fix over just calling it via `std::experimental::filesystem`). I admittedly am not sure whether this change actually took.

#### Moving DLLs

Look, I don't like what I did either.

I manually copied the `OpenImageDenoise.dll` and `tbb.dll` from their rightful homes to the directory where my built `Release` executable was, so that it might run.

Certainly, CMake has a way to do this, but as somebody who is not a CMake wizard at this point, this will have to do.

## Sources

### 3D Models
* Models downloaded from Morgan McGuire's [Computer Graphics Archive](https://casual-effects.com/data)
    * Bunny, Dragon, Teapot, Tree, Fireplace Room
* Turbosquid
    * [Wine Glass](https://www.turbosquid.com/FullPreview/Index.cfm/ID/667624) by OmniStorm (unused)
    * [Secondary Wine Glass](https://www.turbosquid.com/FullPreview/Index.cfm/ID/932821) by Mig91 (unused)
* Sketchfab
    * [Fountain](https://sketchfab.com/3d-models/fountain-07b16f0c118d4073a81522a526183c11) by Eugen Shuklin
    * [Altar](https://sketchfab.com/3d-models/altar-9b20f669e75441bcb34476255d248564) by William Chang
    * [Zelda](https://sketchfab.com/3d-models/ssbb-zelda-6612b024962b4141b1f867babe0f0e6c) by ThatOneGuyWhoDoesThings
    * [Sheik](https://sketchfab.com/3d-models/ssbb-sheik-4916d918d2c44f6bb984b59f082fc48c) by ThatOneGuyWhoDoesThings
    * [Hunter Rifle](https://sketchfab.com/3d-models/hunter-rifle-wip-ae83df4cc35c4eff89b34f266de9af3c) by cotman sam
    * [Textured Cube](https://sketchfab.com/3d-models/textured-cube-a883bf6dfd144419929067067c7f6dff) by Stakler (used for development)
    * [Sci-fi blaster](https://sketchfab.com/3d-models/sci-fi-assault-rifle-laser-blaster-f730872e1ee843e9a3934e9e3f6719c0) by Artem Goyko
    * [Plasma gun](https://sketchfab.com/3d-models/avas-ai-plasma-gun-a23c67a856dc43b1a7f34aacede9f183) by alx_flameniro

### Other Code
* Used [TinyObjLoader](https://github.com/syoyo/tinyobjloader) library for loading `*.obj` files
* Used [TinyGltf](https://github.com/syoyo/tinygltf) library for loading `*.gltf` files
    * I also lifted their `gltf_loader` files from their raytrace examples (and fixed a bug therein). I did not use any other code from the example folder.
* [OpenImageDenoiser](https://github.com/OpenImageDenoise/oidn) for post-processing
* Formerly: Ray-triangle intersection algorithm stolen from the Wikipedia article for the [Moller-Trumbore Intersection Algorithm](https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm). Now, using glm, to my chagrin (I swear I could have intersected both front and back faces had I gotten that algorithm to work correctly).
   
