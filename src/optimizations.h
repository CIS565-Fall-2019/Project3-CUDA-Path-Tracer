#pragma once

// A common place to enable or disable certain optimizations or additional options

//// PRESHADER_SORT
// This enables a step before shader kernels are called that sorts paths by
// the materials they intersect with. When paths are sorted, kernels benefit
// from 1) cache coherene and 2) minimal path divergence.
constexpr bool 
PRESHADER_SORT = true;

//// POSTSHADER_PARTITION
// Postshader Partition looks at all the paths after shading and seperates
// the paths into two groups: completed and not completed. Indicies provided
// to further kernel in the iteration are adjusted so that they ignore completed
// paths. This reduces the overall number of threads executed.
constexpr bool 
POSTSHADER_PARTITION = true;

//// CACHE_ENABLED
// On the first iteration of each render sample, the first set of intersections are 
// the most expensive. This optimization stores the first set of intersections so that
// proceeding render samples are a little faster.
constexpr bool
CACHE_ENABLED = false;

// Not sure if the below is true, but I suspect it will be...
constexpr bool
ANTIALIASING = true;
static_assert(true != (CACHE_ENABLED && ANTIALIASING), "Cannot have cahcing and antialiasing enabled together!");

//// GENERATE_TERRAIN
// This option choses to ignore the scene file and instead generate a random
// terrain based on noise.
constexpr bool
TERRAIN_GENERATION = false;

//// DEPTH_OF_FIELD
// Simulates a depth of field effect by adjusting ray directions and origins
// to fall on a thin-lens model concentric disk.
constexpr bool
DEPTH_OF_FIELD = false;
// It seems that CUDA doesn't like constexpr for __device__variables
#define DOF_LENS_RADIUS 0.1f  // millimeters, controls size of disk of confusion
#define DOF_FOCAL_DIST 5.0f // millimeters, controls focal plane

//// MOTION_BLUR
// Applies a motion blur effect to objects. Geoms will now accept a VELO field containing (x,y,z)
// velocities. Each ray will contain a temporal jitter that is multiplies against VELO.
constexpr bool
MOTION_BLUR = true;