#pragma once

// A common place to enable or disable certain optimizations.

//// PRESHADER_SORT
// This enables a step before shader kernels are called that sorts paths by
// the materials they intersect with. When paths are sorted, kernels benefit
// from 1) cache coherene and 2) minimal path divergence.
#define PRESHADER_SORT 1

//// POSTSHADER_PARTITION
// Postshader Partition looks at all the paths after shading and seperates
// the paths into two groups: completed and not completed. Indicies provided
// to further kernel in the iteration are adjusted so that they ignore completed
// paths. This reduces the overall number of threads executed.
#define POSTSHADER_PARTITION 1