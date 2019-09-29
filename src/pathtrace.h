#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree(Scene* scene);
void pathtrace(uchar4 *pbo, int frame, int iteration);

#if USING_OIDN
/**
Runs the OIDN functions on our vector of image pixels

Does this by doing one last run of pieces of our path tracing to fill buffers correctly to hand over
Returns a new vector of pixel values to save out
*/
gvec3_v runOIDN(gvec3_v image, int width, int height);
#endif