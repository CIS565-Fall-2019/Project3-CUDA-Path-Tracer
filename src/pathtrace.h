#pragma once

#include <vector>
#include "scene.h"

#define SORT_BY_MATERIAL true
#define STREAM_COMPACT true
#define CACHE_FIRST_BOUNCE true
#define MOTION_BLUR false
#define ANTI_ALIASING false
#define DIRECT_LIGHTING true
#define DEPTH_OF_FIELD false
#define DENOISE true
#define TIMER false

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration, float3* albedo, float3* normals);
