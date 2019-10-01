#pragma once

#include <vector>
#include "scene.h"

#define ERRORCHECK 1
#define SORTBYMAT 0
#define CACHEFIRST 0
// When using depth of field, CACHEFIRST must be 0
#define DEPTHOFFIELD 0
#define GLTF 0
#define BBOX 1
#define MOTIONBLUR 0

#define LENSR 0.6f
#define FOCALDIS 11.5f

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);
void resetGeoms();
