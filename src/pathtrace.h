#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree(Scene* scene);
void pathtrace(uchar4 *pbo, int frame, int iteration);
