#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
float pathtrace(uchar4 *pbo, int frame, int iteration); //returns time
