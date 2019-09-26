#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(uchar4 *pbo, int frame, int iteration);


//// Sphere
//OBJECT 6
//sphere
//material 4
//TRANS - 1 4 - 1
//ROTAT       0 0 0
//SCALE       3 3 3
