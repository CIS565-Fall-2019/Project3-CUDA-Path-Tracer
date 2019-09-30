#pragma once

#include <vector>
#include "scene.h"
#include "../../stream_compaction/common.h"

namespace Path_Tracer {
	StreamCompaction::Common::PerformanceTimer& timer();
	void pathtraceInit(Scene *scene);
	void pathtraceFree();
	void pathtrace(uchar4 *pbo, int frame, int iteration);
}
