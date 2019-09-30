#pragma once

#include "common.h"

namespace StreamCompaction {
	namespace Shared {
		StreamCompaction::Common::PerformanceTimer& timer();

		void scan(int n, int *odata, const int *idata, int blockSize);

		int compact(int n, int *idata, int blockSize);
	}
}
