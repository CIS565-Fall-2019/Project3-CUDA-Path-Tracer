#pragma once

#include "common.h"

namespace StreamCompaction {
	namespace Shared {
		StreamCompaction::Common::PerformanceTimer& timer();

		void scanEfficient(int n, int *odata, const int *idata, int blockSize = 128);

		int compactCUDA(int n, int *dev_idata);
	}
}
