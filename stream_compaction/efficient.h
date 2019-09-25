#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scanEfficient(int n, int *odata, const int *idata, int blockSize = 128);

		void scanEfficientCUDA(int n, int *odata, const int *idata, int blockSize = 128);

		void scan(int n, int *odata, const int *idata, int blockSize = 128);

        int compact(int n, int *odata, const int *idata, bool efficient = true, int blockSize = 128);
    }
}
