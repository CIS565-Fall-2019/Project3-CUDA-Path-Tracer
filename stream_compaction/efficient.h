#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

		void workEfficientScan(int n, int *dev_idata, dim3 &threadsPerBlock, dim3 &fullBlocksPerGrid);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);
    }
}
