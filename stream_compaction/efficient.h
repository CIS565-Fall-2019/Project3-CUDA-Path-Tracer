#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scan(int n, int *odata, const int *idata, int blockSize);

		void scan_device(int n, int *odata, const int *idata, int blockSize);

        int compact(int n, int *odata, const int *idata, int blockSize);
    }
}
