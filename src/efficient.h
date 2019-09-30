#pragma once

#include "common.h"

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

		void scanShared(int n, int *odata, const int *idata);

        void scan(int n, int *odata, const int *idata);

        int compact(int n, int *odata, const int *idata);

		int compactShared(int n, int *idata);

		void scanCompact(int n, int *odata, const int *idata);
    }
}
