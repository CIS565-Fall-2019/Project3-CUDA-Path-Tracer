#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

		void scanHelper(int n, int *odata, const int *idata) {
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i - 1];
			}
		}

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			scanHelper(n, odata, idata);
	        timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int k = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[k++] = idata[i];
				}
			}
	        timer().endCpuTimer();
            return k;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int *map_array = new int[n];
			int *scanned = new int[n];

			// mapping to boolean
			for (int i = 0; i < n; i++) {
				map_array[i] = idata[i] != 0 ? 1 : 0;
			}
			// scanning exclusively
			scanHelper(n, scanned, map_array);

			//scatter results
			int k = 0;
			for (int i = 0; i < n; i++) {
				if (map_array[i]) {
					k++;
					odata[scanned[i]] = idata[i];
				}
			}
			delete[n] map_array;
			delete[n] scanned;
	        timer().endCpuTimer();
            return k;
        }
    }
}
