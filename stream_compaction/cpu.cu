#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
		bool timeInProg = false;

        PerformanceTimer& timer()
        {
	        static PerformanceTimer timer;
	        return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
	        if (!timeInProg) { timer().startCpuTimer(); }

			// Exclusive Prefix Sum
			if (n >= 1) { odata[0] = 0; }

			for (int i = 1; i < n; i++) {
				odata[i] = idata[i - 1] + odata[i - 1];
			}

			if (!timeInProg) { timer().endCpuTimer(); }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();

            // Only output the indices of idata that aren't 0s
			int currIdx = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] == 0) { continue; }
				odata[currIdx] = idata[i];
				currIdx++;
			}

	        timer().endCpuTimer();
            return currIdx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			timeInProg = true;
	        
			// Mapping input to [0,1]
			int* binaryMap = new int[n];
			for (int i = 0; i < n; i++) {
				binaryMap[i] = (idata[i] == 0 ? 0 : 1);
			}

			// Scanning
			int* scanResult = new int[n];
			scan(n, scanResult, binaryMap);

			// Scatter
			int newLength = 0;
			for (int i = 0; i < n; i++) {
				if (binaryMap[i] == 1) {
					odata[scanResult[i]] = idata[i];
					newLength = scanResult[i] + 1;
				}
			}
			
	        timer().endCpuTimer();
			timeInProg = false;

			delete[] binaryMap;
			delete[] scanResult;

            return newLength;
        }
    }
}
