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

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
			bool timer_started = false;
			try { 
				timer().startCpuTimer();
			}
			catch (const std::exception& e) {
				timer_started = true;
			}
			odata[0] = 0;
			for (int i = 1; i < n; i++) {
				odata[i] = odata[i - 1] + idata[i-1];
			}
			if (timer_started == false) {
				timer().endCpuTimer();
			}
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int count = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[count] = idata[i];
					count++;
				}
			}
            // TODO
	        timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
			int *scanned = new int[n];
			int *mask = new int[n];
			for (int i = 0; i < n; i++) {
				mask[i] = idata[i] == 0 ? 0: 1;
			}
			scan(n, scanned, mask);
			int length = 0;
			for (int i = 0; i < n; i++) {
				if (idata[i] != 0) {
					odata[scanned[i]] = idata[i];
					length = scanned[i];
				}
			}
	        timer().endCpuTimer();
            return length+1;
        }
    }
}
