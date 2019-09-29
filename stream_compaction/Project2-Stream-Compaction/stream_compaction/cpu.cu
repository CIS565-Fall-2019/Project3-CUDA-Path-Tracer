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
			bool tmp=true;
			try { 
				timer().startCpuTimer();
			}
			catch (const std::runtime_error& e) {
				tmp = false;
			}
			
            // TODO
			if (n > 0) {
				odata[0] = 0;
				for (int i = 0; i < n-1; i++) {
					odata[i+1] = idata[i] + odata[i];
				}
			}
			if(tmp ==true) timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            // TODO
			if (n > 0) {
				timer().startCpuTimer();
				int counter = 0;
				for (int i = 0; i < n; i++) {
					if (idata[i] != 0) {
						odata[counter] = idata[i];
						counter+=1;
					}
				}
				timer().endCpuTimer();
				return counter;
			}
            return -1;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        // TODO
			if (n > 0) {
				timer().startCpuTimer();

				int * indicator = new int[n];
				int * scanIndex = new int[n];
				int tmp = 0;

				// Compute indicator array
				for (int i = 0; i < n; i++) {
					if (idata[i] != 0) {
						indicator[i] = 1;
					}
					else {
						indicator[i] = 0;
					}
				}

				// Compute scan
				scan(n, scanIndex, indicator);

				//Scatter
				for (int i = 0; i < n; i++) {
					if (indicator[i] == 1) {
						odata[scanIndex[i]] = idata[i];
						tmp = scanIndex[i];
					}
				}
				timer().endCpuTimer();
				return tmp+1;
			}
            return -1;
        }
    }
}
