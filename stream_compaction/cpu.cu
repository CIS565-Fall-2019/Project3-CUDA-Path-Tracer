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
            bool standalone = true;
            try { timer().startCpuTimer(); }
            catch (std::exception) { standalone = false; }

            int sum = 0;
            for (int i = 0; i < n; i++)
            {
                odata[i] = sum;
                sum += idata[i];
            }

	        if(standalone){ timer().endCpuTimer(); }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
            
            int idxInOut = 0;
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                {
                    odata[idxInOut] = idata[i];
                    idxInOut++;
                }
            }

	        timer().endCpuTimer();
            return idxInOut;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
	        timer().startCpuTimer();
	        
            int* temp = new int[n];
            int* tempScan = new int[n];
            for (int i = 0; i < n; i++)
            {
                if (idata[i] != 0)
                    temp[i] = 1;
                else
                    temp[i] = 0;
            }

            scan(n, tempScan, temp);

            int num = 0;
            for (int i = 0; i < n; i++)
            {
                if (temp[i] == 1)
                {
                    odata[tempScan[i]] = idata[i];
                    num++;
                }
            }

	        timer().endCpuTimer();
            return num;
        }
    }
}
