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
            timer().startCpuTimer();
            // TODO
            int running = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = running;
                running += idata[i];
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            // TODO
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return 0;
            }
            timer().startCpuTimer();
            // TODO
             //Step 1:  Compute temporary array containing
            int* mask = new int[n];
            for (int i = 0; i < n; ++i) {
                mask[i] = (idata[i] != 0) ? 1 : 0;
            }

            //Step 2:  Run exclusive scan on temporary array
            int* indices = new int[n];
            int running = 0;
            for (int i = 0; i < n; ++i) {
                indices[i] = running;
                running += mask[i];
            }

            //Step 3: Scatter
            for (int i = 0; i < n; ++i) {
                if (mask[i]) {
                    odata[indices[i]] = idata[i];
                }
            }

            int res = indices[n - 1] + mask[n - 1];
            delete[] mask;
            delete[] indices;

            timer().endCpuTimer();
            return res;
        }
    }
}
