#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
    namespace Thrust {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) return;

            thrust::host_vector<int>    h_in(idata, idata + n);
            thrust::device_vector<int>  d_in = h_in;
            thrust::device_vector<int>  d_out(n);

            timer().startGpuTimer();
            thrust::exclusive_scan(d_in.begin(), d_in.end(), d_out.begin());
            timer().endGpuTimer();

            thrust::copy(d_out.begin(), d_out.end(), odata);
        }
    }
}
