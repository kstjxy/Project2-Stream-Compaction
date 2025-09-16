#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__

        __global__ void kernNaiveScanStep(int n, int offset, int* odata, const int* idata) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;

            int v = idata[i];
            if (i >= offset) {
                v += idata[i - offset];
            }
            odata[i] = v;   
        }

        __global__ void kernInclusiveToExclusive(int n, int* odata, const int* idata) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;

            odata[i] = (i == 0) ? 0 : idata[i - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) return;

            int* devA = nullptr; // input
            int* devB = nullptr; // output
            size_t bytes = n * sizeof(int);
            cudaMalloc((void**)&devA, bytes);
            checkCUDAError("cudaMalloc devA failed");
            cudaMalloc((void**)&devB, bytes);
            checkCUDAError("cudaMalloc devB failed");

            // Copy input to device
            cudaMemcpy(devA, idata, bytes, cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy failed");

            const int BLOCK_SIZE = 128;
            dim3 block(BLOCK_SIZE);
            dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int levels = ilog2ceil(n);
            timer().startGpuTimer();
            for (int d = 1; d <= levels; ++d) {
                int offset = 1 << (d - 1);
                kernNaiveScanStep <<<grid, block >>> (n, offset, devB, devA);
                checkCUDAError("kernNaiveScanStep failed");
                std::swap(devA, devB);
            }

            // Convert to inclusive to exclusive
            kernInclusiveToExclusive <<<grid, block >>> (n, devB, devA);
            checkCUDAError("kernInclusiveToExclusive failed");
            timer().endGpuTimer();

            cudaMemcpy(odata, devB, bytes, cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy failed");

            cudaFree(devA);
            cudaFree(devB);
        }
    }
}
