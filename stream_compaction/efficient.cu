#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int nPow2, int d, int* data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = 1 << (d + 1);
            int bi = i * stride + (stride - 1);
            if (bi >= nPow2) return;

            int ai = bi - (1 << d);
            data[bi] += data[ai];
        }

        __global__ void kernDownSweep(int nPow2, int d, int* data) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int stride = 1 << (d + 1);
            int bi = i * stride + (stride - 1);
            if (bi >= nPow2) return;

            int ai = bi - (1 << d);
            int t = data[ai];
            data[ai] = data[bi];
            data[bi] += t;
        }

        static void scanInPlace(int* devData, int nPow2) {
            if (nPow2 <= 0) return;

            const int BLOCK_SIZE = 128;
            int levels = ilog2ceil(nPow2);  // = log2(nPow2) since nPow2 is power-of-two

            // Up-sweep
            for (int d = 0; d < levels; ++d) {
                int numOps = nPow2 >> (d + 1);
                dim3 block(BLOCK_SIZE);
                dim3 grid((numOps + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweep << <grid, block >> > (nPow2, d, devData);
                checkCUDAError("kernUpSweep");
            }

            // Set root to 0 (exclusive) WITHOUT another kernel
            // (only two helper kernels are allowed)
            cudaMemset(devData + (nPow2 - 1), 0, sizeof(int));
            checkCUDAError("cudaMemset root");

            // Down-sweep
            for (int d = levels - 1; d >= 0; --d) {
                int numOps = nPow2 >> (d + 1);
                dim3 block(BLOCK_SIZE);
                dim3 grid((numOps + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernDownSweep <<<grid, block >>> (nPow2, d, devData);
                checkCUDAError("kernDownSweep");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) return;

            int nPow2 = 1 << ilog2ceil(n);

            int* devData = nullptr;
            cudaMalloc(&devData, nPow2 * sizeof(int));
            checkCUDAError("cudaMalloc");
            cudaMemset(devData, 0, nPow2 * sizeof(int));
            checkCUDAError("cudaMemset");

            cudaMemcpy(devData, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            scanInPlace(devData, nPow2); // exclusive scan in place
            timer().endGpuTimer();

            cudaMemcpy(odata, devData, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(devData);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            if (n <= 0) return 0;

            const int BLOCK_SIZE = 128;
            dim3 block(BLOCK_SIZE);
            dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* devIdata = nullptr, * devBools = nullptr, * devIndices = nullptr, * devOdata = nullptr;

            cudaMalloc(&devIdata, n * sizeof(int));
            cudaMalloc(&devBools, n * sizeof(int));
            cudaMalloc(&devOdata, n * sizeof(int));
            checkCUDAError("cudaMalloc inputs");
            cudaMemcpy(devIdata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            // Map
            StreamCompaction::Common::kernMapToBoolean <<<grid, block >>> (n, devBools, devIdata);
            checkCUDAError("kernMapToBoolean");

            // Exclusive scan
            int nPow2 = 1 << ilog2ceil(n);
            cudaMalloc(&devIndices, nPow2 * sizeof(int));
            checkCUDAError("cudaMalloc devIndices");
            cudaMemset(devIndices, 0, nPow2 * sizeof(int));
            checkCUDAError("cudaMemset devIndices");
            cudaMemcpy(devIndices, devBools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("D2D bools->indices");

            scanInPlace(devIndices, nPow2);

            // Scatter
            StreamCompaction::Common::kernScatter <<<grid, block >>> (n, devOdata, devIdata, devBools, devIndices);
            checkCUDAError("kernScatter");

            timer().endGpuTimer();

            int lastIdx = 0, lastFlag = 0;
            cudaMemcpy(&lastIdx, devIndices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastFlag, devBools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("D2H count");
            int count = lastIdx + lastFlag;

            if (count > 0) {
                cudaMemcpy(odata, devOdata, count * sizeof(int), cudaMemcpyDeviceToHost);
                checkCUDAError("D2H odata");
            }

            cudaFree(devIdata);
            cudaFree(devBools);
            cudaFree(devIndices);
            cudaFree(devOdata);

            return count;
        }
    }
}
