#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() { static PerformanceTimer t; return t; }

        __global__ void kernUpSweep(int nPow2, int d, int numOps, int* data) {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= (unsigned int)numOps) return;

            unsigned int stride = 1u << (d + 1);
            unsigned int bi = i * stride + (stride - 1u);
            unsigned int ai = bi - (1u << d);
            data[bi] += data[ai];
        }

        __global__ void kernDownSweep(int nPow2, int d, int numOps, int* data) {
            unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= (unsigned int)numOps) return;

            unsigned int stride = 1u << (d + 1);
            unsigned int bi = i * stride + (stride - 1u);
            unsigned int ai = bi - (1u << d);

            int t = data[ai];
            data[ai] = data[bi];
            data[bi] += t;
        }

        static void scanInPlace(int* devData, int nPow2) {
            if (nPow2 <= 0) return;

            const int BLOCK_SIZE = 128;
            const int levels = ilog2ceil(nPow2);

            // Up-sweep
            for (int d = 0; d < levels; ++d) {
                int numOps = nPow2 >> (d + 1);
                dim3 block(BLOCK_SIZE);
                dim3 grid((numOps + BLOCK_SIZE - 1) / BLOCK_SIZE);

                kernUpSweep <<<grid, block >>> (nPow2, d, numOps, devData);
                cudaDeviceSynchronize();
                checkCUDAError("kernUpSweep");
            }

            cudaMemset(devData + (nPow2 - 1), 0, sizeof(int));
            checkCUDAError("cudaMemset root");

            // Down-sweep
            for (int d = levels - 1; d >= 0; --d) {
                int numOps = nPow2 >> (d + 1);
                dim3 block(BLOCK_SIZE);
                dim3 grid((numOps + BLOCK_SIZE - 1) / BLOCK_SIZE);

                kernDownSweep <<<grid, block >>> (nPow2, d, numOps, devData);
                cudaDeviceSynchronize();
                checkCUDAError("kernDownSweep");
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            if (n <= 0) return;

            const int nPow2 = 1 << ilog2ceil(n);

            int* devData = nullptr;
            cudaMalloc(&devData, nPow2 * sizeof(int));
            checkCUDAError("cudaMalloc devData");
            cudaMemset(devData, 0, nPow2 * sizeof(int));
            checkCUDAError("cudaMemset devData");

            cudaMemcpy(devData, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("H2D idata");

            timer().startGpuTimer();
            scanInPlace(devData, nPow2);
            timer().endGpuTimer();

            cudaMemcpy(odata, devData, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("D2H odata");
            cudaFree(devData);
        }

        /**
         * Work-efficient compaction using the same scanInPlace.
         */
        int compact(int n, int* odata, const int* idata) {
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

            // Map -> bools
            StreamCompaction::Common::kernMapToBoolean <<<grid, block >>> (n, devBools, devIdata);
            cudaDeviceSynchronize();
            checkCUDAError("kernMapToBoolean");

            // Scan bools (exclusive) -> indices
            int nPow2 = 1 << ilog2ceil(n);
            cudaMalloc(&devIndices, nPow2 * sizeof(int));
            cudaMemset(devIndices, 0, nPow2 * sizeof(int));
            cudaMemcpy(devIndices, devBools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            scanInPlace(devIndices, nPow2);

            // Scatter
            StreamCompaction::Common::kernScatter <<<grid, block >>> (n, devOdata, devIdata, devBools, devIndices);
            cudaDeviceSynchronize();
            checkCUDAError("kernScatter");

            timer().endGpuTimer();

            int lastIdx = 0, lastFlag = 0;
            cudaMemcpy(&lastIdx, devIndices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastFlag, devBools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            int count = lastIdx + lastFlag;
            if (count > 0) {
                cudaMemcpy(odata, devOdata, count * sizeof(int), cudaMemcpyDeviceToHost);
            }

            cudaFree(devIdata);
            cudaFree(devBools);
            cudaFree(devIndices);
            cudaFree(devOdata);
            return count;
        }
    }
}
