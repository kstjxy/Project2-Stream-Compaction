#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "common.h"
#include "radix.h"

namespace StreamCompaction {
    namespace Radix {
        using StreamCompaction::Common::PerformanceTimer;

        PerformanceTimer& timer() {
            static PerformanceTimer t;
            return t;
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

        static void blellochScanInPlace(int* devData, int nPow2) {
            if (nPow2 <= 0) return;
            const int BLOCK_SIZE = 128;
            int levels = ilog2ceil(nPow2);

            // up-sweep
            for (int d = 0; d < levels; ++d) {
                int numOps = nPow2 >> (d + 1);
                dim3 block(BLOCK_SIZE);
                dim3 grid((numOps + BLOCK_SIZE - 1) / BLOCK_SIZE);
                kernUpSweep <<<grid, block >>> (nPow2, d, devData);
                checkCUDAError("Radix kernUpSweep");
            }

            cudaMemset(devData + (nPow2 - 1), 0, sizeof(int));
            checkCUDAError("Radix root memset");

            // down-sweep
            for (int d = levels - 1; d >= 0; --d) {
                int numOps = nPow2 >> (d + 1);
                dim3 block(128);
                dim3 grid((numOps + 127) / 128);
                kernDownSweep <<<grid, block >>> (nPow2, d, devData);
                checkCUDAError("Radix kernDownSweep");
            }
        }


        // Compute b (bit flags) and e (= !b) for this bit position.
        __global__ void kernBitFlags(int n, int bit, int* b, int* e, const int* idata) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;
            unsigned int v = static_cast<unsigned int>(idata[i]);
            int flag = (v >> bit) & 1;
            b[i] = flag;
            e[i] = 1 - flag; 
        }

        // Scatter
        __global__ void kernScatterByBit(int n, int totalFalses,
            const int* idata, const int* b, const int* f,
            int* odata) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= n) return;
            int dest = b[i] ? (i - f[i] + totalFalses) : f[i];
            odata[dest] = idata[i];
        }


        void sort(int n, int* odata, const int* idata) {
            if (n <= 0) return;

            const int BLOCK_SIZE = 128;
            dim3 block(BLOCK_SIZE);
            dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

            int* devIn = nullptr;
            int* devOut = nullptr;
            int* devB = nullptr;      // bit flags
            int* devE = nullptr;      // e = !b
            int* devF = nullptr;      // exclusive scan(e)
            cudaMalloc(&devIn, n * sizeof(int));
            cudaMalloc(&devOut, n * sizeof(int));
            cudaMalloc(&devB, n * sizeof(int));
            cudaMalloc(&devE, n * sizeof(int));

            cudaMemcpy(devIn, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // padded scan buffer
            int nPow2 = 1 << ilog2ceil(n);
            cudaMalloc(&devF, nPow2 * sizeof(int));

            timer().startGpuTimer();

            for (int bit = 0; bit < 32; ++bit) {
                kernBitFlags <<<grid, block >>> (n, bit, devB, devE, devIn);

                cudaMemset(devF, 0, nPow2 * sizeof(int));
                cudaMemcpy(devF, devE, n * sizeof(int), cudaMemcpyDeviceToDevice);
                blellochScanInPlace(devF, nPow2);

                int h_lastE = 0, h_lastF = 0;
                cudaMemcpy(&h_lastE, devE + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                cudaMemcpy(&h_lastF, devF + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
                int totalFalses = h_lastE + h_lastF;

                kernScatterByBit <<<grid, block >>> (n, totalFalses, devIn, devB, devF, devOut);

                std::swap(devIn, devOut);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, devIn, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(devIn);
            cudaFree(devOut);
            cudaFree(devB);
            cudaFree(devE);
            cudaFree(devF);
        }
    }
}
