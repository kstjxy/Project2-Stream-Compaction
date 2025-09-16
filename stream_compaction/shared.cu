#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "common.h"
#include "shared.h"

namespace StreamCompaction {
    namespace Shared {

        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer() {
            static PerformanceTimer t;
            return t;
        }


#ifndef LOG_NUM_BANKS
#define LOG_NUM_BANKS 5 
#endif
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)


#ifndef SHARED_BLOCK_SIZE
#define SHARED_BLOCK_SIZE 256
#endif

// Elements per block (each thread loads 2)
        static __host__ __device__ inline int elemsPerBlock() {
            return SHARED_BLOCK_SIZE * 2;
        }

        __global__ void kernScanBlockSharedExclusive(int n,
            int* __restrict__ odata,
            const int* __restrict__ idata,
            int* __restrict__ blockSums) {
            extern __shared__ int temp[];

            const int thid = threadIdx.x;
            const int n2 = blockDim.x * 2;
            const int start = 2 * blockDim.x * blockIdx.x;

            // Global indices
            const int gi = start + thid;
            const int gj = start + thid + blockDim.x;

            // Shared indices
            const int ai = thid;
            const int bi = thid + blockDim.x;
            const int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
            const int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

            temp[ai + bankOffsetA] = (gi < n) ? idata[gi] : 0;
            temp[bi + bankOffsetB] = (gj < n) ? idata[gj] : 0;
            __syncthreads();

            // Up-sweep
            int offset = 1;
            for (int d = n2 >> 1; d > 0; d >>= 1) {
                __syncthreads();
                if (thid < d) {
                    int ai2 = offset * (2 * thid + 1) - 1;
                    int bi2 = offset * (2 * thid + 2) - 1;
                    ai2 += CONFLICT_FREE_OFFSET(ai2);
                    bi2 += CONFLICT_FREE_OFFSET(bi2);
                    temp[bi2] += temp[ai2];
                }
                offset <<= 1;
            }

            if (thid == 0) {
                int rootIdx = n2 - 1 + CONFLICT_FREE_OFFSET(n2 - 1);
                if (blockSums) {
                    blockSums[blockIdx.x] = temp[rootIdx];
                }
                temp[rootIdx] = 0;
            }

            // Down-sweep
            for (int d = 1; d < n2; d <<= 1) {
                offset >>= 1;
                __syncthreads();
                if (thid < d) {
                    int ai2 = offset * (2 * thid + 1) - 1;
                    int bi2 = offset * (2 * thid + 2) - 1;
                    ai2 += CONFLICT_FREE_OFFSET(ai2);
                    bi2 += CONFLICT_FREE_OFFSET(bi2);
                    int t = temp[ai2];
                    temp[ai2] = temp[bi2];
                    temp[bi2] += t;
                }
            }
            __syncthreads();

            if (gi < n) odata[gi] = temp[ai + bankOffsetA];
            if (gj < n) odata[gj] = temp[bi + bankOffsetB];
        }

        __global__ void kernAddBlockIncrements(int n, int* __restrict__ odata,
            const int* __restrict__ blockIncr) {
            const int thid = threadIdx.x;
            const int base = 2 * blockDim.x * blockIdx.x;
            const int i = base + thid;
            const int j = base + thid + blockDim.x;

            const int incr = blockIncr[blockIdx.x];
            if (i < n) odata[i] += incr;
            if (j < n) odata[j] += incr;
        }

        static void scanLevel(int n, int* d_out, const int* d_in,
            int*& d_blockSums, int& numBlocks) {
            const int BLOCK = SHARED_BLOCK_SIZE;
            const int ELEMS = elemsPerBlock();
            numBlocks = (n + ELEMS - 1) / ELEMS;

            cudaMalloc(&d_blockSums, std::max(1, numBlocks) * sizeof(int));
            checkCUDAError("cudaMalloc d_blockSums");

            dim3 block(BLOCK);
            dim3 grid(numBlocks);

            const int n2 = ELEMS;
            const int smemInts = n2 + (n2 >> LOG_NUM_BANKS);
            const size_t smemBytes = smemInts * sizeof(int);

            kernScanBlockSharedExclusive <<<grid, block, smemBytes >>> (n, d_out, d_in, d_blockSums);
            checkCUDAError("kernScanBlockSharedExclusive");
        }

        static void recurseScan(int n, int* d_out, const int* d_in) {
            if (n <= 0) return;

            int* d_blockSums = nullptr;
            int numBlocks = 0;

            scanLevel(n, d_out, d_in, d_blockSums, numBlocks);

            if (numBlocks > 1) {
                // Recursively scan
                int* d_blockIncr = nullptr;
                cudaMalloc(&d_blockIncr, numBlocks * sizeof(int));
                checkCUDAError("cudaMalloc d_blockIncr");

                recurseScan(numBlocks, d_blockIncr, d_blockSums);

                const int BLOCK = SHARED_BLOCK_SIZE;
                dim3 block(BLOCK);
                dim3 grid(numBlocks);
                kernAddBlockIncrements <<<grid, block >>> (n, d_out, d_blockIncr);
                checkCUDAError("kernAddBlockIncrements");

                cudaFree(d_blockIncr);
            }

            cudaFree(d_blockSums);
        }


        void scan(int n, int* odata, const int* idata) {
            if (n <= 0) return;

            int* d_in = nullptr, * d_out = nullptr;

            cudaMalloc(&d_in, n * sizeof(int));
            cudaMalloc(&d_out, n * sizeof(int));
            checkCUDAError("cudaMalloc shared scan buffers");

            cudaMemcpy(d_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("H2D idata");

            timer().startGpuTimer();
            recurseScan(n, d_out, d_in);
            timer().endGpuTimer();

            cudaMemcpy(odata, d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("D2H odata");

            cudaFree(d_in);
            cudaFree(d_out);
        }

    }
}
