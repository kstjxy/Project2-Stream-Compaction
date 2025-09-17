CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Author: Crystal Jin
  *  [LinkedIn](https://www.linkedin.com/in/xiaoyue-jin), [personal website](https://xiaoyuejin.com)

* Tested on: Windows 11, i7-14700K @ 3.40GHz, 64GB RAM, NVIDIA GeForce RTX 4080 SUPER

## Assignment Overview

I implemented GPU stream compaction in CUDA and analyzed its performance. Stream compaction filters an input array into elements that meet a criterion, which requires an efficient exclusive prefix-sum (scan); I therefore implemented and compared multiple scan variants alongside the compaction. My GPU algorithms follow GPU Gems 3, Chapter 39, with a CPU baseline for correctness and timing comparisons.


### Features Implemented
- **CPU baseline:** exclusive scan and two compaction paths (with and without scan).
- **Naive CUDA scan:** multi-pass, global-memory, O(n log n).
- **Work-efficient CUDA scan:** Blelloch up-/down-sweep with NPOT padding.
- **Thrust wrappers:** device-vector timing with `exclusive_scan`.
#### Extra credits:
- **Optimized efficient scan:** fewer idle threads and safer index math.
- **Radix sort (LSD):** built on scan + scatter.
- **Shared-memory scan:** bank-conflict-free with hierarchical block sums.

---

### CPU Scan & Compaction (`cpu.cu`)
A simple, correct CPU baseline that defines expected outputs and validates all GPU results. Implements exclusive prefix sum in O(n), a straight-line compaction (`compactWithoutScan`), and a scan-based compaction (`compactWithScan`: mask → exclusive scan → scatter). This module is the oracle for tests and a performance reference.

### Naive GPU Scan (`naive.cu`)
A textbook O(n log n) CUDA scan that ping-pongs between two global arrays across `log₂ n` passes. For each distance `2^d`, a kernel reads from buffer A and writes to B, then swaps; after building an inclusive scan it converts to exclusive with a final shift. Works for NPOT by guarding indices; easy to understand but launch- and bandwidth-heavy.

### Work-Efficient GPU Scan (original `efficient.cu`)
Up-sweep/down-sweep exclusive scan with padding to the next power of two. Builds a reduction tree (up-sweep), zeroes the root for exclusivity, then propagates partials back down (down-sweep). Uses global memory and launches two kernels per level; correct and O(n) work overall, but can under-utilize the GPU at deep levels.

### Thrust Scan & Compaction (`thrust.cu`)
Thin wrappers around Thrust for fast, well-tuned device scans and optional compaction. Copies inputs into `thrust::device_vector`s (outside the timer) and times only `thrust::exclusive_scan`; for compaction, `thrust::remove_if` can drop zeros. Serves as an upper-bound baseline and sanity check.

### Extra Credit 0 — Faster GPU Optimization (revised `efficient.cu`)
**Why slow:** low occupancy at deep levels (`numOps` shrinks), many launches + strided global accesses, unnecessary index work for idle threads, and signed-index overflow risk at large `n`.  
**What I changed:** pass `numOps` to kernels and early-return (`k ≥ numOps`) to compact work; use `unsigned` for `stride/bi/ai` to avoid overflow.

### Extra Credit 1 — Radix Sort (`radix.h`/`radix.cu`)
Stable 32-bit LSD radix sort using scan as the primitive for partitioning each bit. For each bit: build flags (`b`, `e = !b`), exclusive-scan `e` to get positions, compute `totalFalse`, then scatter zeros before ones; swap buffers and repeat 32 times. Handles NPOT via padded scan buffer and reuses the work-efficient scan kernels. **Test cases** in `main.cpp` (POT & NPOT) compare against `std::sort` and print CUDA timings.

### Extra Credit 2 — Shared Memory & Hardware Optimization (`shared/shared.cu`)
Bank-conflict-free shared-memory scan with hierarchical block sums. Each block scans `2·BLOCK_SIZE` items in `__shared__` using the `CONFLICT_FREE_OFFSET(index >> LOG_NUM_BANKS)` padding trick from GPU Gems §39.2.3, writes per-block sums, recursively scans those sums, and adds them back. Minimizes global traffic, avoids bank conflicts, and improves throughput. **Test cases** (POT & NPOT) validate against CPU scan and report timings.

---

## Performance Analysis

### Block-Size Optimization (N = 2^24, time in ms)
I tuned the CUDA block size for each scan implementation to minimize runtime on my GPU.

| Scan Method            | 32     | 64     | 128    | 256    | 512    | 1024   |
|------------------------|--------|--------|--------|--------|--------|--------|
| Naive                  | 7.4317 | 6.3119 | 6.2064 | 6.4394 | 6.1582 | **5.9467** |
| Efficient (Optimized)  | 2.1963 | 1.9826 | 1.6954 | 1.7118 | **1.6691** | 2.2754 |
| Shared Memory          | 1.4097 | 1.0900 | 1.1217 | **1.0258** | 1.1377 | 1.3383 |

<img width="560" height="360" alt="block_size_sweep" src="https://github.com/user-attachments/assets/abf5ef5e-fc25-4284-9249-9926f5284696" />


**Highlights:** Naive best @ **1024** (5.95 ms), Efficient best @ **512** (1.67 ms), Shared best @ **256** (1.03 ms). Shared-memory is fastest overall at this size; each method has a different sweet spot.

### GPU vs CPU (block size = 128, time in ms)
| Method \ 2^k | 19     | 20     | 21      | 22      | 23      | 24      |
|--------------|--------|--------|---------|---------|---------|---------|
| **CPU**      | 0.1598 | 0.3345 | 0.6606  | 1.3902  | 2.7342  | 5.4556  |
| Naive        | 0.2708 | 0.3750 | 0.5072  | 0.6759  | 1.9117  | 5.7871  |
| Efficient (Opt) | 0.4294 | 0.5540 | 0.6564  | 0.6767  | 1.0208  | 1.7050  |
| Shared Memory| 0.2205 | 0.4818 | 0.4972  | 0.5519  | 0.7117  | 1.1824  |
| Thrust       | 0.4127 | 0.4439 | 0.5438  | 0.5870  | 0.4936  | 0.8315  |

<img width="560" height="360" alt="scan_comparison" src="https://github.com/user-attachments/assets/c43d6a7c-9b3a-4502-ac6d-44d2fb832af9" />


**Cross-over:** all GPU methods beat CPU from **k = 21** (N = 2^21).  
**Fastest by size:** k=19–20: **CPU**; k=21–22: **Shared**; k=23–24: **Thrust**.  
**Speedup at N = 2^24:** Naive **0.94×**, Efficient **3.20×**, Shared **4.61×**, Thrust **6.56×**.

### Performance Bottlenecks
Mostly **memory bandwidth** limited. Naive is bandwidth + launch bound (many passes); work-efficient is bandwidth bound with low occupancy at deep levels (optimized version reduces idle work); shared-memory minimizes global traffic but still hits bandwidth ceiling; Thrust runs near the memory roofline. For small N, GPU variants are launch/overhead bound, so CPU can win.

---

## Output of Test Program

Tested with block size = 128 and array size = 2^24.  
Includes self-added test cases for Shared-Memory + Hardware Optimization and Radix Sort.

```
****************
** SCAN TESTS **
****************
    [  36  34  17   2  28  27  39  30  39  31   8   5  21 ...  13   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 5.6452ms    (std::chrono Measured)
    [   0  36  70  87  89 117 144 183 213 252 283 291 296 ... 410924736 410924749 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 5.4766ms    (std::chrono Measured)
    [   0  36  70  87  89 117 144 183 213 252 283 291 296 ... 410924724 410924724 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 6.30118ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 6.06934ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 1.75136ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 1.64736ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.719872ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 0.729088ms    (CUDA Measured)
    passed
==== shared-mem scan (bank-conflict free), power-of-two ====
   elapsed time: 1.08144ms    (CUDA Measured)
    passed
==== shared-mem scan (bank-conflict free), non-power-of-two ====
   elapsed time: 1.0585ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   0   3   2   0   3   1   0   3   3   2   3   3 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 27.4266ms    (std::chrono Measured)
    [   3   2   3   1   3   3   2   3   3   2   1   2   2 ...   3   3 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 27.6656ms    (std::chrono Measured)
    [   3   2   3   1   3   3   2   3   3   2   1   2   2 ...   3   3 ]
    passed
==== cpu compact with scan ====
   elapsed time: 53.2503ms    (std::chrono Measured)
    [   3   2   3   1   3   3   2   3   3   2   1   2   2 ...   3   3 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 4.8008ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 5.58944ms    (CUDA Measured)
    passed

********************
** RADIX SORT TEST **
********************
==== radix sort, power-of-two ====
   elapsed time: 91.9286ms    (CUDA Measured)
    passed
==== radix sort, non-power-of-two ====
   elapsed time: 91.5685ms    (CUDA Measured)
    passed
```
