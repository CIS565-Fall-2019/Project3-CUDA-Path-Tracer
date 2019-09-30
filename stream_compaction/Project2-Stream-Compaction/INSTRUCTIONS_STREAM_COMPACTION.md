Project 2 Stream Compaction Instructions
========================

**Summary:** In this project, you'll implement GPU stream compaction in CUDA,
from scratch. This algorithm is widely used, and will be important for
accelerating your path tracer project.

Your stream compaction implementations in this project will simply remove `0`s
from an array of `int`s. In the path tracer, you will remove terminated paths
from an array of rays.

In addition to being useful for your path tracer, this project is meant to
reorient your algorithmic thinking to the way of the GPU. On GPUs, many
algorithms can benefit from massive parallelism and, in particular, data
parallelism: executing the same code many times simultaneously with different
data.

You'll implement a few different versions of the *Scan* (*Prefix Sum*)
algorithm. First, you'll implement a CPU version of the algorithm to reinforce
your understanding. Then, you'll write a few GPU implementations: "naive" and
"work-efficient." Finally, you'll use some of these to implement GPU stream
compaction.

**Algorithm overview & details:** There are two primary references for details
on the implementation of scan and stream compaction.

* The [slides on Parallel Algorithms](https://docs.google.com/presentation/d/1ETVONA7QDM-WqsEj4qVOGD6Kura5I6E9yqH-7krnwZ0/edit#slide=id.p126)
  for Scan, Stream Compaction, and Work-Efficient Parallel Scan.
* GPU Gems 3, Chapter 39 - [Parallel Prefix Sum (Scan) with CUDA](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html).
    - This online version contains a few small errors (in superscripting, missing braces, bad indentation, etc.) 
    - We maintain a fix for this at [GPU Gem 3 Ch 39 Patch](https://github.com/CIS565-Fall-2017/Project2-Stream-Compaction/blob/master/INSTRUCTION.md#gpu-gem-3-ch-39-patch). If you find more errors in the chapter, welcome to open new pull requests to contribute.
* If you are still unclear after reading the steps, take a look at the last chapter - [Algorithm Examples](https://github.com/CIS565-Fall-2017/Project2-Stream-Compaction/blob/master/INSTRUCTION.md#algorithm-examples).
* [Recitation slides](https://docs.google.com/presentation/d/1daOnWHOjMp1sIqMdVsNnvEU1UYynKcEMARc_W6bGnqE/edit?usp=sharing)

Your GPU stream compaction implementation will live inside of the
`stream_compaction` subproject. In this way, you will be able to easily copy it
over for use in your GPU path tracer.


## Part 0: The Usual

This project (and all other CUDA projects in this course) requires an NVIDIA
graphics card with CUDA capability. Any card with Compute Capability 2.0
(`sm_20`) or greater will work. Check your GPU on this
[compatibility table](https://developer.nvidia.com/cuda-gpus).
If you do not have a personal machine with these specs, you may use those
computers in SIG Lab which have supported GPUs.

### Useful existing code

* `stream_compaction/common.h`
  * `checkCUDAError` macro: checks for CUDA errors and exits if there were any.
  * `ilog2ceil(x)`: computes the ceiling of log2(x), as an integer.
* `main.cpp`
  * Some testing code for your implementations.

**Note 1:** The tests will simply compare against your CPU implementation
Do it first!

**Note 2:** The tests default to an array of size 256.
Test with something larger (10,000? 1,000,000?), too!


## Part 1: CPU Scan & Stream Compaction

This stream compaction method will remove `0`s from an array of `int`s.

Do this first, and double check the output! It will be used as the expected
value for the other tests.

In `stream_compaction/cpu.cu`, implement:

* `StreamCompaction::CPU::scan`: compute an exclusive prefix sum. For performance comparison, this is supposed to be a simple `for` loop. But for better understanding before starting moving to GPU, you can simulate the GPU scan in this function first.
* `StreamCompaction::CPU::compactWithoutScan`: stream compaction without using
  the `scan` function.
* `StreamCompaction::CPU::compactWithScan`: stream compaction using the `scan`
  function. Map the input array to an array of 0s and 1s, scan it, and use
  scatter to produce the output. You will need a **CPU** scatter implementation
  for this (see slides or GPU Gems chapter for an explanation).

These implementations should only be a few lines long.


## Part 2: Naive GPU Scan Algorithm

In `stream_compaction/naive.cu`, implement `StreamCompaction::Naive::scan`

This uses the "Naive" algorithm from GPU Gems 3, Section 39.2.1. Example 39-1 uses shared memory. This is not required in this project. You can simply use global memory. As a result of this, you will have to do `ilog2ceil(n)` separate kernel invocations.

Since your individual GPU threads are not guaranteed to run simultaneously, you
can't generally operate on an array in-place on the GPU; it will cause race
conditions. Instead, create two device arrays. Swap them at each iteration:
read from A and write to B, read from B and write to A, and so on.

Beware of errors in Example 39-1 in the chapter; both the pseudocode and the CUDA
code in the online version of Chapter 39 are known to have a few small errors
(in superscripting, missing braces, bad indentation, etc.)

Be sure to test non-power-of-two-sized arrays.

## Part 3: Work-Efficient GPU Scan & Stream Compaction

### 3.1. Scan

In `stream_compaction/efficient.cu`, implement
`StreamCompaction::Efficient::scan`

Most of the text in Part 2 applies.

* This uses the "Work-Efficient" algorithm from GPU Gems 3, Section 39.2.2.
* This can be done in place - it doesn't suffer from the race conditions of
  the naive method, since there won't be a case where one thread writes to
  and another thread reads from the same location in the array.
* Beware of errors in Example 39-2.
* Test non-power-of-two-sized arrays.

Since the work-efficient scan operates on a binary tree structure, it works
best with arrays with power-of-two length. Make sure your implementation works
on non-power-of-two sized arrays (see `ilog2ceil`). This requires extra memory
- your intermediate array sizes will need to be rounded to the next power of
two.

### 3.2. Stream Compaction

This stream compaction method will remove `0`s from an array of `int`s.

In `stream_compaction/efficient.cu`, implement
`StreamCompaction::Efficient::compact`

For compaction, you will also need to implement the scatter algorithm presented
in the slides and the GPU Gems chapter.

In `stream_compaction/common.cu`, implement these for use in `compact`:

* `StreamCompaction::Common::kernMapToBoolean`
* `StreamCompaction::Common::kernScatter`


## Part 4: Using Thrust's Implementation

In `stream_compaction/thrust.cu`, implement:

* `StreamCompaction::Thrust::scan`

This should be a very short function which wraps a call to the Thrust library
function `thrust::exclusive_scan(first, last, result)`.

To measure timing, be sure to exclude memory operations by passing
`exclusive_scan` a `thrust::device_vector` (which is already allocated on the
GPU).  You can create a `thrust::device_vector` by creating a
`thrust::host_vector` from the given pointer, then casting it.

For thrust stream compaction, take a look at [thrust::remove_if](https://thrust.github.io/doc/group__stream__compaction.html). It's not required to analyze `thrust::remove_if` but you're encouraged to do so.

## Part 5: Why is My GPU Approach So Slow? (Extra Credit) (+5)

If you implement your efficient scan version following the slides closely, there's a good chance 
that you are getting an "efficient" gpu scan that is actually not that efficient -- it is slower than the cpu approach? 

Though it is totally acceptable for this assignment, 
In addition to explain the reason of this phenomena, you are encouraged to try to upgrade your work-efficient gpu scan. 

Thinking about these may lead you to an aha moment: 
- What is the occupancy at a deeper level in the upper/down sweep? Are most threads actually working?
- Are you always launching the same number of blocks throughout each level of the upper/down sweep?
- If some threads are being lazy, can we do an early termination on them? 
- How can I compact the threads? What should I modify to keep the remaining threads still working correctly?

Keep in mind this optimization won't need you change a lot of your code structures. 
It's all about some index calculation hacks.

If you don't run into the slower gpu approach. 
Congratulations! You are way ahead and you earn this extra credit automatically. 


## Part 6: Radix Sort (Extra Credit) (+10)

Add an additional module to the `stream_compaction` subproject. Implement radix
sort using one of your scan implementations. Add tests to check its correctness.

## Part 7: GPU Scan Using Shared Memory && Hardware Optimization(Extra Credit) (+10)

Implement [GPU Gem Ch 39](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html) Example 39.1, 39.2.

Notice that the size of the shared memory is dynamic and related to the block size. Since each SM has limited shared memory, the block size you set will affect the occupancy of the blocks in each SM. For example, let's say your graphics card has N Kb of shared memory per SM, if you use the maximum of N Kb shared memory per block, then you would have a max occupancy of 1 block per SM. This might not be the best performance.

Besides we can optimize the efficiency by changing our memory access pattern to avoid bank conflicts. See [GPU Gem Ch 39](https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch39.html) Section 39.2.3. This hasn't been covered in the course but we encourage you to challenge yourself.

## Write-up

1. Update all of the TODOs at the top of your `README.md`.
2. Add a description of this project including a list of its features.
3. Add your performance analysis (see below).

All extra credit features must be documented in your `README.md`, explaining its
value (with performance comparison, if applicable!) and showing an example how
it works. For radix sort, show how it is called and an example of its output.

Always profile with Release mode builds and run without debugging.

### Questions

* Roughly optimize the block sizes of each of your implementations for minimal
  run time on your GPU.
  * (You shouldn't compare unoptimized implementations to each other!)

* Compare all of these GPU Scan implementations (Naive, Work-Efficient, and
  Thrust) to the serial CPU version of Scan. Plot a graph of the comparison
  (with array size on the independent axis).
  * We wrapped up both CPU and GPU timing functions as a performance timer class for you to conveniently measure the time cost.
    * We use `std::chrono` to provide CPU high-precision timing and CUDA event to measure the CUDA performance.
    * For CPU, put your CPU code between `timer().startCpuTimer()` and `timer().endCpuTimer()`.
    * For GPU, put your CUDA code between `timer().startGpuTimer()` and `timer().endGpuTimer()`. Be sure **not** to include any *initial/final* memory operations (`cudaMalloc`, `cudaMemcpy`) in your performance measurements, for comparability.
    * Don't mix up `CpuTimer` and `GpuTimer`.
  * To guess at what might be happening inside the Thrust implementation (e.g.
    allocation, memory copy), take a look at the Nsight timeline for its
    execution. Your analysis here doesn't have to be detailed, since you aren't
    even looking at the code for the implementation.

* Write a brief explanation of the phenomena you see here.
  * Can you find the performance bottlenecks? Is it memory I/O? Computation? Is
    it different for each implementation?

* Paste the output of the test program into a triple-backtick block in your
  README.
  * If you add your own tests (e.g. for radix sort or to test additional corner
    cases), be sure to mention it explicitly.

These questions should help guide you in performance analysis on future
assignments, as well.

## GPU Gem 3 Ch 39 Patch

* Example 1
![](img/example-1.png)

* Example 2
![](img/example-2.jpg)

* Figure-39-4
![](img/figure-39-4.jpg)

* Figure-39-2. This image shows an naive inclusive scan. We should convert this to an exclusive one for compaction.
![](img/figure-39-2.jpg)

## Algorithm Examples

* scan: 
  - goal: produce a prefix sum array of a given array (we only care about exclusive scan here)
  - input
    - [1 5 0 1 2 0 3]
  - output
    - [0 1 6 6 7 9 9]
* compact: 
  - goal: closely and neatly packed the elements != 0
  - input
    - [1 5 0 1 2 0 3]
  - output
    - [1 5 1 2 3]
* compactWithoutScan (CPU)
  - an implementation of compact. So the goal, input and output should all be the same as compact
  - Simply loop through the input array, meanwhile maintain a pointer indicating which address shall we put the next non-zero element
* compactWithScan (CPU/GPU)
  - an implementation of compact. So the goal, input and output should all be the same as compact
  - 3 steps
    - map
      + goal: map our original data array (integer, Light Ray, etc) to a bool array
      + input
        - [1 5 0 1 2 0 3]
      + output
        - [1 1 0 1 1 0 1]
    - scan
        + take the output of last step as input
        + input
          - [1 1 0 1 1 0 1]
        + output
          - [0 1 2 2 3 4 4]
    - scatter
        + preserve non-zero elements and compact them into a new array
        + input:
          + original array
            - [1 5 0 1 2 0 3]
          + mapped array
            - [1 1 0 1 1 0 1]
          + scanned array
            - [0 1 2 2 3 4 4]
        + output:
          - [1 5 1 2 3]
        + This can be done in parallel on GPU
        + You can try multi-threading on CPU if you want (not required and not our focus)
        + for each element input[i] in original array
          - if it's non-zero (given by mapped array)
          - then put it at output[index], where index = scanned[i]