### To Do's -
- [x] An Even Easier Introduction to CUDA by [nvidia](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [x] Hello World Kernel
- [x] Simple Addition Kernel
- [x] Vector Addition Kernel

---

### Notes

* Did the setup for CUDA using the freeCodeCamp [video](https://youtu.be/86FAWCzIe_4?si=DjylDt9YqV_6CmuY).
* Made a simple hello world in c++ run using nvcc.
* Then converted the simple hello world to a cuda kernel using `__global__` .
* This was like the entry point to write the kernels.
* Then I go through the blog mentioned and did a simple kernel for vector addition.
*  `__global__` to the function, which tells the CUDA C++ compiler that this is a function that runs on the GPU and can be called from CPU code.
* The `__global__` functions are known as kernels, and code that runs on the GPU is often called device code, while code that runs on the CPU is host code.
* To compute on GPU, we need to allocate the memory accessible by the GPU.
* Unified Memory in CUDA makes this easy by providing a single memory space accessible by all GPUs and CPUs in the system.
* To allocate data in unified memory, call `cudaMallocManaged()`, which returns a pointer that you can access from host (CPU) code or device (GPU) code. To free the data, just pass the pointer to `cudaFree()`.
* CUDA kernel launches are specified using the triple angle bracket syntax <<< >>>.
* The CPU will wait until the kernel is done before it accesses the results (because CUDA kernel launches don’t block the calling CPU thread). To do this, just call `cudaDeviceSynchronize()` before doing the final error checking on the CPU.
* how do you make it parallel? The key is in CUDA’s <<<1, 1>>>syntax. This is called the execution configuration, and it tells the CUDA runtime how many parallel threads to use for the launch on the GPU. There are two parameters here, but let’s start by changing the second one: the number of threads in a thread block. CUDA GPUs run kernels using blocks of threads that are a multiple of 32 in size, so 256 threads is a reasonable size to choose.
* the first parameter of the execution configuration specifies the number of thread blocks. Together, the blocks of parallel threads make up what is known as the grid.

---

### Issue Faced

The main issue I faced was profiling. As the blog is old