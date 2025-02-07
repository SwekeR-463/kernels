### To-Do's
- [ ] CH7
- [ ] Conv Kernel

### Notes
* in high perf computing, conv pattern is referred to as stencil computation, which appears widely in numerical methods for solving differential equations
* the missing elements while making a 1d conv operation for the first element of a given array are referred to as "ghost cells" or "halo cells"
* all applns don't assume that ghost cells contain 0,for eg, some applns might assume that the ghost cells contain the same value as the closest valid data elements
* with boundaries in both the x and y dimensions, there are more complex boundary conditions: the calculation of an output element may involve boundary conditions along a horizontal boundary, a vertical boundary, or both
* ratio of floating-point arithmetic calculation to global memory accesses is only about 1.0 in the kernel
* like global memory variables, constant memory variables are also visible to all thread blocks. The main difference is that a constant memory variable cannot be changed by threads during kernel execution, furthermore, the size of the constant memory is quite small, currently at 64KB
* `cudaMemcpytoSymbol` is a special memory copy func that informs the CUDA runtime that the data being copied into the constant memory will not be changes during the kernel executiom
* in order to mitigate the effect of memory bottleneck, modern processors commonly employ on-chip cache memories, or caches, to reduce the number of variables that need to be accessed from the main memory (DRAM)
* 