### To-Do's
- [x] CH13 PMPP
- [ ] Quadtree Kernel
- [ ] ReLU Kernel

### Notes
* CUDA Dynamic Parallelism is an extension to the CUDA programming model enabling a CUDA kernel to create new thread grids by launching new kernels
* Dynamic Parallelism -> writing a kernel launch statement inside of a kernel
* The stream must have been allocated in the same thread-block where the call is being made
* A parent thread and its child grid can make their global memory data visible to each other, with weak consistency guarantees between child and parent
* Zero-copy system memory has identical consistency guarantees as global memory, and follows the same semantics as detailed above
* Zero-copy system memory has identical consistency guarantees as global memory, and follows the same semantics as detailed above
* Local memory is private storage for a thread, and is not visible outside of that thread
* Shared memory is private storage for an executing thread-block, and data is not visible outside of that thread-block
* Texture memory accesses (read-only) are performed on a memory region that may be aliased to the global memory region
* L1 cache, or Level 1 cache, is a type of high-speed memory that's built into a CPU. It's the fastest and smallest cache memory, and it's designed to store frequently used instructions and data
* A thread that invokes `cudaDeviceSynchronize()` will wait until all kernels launched by any thread in the thread-block have completed
* If a parent kernel launches other child kernels and does not explicitly synchronize on the completion of those kernels, then the runtime will perform the synchronization implicitly before the parent kernel terminates
* Bezier Curves -> which are frequently used in computer graphics to draw smooth, intuitive curves that are defined by a set of control points, which are typically defined by a user
* which are frequently used in computer graphics to draw smooth, intuitive curves that are defined by a set of control points, which are typically defined by a user
* 