### To-Do's -
- [x] PMPP CH1
- [x] PMPP CH2
- [x] Improved Vector Addition Kernel
- [x] Vector Multiplication Kernel
- [x] Test the speed against Pytorch Vector Multiplication

---

### Notes

* GPU Architecture -> threads -> grouped into thread blocks -> organized into grids
* All of this operates on streaming multiprocessor with DRAM for storage.
* Amdahl's law is a formula that predicts how much faster a task can be completed when a system's resources are improved. It's used in computer architecture to determine which parts of a system to improve. 
* basis of data parallelism: (re)organize the computation around the data, such that we can execute the resulting independent computations in parallel to complete the overall job faster, often much faster
* Compilation process of a CUDA C Program -> cuda c program -> nvcc compiler -> host code & device code(ptx) -> host c preprocessor,compiler/linker & device jit compiler -> heterogenous computing platform CPU & GPU
* A thread consists of the code of the program,
the particular point in the code that is being executed, and the values of its
variables and data structures.
* d_ for device code, h_ for host code
* cudaMalloc() -> Allocates object in the device
global memory -> Two parameters -> Address of a pointer to the allocated object -> Size of allocated object in terms of bytes
* cudaFree() -> Frees object from device global memory -> Pointer to freed object
* cudaMemcpy() -> Memory data transfer -> Requires four parameters -> Pointer to destination -> Pointer to source -> Number of bytes copied -> Type/Direction of transfer
* the number of threads in a block is available in a built-in blockDim variable
* threadIdx gives each thread an unique coordinate within a block
* blockIdx variable gives all threads in a block a common block coordinate
