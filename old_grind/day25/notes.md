### To-Do's
- [x] CH17 PMPP
- [ ] cuBLAS GEMM Kernel
- [ ] cuBLAS GEMM Strided Batched Kernel

### Notes
* 4 steps decomposition -> problem decomposition, algorithm selection, implementation in a language, and performance tuning
* he grid-centric arrangement has a memory access behavior called gather, where each thread gathers or collects the effect of input atoms into a grid point
* The atom-centric arrangement, on the other hand, exhibits a memory access behavior called scatter, where each thread scatters or distributes the effect of an atom into grid points
* 