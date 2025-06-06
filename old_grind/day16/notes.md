### To-Do's
- [x] Text Histogram Kernel
- [x] Ch10 PMPP

### Notes
* An intuitive approach is to inverse the matrix such that X=A-1*(-Y). This technique can be used for matrices of moderate size through methods such as Gaussian elimination
* the conjugate gradient method can be used to iteratively solve the corresponding linear system with guaranteed convergence to a solution, the conjugate gradient methods predicts a solution for X and performs AxX+Y. If the result is not close to a 0 vector, a gradient vector formula can be used to refine the predicted X and another iteration of AxX+Y performed
* CSR completely removes all zero elements from the storage. It incurs storage overhead by introducing the col_index and row_ptr arrays
* in a sparse matrix where only 1% of the elements are nonzero values, the total storage for the CSR representation, including the overhead, would be around 2% of the space required to store both zero and nonzero elements
* The problems of noncoalesced memory accesses and control divergence can be addressed by applying data padding and transposition on the sparse matrix data,these ideas were used in the ELL storage format, whose name came from the sparse matrix package in ELLPACK, a package for solving elliptic boundary value problems
* by arranging the elements in the column major order,all adjacent threads are now accessing adjacent memory locations, enabling memory coalescing, thereby using memory bandwidth more efficiently
* The CGMA value is essentially 1, limiting the attainable FLOPS to a small fraction of the peak performance