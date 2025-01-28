### To-Do's -
- [x] PMPP Ch3
- [ ] RGB to Grey Kernel
- [ ] Matrix Transpose Kernel

### Notes
* threads are organized into a 2-level hierarchy -> a grid consisting of blocks -> a block consisting of threads
* all CUDA kernels in a grid execute the same kernel function as they rely on coordinates to distinguish themselves from one another & identify the appropriate portion of data to process
* numBlocks(gridDim) & blockDim in a kernel always reflect the dimensions of the grid and the blocks
* the allowed values of gridDim.x, gridDim.y, gridDim.z is from 1 to 65,536
* the blockIdx.x values range from 0 to gridDim.x - 1, same for y & z
* a 2-D matrix can be linearized by -> row-major layout where we place all elements of the same row in consecutive memory locations & column-major layout where we place all elements of the same column
---
* The formula 0.21f * r + 0.71f * g + 0.07f * b is designed to produce a grayscale value that reflects the perceived brightness of the RGB pixel, based on human vision.
* It does not explicitly clamp the output to [0, 255], but the output will naturally fall within this range if the input RGB values are valid.
* It does not simply map dark colors to black and light colors to white; instead, it calculates a weighted average to produce a grayscale value that represents the brightness of the original color.
---
* Blur v/s Average Pooling -> while both operations involve aggregating local pixel information, they serve different purposes and are applied in different contexts. Blurring is more about smoothing, while pooling is about downsampling and feature extraction in CNNs.
---
* the ability to execute the same application code on hardware with different numbers of execution resources is referred to as transparent scalability
* The warp is the unit of thread scheduling in SMs
* If more than one warp is ready for execution, a priority mechanism is used to select one for execution. This mechanism of filling the latency time of operations with work from other threads is often called “latency tolerance” or “latency hiding”