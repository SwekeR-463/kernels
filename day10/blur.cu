#include <iostream>
#include <cuda_runtime.h>

#define BLUR_SIZE 1  
#define BLOCK_SIZE 16  

// blur kernel
__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    if (Col < w && Row < h) {
        int pixVal = 0;
        int pixels = 0;

        // get the average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int curRow = Row + blurRow;
                int curCol = Col + blurCol;

                if (curRow > -1 && curRow < h && curCol > -1 && curCol < w) {
                    pixVal += in[curRow * w + curCol];
                    pixels++;
                }
            }
        }

        // write the new pixel value to the output image
        out[Row * w + Col] = (unsigned char)(pixVal / pixels);
    }
}

int main() {
    int w = 1024;  
    int h = 1024;  

    unsigned char *h_in = (unsigned char *)malloc(w * h * sizeof(unsigned char));
    unsigned char *h_out = (unsigned char *)malloc(w * h * sizeof(unsigned char));

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            h_in[i * w + j] = (i + j) % 256;
        }
    }

    unsigned char *d_in, *d_out;
    cudaMalloc((void **)&d_in, w * h * sizeof(unsigned char));
    cudaMalloc((void **)&d_out, w * h * sizeof(unsigned char));

    cudaMemcpy(d_in, h_in, w * h * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 blocksPerGrid((w + BLOCK_SIZE - 1) / BLOCK_SIZE, (h + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    blurKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, w, h);

    cudaMemcpy(h_out, d_out, w * h * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);

    cudaFree(d_in);
    cudaFree(d_out);

    float time = 0;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // display the output image (example: print first 10x10 pixels)
    std::cout << "Blurred Image (first 10x10 pixels):" << std::endl;
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            std::cout << (int)h_out[i * w + j] << " ";
        }
        std::cout << std::endl;
    }

    free(h_in);
    free(h_out);

    return 0;
}
// Kernel Execution Time: 1.3263ms