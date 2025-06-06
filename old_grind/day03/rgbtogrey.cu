#include<iostream>
#include<cuda_runtime.h>

#define CHANNELS 3

// rgb to greyscale kernel
__global__ void colorToGreyScale(unsigned char *Pout, unsigned char *Pin, int width, int height) {

    int Col = threadIdx.x + blockIdx.x + blockDim.x;
    int Row = threadIdx.y + blockIdx.y + blockDim.y;

    if (Col < width && Row < height) {
        // for 1D of greyscale image
        int greyOffset = Row * width + Col;

        int rgbOffset = greyOffset + CHANNELS;

        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];

        Pout[greyOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

int main() {
    int width = 1920;
    int height = 1080;

    size_t rgbSize = width * height * CHANNELS * sizeof(unsigned char);
    size_t graySize = width * height * sizeof(unsigned char);

    unsigned char *h_rgbImage = (unsigned char *)malloc(rgbSize);
    unsigned char *h_grayImage = (unsigned char *)malloc(graySize);

    // initialize the image
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int offset = (i * width + j) * CHANNELS;
            h_rgbImage[offset] = (unsigned char)(j % 256);       // red
            h_rgbImage[offset + 1] = (unsigned char)((i + j) % 256); // green
            h_rgbImage[offset + 2] = (unsigned char)(i % 256);   // blue
        }
    }

    float time = 0.0f;

    unsigned char *d_rgbImage, *d_grayImage;
    cudaMalloc((void **)&d_rgbImage, rgbSize);
    cudaMalloc((void **)&d_grayImage, graySize);

    cudaMemcpy(d_rgbImage, h_rgbImage, rgbSize, cudaMemcpyHostToDevice);

    dim3 blockDim(32, 32);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    colorToGreyScale<<<gridDim, blockDim>>>(d_grayImage, d_rgbImage, width, height);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(h_grayImage, d_grayImage, graySize, cudaMemcpyDeviceToHost);

    std::cout << "Kernel Execution Time: " << time << "ms" << std::endl;

    cudaFree(d_rgbImage);
    cudaFree(d_grayImage);

    free(h_rgbImage);
    free(h_grayImage);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
// Kernel Execution Time: 0.188224ms