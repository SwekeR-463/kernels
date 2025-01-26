#include <iostream>

__global__ void helloCUDA()
{
    printf("Hello World from GPU!\n");
}

int main()
{
    helloCUDA<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}