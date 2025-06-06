#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <cuda_runtime.h>

// hyperparameters
#define INPUT_SIZE 784
#define HIDDEN_SIZE 4096
#define OUTPUT_SIZE 10
#define TRAIN_SIZE 10000
#define TEST_SIZE 1000
#define BATCH_SIZE 32
#define EPOCHS 20
#define LEARNING_RATE 0.05

typedef struct {
    float *weights1;
    float *weights2;
    float *bias1;
    float *bias2;
    float *grad_weights1;
    float *grad_weights2;
    float *grad_bias1;
    float *grad_bias2;
} NeuralNetwork;

// cuda check
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// load batched image data
void load_data(const char *filename, float *data, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(data, sizeof(float), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading data: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// load batch labels
void load_labels(const char *filename, int *labels, int size) {
    FILE *file = fopen(filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    size_t read_size = fread(labels, sizeof(int), size, file);
    if (read_size != size) {
        fprintf(stderr, "Error reading labels: expected %d elements, got %zu\n", size, read_size);
        exit(1);
    }
    fclose(file);
}

// initialization for weights
void initialize_weights(float *weights, int size) {
    float scale = sqrtf(2.0f / size);
    for (int i = 0; i < size; i++) {
        weights[i] = ((float)rand() / RAND_MAX) * scale - (scale / 2.0f);
    }
}

// initialization for biases
void initialize_bias(float *bias, int size) {
    for (int i = 0; i < size; i++) {
        bias[i] = 0.0f;
    }
}

// cuda kernel for matrix multiplication (A @ B)
__global__ void matmul_a_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// cuda kernel for matrix multiplication (A @ B.T)
__global__ void matmul_a_bt_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[col * n + i];
        }
        C[row * k + col] = sum;
    }
}

// cuda kernel for matrix multiplication (A.T @ B)
__global__ void matmul_at_b_kernel(float *A, float *B, float *C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < m; ++i) {
            sum += A[i * n + row] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// cuda kernel for GeLU activation
__global__ void gelu_kernel(float *x, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608f * (x_val + 0.044715f * x_val * x_val * x_val)));
        x[idx] = x_val * cdf;
    }
}

// cuda kernel for GeLU derivative
__global__ void dgelu_kernel(float *x, float *d_gelu_out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x_val = x[idx];
        float tanh_term = tanhf(0.7978845608f * (x_val + 0.044715f * x_val * x_val * x_val));
        float sech_term = 1.0f / coshf(0.7978845608f * (x_val + 0.044715f * x_val * x_val * x_val));
        float cdf = 0.5f * (1.0f + tanh_term);
        float pdf = 0.5f * sqrtf(2.0f / M_PI) * expf(-0.5f * x_val * x_val);
        d_gelu_out[idx] = cdf + x_val * pdf;
    }
}

// cuda kernel for bias addition
__global__ void bias_add_kernel(float *x, float *bias, int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / size;
    int i = idx % size;

    if (b < batch_size && i < size) {
        x[idx] += bias[i];
    }
}

// cuda kernel for softmax
__global__ void softmax_kernel(float *x, int batch_size, int size) {
    int b = blockIdx.x;
    if (b < batch_size) {
        float max_val = x[b * size];
        for (int i = 1; i < size; ++i) {
            max_val = fmaxf(max_val, x[b * size + i]);
        }

        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            x[b * size + i] = expf(x[b * size + i] - max_val);
            sum += x[b * size + i];
        }

        for (int i = 0; i < size; ++i) {
            x[b * size + i] = fmaxf(x[b * size + i] / sum, 1e-7f);
        }
    }
}

// forward function for NN 
void forward(NeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int batch_size) {
    dim3 block_size(32, 32);
    dim3 grid_size((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (batch_size + block_size.y - 1) / block_size.y);

    // input to Hidden (X @ W1)
    matmul_a_b_kernel<<<grid_size, block_size>>>(d_input, nn->weights1, d_hidden, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // add bias1
    bias_add_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, nn->bias1, batch_size, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // apply GeLU
    gelu_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // hidden to output (Hidden @ W2)
    grid_size.x = (OUTPUT_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_b_kernel<<<grid_size, block_size>>>(d_hidden, nn->weights2, d_output, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // add bias2
    bias_add_kernel<<<(batch_size * OUTPUT_SIZE + 255) / 256, 256>>>(d_output, nn->bias2, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // apply softmax
    softmax_kernel<<<batch_size, 1>>>(d_output, batch_size, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

// cross-entropy loss function
float cross_entropy_loss(float *output, int *labels, int batch_size) {
    float total_loss = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        total_loss -= logf(fmaxf(output[b * OUTPUT_SIZE + labels[b]], 1e-7f));
    }
    return total_loss / batch_size;
}

// cuda kernel to zero out gradients
__global__ void zero_grad_kernel(float *grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = 0.0f;
    }
}

// cuda kernel for computing output gradients
__global__ void compute_output_gradients_kernel(float *grad_output, float *output, int *labels, int batch_size) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch_size) {
        for (int i = 0; i < OUTPUT_SIZE; ++i) {
            grad_output[b * OUTPUT_SIZE + i] = output[b * OUTPUT_SIZE + i];
        }
        grad_output[b * OUTPUT_SIZE + labels[b]] -= 1.0f;
    }
}

// cuda kernel for updating gradients
__global__ void update_gradients_kernel(float *grad_weights, float *grad_bias, float *grad_layer, float *prev_layer, int batch_size, int prev_size, int curr_size) {
    int i = blockIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < curr_size && j < prev_size) {
        float grad_w_sum = 0.0f;
        for (int b = 0; b < batch_size; ++b) {
            grad_w_sum += grad_layer[b * curr_size + i] * prev_layer[b * prev_size + j];
        }
        atomicAdd(&grad_weights[i * prev_size + j], grad_w_sum);

        if (j == 0) {
            float grad_b_sum = 0.0f;
            for (int b = 0; b < batch_size; ++b) {
                grad_b_sum += grad_layer[b * curr_size + i];
            }
            atomicAdd(&grad_bias[i], grad_b_sum);
        }
    }
}

// element-wise multiplication of gradients
__global__ void multiply_gradients_kernel(float *grad1, float *grad2, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad1[idx] *= grad2[idx];
    }
}

// backward function for NN
void backward(NeuralNetwork *nn, float *d_input, float *d_hidden, float *d_output, int *d_labels, int batch_size) {
    // zero out gradients
    zero_grad_kernel<<<(HIDDEN_SIZE * INPUT_SIZE + 256 - 1) / 256, 256>>>(nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    zero_grad_kernel<<<(OUTPUT_SIZE * HIDDEN_SIZE + 256 - 1) / 256, 256>>>(nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    zero_grad_kernel<<<(HIDDEN_SIZE + 256 - 1) / 256, 256>>>(nn->grad_bias1, HIDDEN_SIZE);
    zero_grad_kernel<<<(OUTPUT_SIZE + 256 - 1) / 256, 256>>>(nn->grad_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaDeviceSynchronize());

    // compute gradients for output layer
    float *d_grad_output;
    CUDA_CHECK(cudaMalloc(&d_grad_output, batch_size * OUTPUT_SIZE * sizeof(float)));
    compute_output_gradients_kernel<<<(batch_size + 255) / 256, 256>>>(d_grad_output, d_output, d_labels, batch_size);
    CUDA_CHECK(cudaGetLastError());

    // update gradients for weights2 (W2.grad = grad_output.T @ hidden)
    dim3 block_size(32, 32);
    dim3 grid_size((HIDDEN_SIZE + block_size.x - 1) / block_size.x, (OUTPUT_SIZE + block_size.y - 1) / block_size.y);
    matmul_at_b_kernel<<<grid_size, block_size>>>(d_hidden, d_grad_output, nn->grad_weights2, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // update gradients for bias2
    update_gradients_kernel<<<grid_size, block_size>>>(nn->grad_weights2, nn->grad_bias2, d_grad_output, d_hidden, batch_size, HIDDEN_SIZE, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // compute dX2 (gradient of loss w.r.t. input of second layer)
    float *d_dX2;
    CUDA_CHECK(cudaMalloc(&d_dX2, batch_size * HIDDEN_SIZE * sizeof(float)));
    grid_size.x = (HIDDEN_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (batch_size + block_size.y - 1) / block_size.y;
    matmul_a_bt_kernel<<<grid_size, block_size>>>(d_grad_output, nn->weights2, d_dX2, batch_size, OUTPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // compute GeLU derivative
    float *d_grad_hidden;
    CUDA_CHECK(cudaMalloc(&d_grad_hidden, batch_size * HIDDEN_SIZE * sizeof(float)));
    dgelu_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_hidden, d_grad_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // multiply gradients
    multiply_gradients_kernel<<<(batch_size * HIDDEN_SIZE + 255) / 256, 256>>>(d_dX2, d_grad_hidden, batch_size * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // update gradients for weights1 (W1.grad = d_GeLU_out.T @ input)
    grid_size.x = (INPUT_SIZE + block_size.x - 1) / block_size.x;
    grid_size.y = (HIDDEN_SIZE + block_size.y - 1) / block_size.y;
    matmul_at_b_kernel<<<grid_size, block_size>>>(d_input, d_dX2, nn->grad_weights1, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // update gradients for bias1
    update_gradients_kernel<<<grid_size, block_size>>>(nn->grad_weights1, nn->grad_bias1, d_dX2, d_input, batch_size, INPUT_SIZE, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaFree(d_grad_output));
    CUDA_CHECK(cudaFree(d_dX2));
    CUDA_CHECK(cudaFree(d_grad_hidden));

    CUDA_CHECK(cudaDeviceSynchronize());
}

// gradient descent step
__global__ void update_weights_kernel(float *weights, float *grad_weights, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= LEARNING_RATE * grad_weights[idx];
    }
}

void update_weights(NeuralNetwork *nn) {
    int block_size = 256;
    int grid_size;

    // update weights1
    grid_size = (HIDDEN_SIZE * INPUT_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->weights1, nn->grad_weights1, HIDDEN_SIZE * INPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // update weights2
    grid_size = (OUTPUT_SIZE * HIDDEN_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->weights2, nn->grad_weights2, OUTPUT_SIZE * HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // update bias1
    grid_size = (HIDDEN_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->bias1, nn->grad_bias1, HIDDEN_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // update bias2
    grid_size = (OUTPUT_SIZE + block_size - 1) / block_size;
    update_weights_kernel<<<grid_size, block_size>>>(nn->bias2, nn->grad_bias2, OUTPUT_SIZE);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}