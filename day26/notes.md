### To-Do's
- [x] cuBLAS Strided Batched GEMM Kernel
- [ ] Naive Layer Norm Kernel

### Notes
* one small observation that i found is that while playing around the batch count for the gemm, 32 & 64 were the sweet spot for the operation on my 4050 (6gb)
---
* layer normalization by computing the mean and variance used for normalization from all of the summed inputs to the neurons in a layer on a single training case
* layer normalization performs exactly the same computation at training and test times
* layer normalization directly estimates the normalization statistics from the summed inputs to the neurons within a hidden layer so the normalization does not introduce any new dependencies between training cases
* it works well for RNNs and improves both the training time and the generalization performance of several existing RNN models
* 
---
### LayerNorm Formulas

### Input
Let \( \mathbf{x} \) be the input vector of size \( d \) (feature dimension).

### Mean
Compute the mean \( \mu \) of the input vector:
\[
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
\]

### Variance
Compute the variance \( \sigma^2 \) of the input vector:
\[
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
\]

### Normalization
Normalize the input vector using the mean and variance:
\[
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
\]
where \( \epsilon \) is a small constant (e.g., \( 10^{-5} \)) added for numerical stability.