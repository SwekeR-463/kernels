### To-Do's
- [x] RMS Norm Kernel
- [ ] Tensor Core Mat Mul

### Notes
* RMSNorm normalizes activations based on the root mean square of the activations themselves, rather than using mini-batch or layer statistics
* This approach ensures that the activations are consistently scaled regardless of the mini-batch size or the number of features
* RMSNorm introduces learnable scale parameters, offering similar adaptability to BatchNorm
* Given an input vector \( x \in \mathbb{R}^d \), RMSNorm is defined as:

$$
\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot \gamma
$$

where:

$$
\text{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}
$$

- \( \gamma \) is a trainable scaling parameter.
- \( \epsilon \) is a small constant for numerical stability.
- \( d \) is the dimensionality of \( x \).
