### To-Do's
- [ ] RMS Norm Kernel
- [ ] Tensor Core Mat Mul

### Notes
* RMSNorm normalizes activations based on the root mean square of the activations themselves, rather than using mini-batch or layer statistics
* This approach ensures that the activations are consistently scaled regardless of the mini-batch size or the number of features
* RMSNorm introduces learnable scale parameters, offering similar adaptability to BatchNorm
* 
