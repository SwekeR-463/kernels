### To-Do's
- [x] Optimized Layer Norm Kernel

### Notes
* when calculating the mean, we have to actually have, you know, tile and strides, so that we can do something like matrix multiplication, where we can add two numbers, find their mean, store in intermediate result like that until we have done for the n numbers
* then for variance also, we have to do the same. But here what we have to do is variance is actually a bigger formula. And so we have to divide the formula into smaller parts and then do the same
* then we will normalize it
* further optimizations can be using fp16, tensor cores, hardware mathematical functions