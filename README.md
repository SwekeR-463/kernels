# Kernels in CUDA || Triton

kernels of different DL funcs

### **activation**
* ELU (fp32, fp16, fp16x2, fp16x8_packed)
* GeLU
* Sigmoid (fp32, fp16, fp16x8_packed)
* ReLU (fp32, fp16)
* Swish (fp32, fp16)

### **embedding**
* similar kernel to `torch.nn.functional.embedding` in fp32 & fp16