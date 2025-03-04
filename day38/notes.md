### To-Do's
- [ ] Flash Attention Forward Pass

### Notes
* flash attention processes attention in blocks to avoid excessive memory transfers between HRAM and SRAM, making it much more efficient
* instead of storing large intermediate attention matrices in memory (which is expensive), flash attention splits computations into smaller chunks, keeping only what's needed in fast, on-chip memory (SRAM) this minimizes unnecessary memory reads/writes, making it both faster and more memory-efficient than standard attention