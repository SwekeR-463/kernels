### To-Do's
- [ ] 2D Conv Kernel
- [x] Prefix Sum kernl
- [x] CH8 PMPP

### Notes
* an inclusive scan operation takes a binary associative operator ⊕ and an input array of n elements [x0, x1, …, xn−1] and returns the following output array: [ x0 , ( x0 ⊕ x1 ), …, ( x0 ⊕ x1 ⊕ … ⊕ xn −1 )]
* An exclusive scan operation is similar to an inclusive operation, except that the former returns the following output array: [0, x0 , ( x0 ⊕ x1 ), …, ( x0 ⊕ x1 ⊕ … ⊕ xn −2 )]
* In practice, parallel scan is often used as a primitive operation in parallel algorithms that perform radix sort, quick sort, string comparison, polynomial evaluation, solving recurrences, tree operations, stream compaction, and histograms
* 