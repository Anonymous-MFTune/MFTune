# Gcc latest Configuration Parameters

This table summarizes selected Gcc latest configuration parameters, showing the configured defaults, allowed ranges, types, and official descriptions with documentation links for verification.

| Parameter | Default Value | Valid Range / Enum Values | Type | Description |
|-----------|----------|--------|------|-------------|
| [-ffast-math](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-ffast-math) | Off      | On, Off | enum | Enables aggressive floating-point optimizations that may violate IEEE or ANSI standards. | |
| [-funroll-all-loops](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-funroll-all-loops) | Off      | On, Off | enum | Unrolls all loops, which can increase performance at the cost of code size. | |
| [-fno-inline-small-functions](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-finline-small-functions) | Off      | On, Off | enum | Prevents inlining of functions that are small enough to be inlined. | |
| [-finline-functions](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-finline-functions) | Off      | On, Off | enum | Enables inlining of functions, which can improve performance. | |
| [-fno-math-errno](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fno-math-errno) | Off      | On, Off | enum | Disables setting `errno` after math functions, allowing for faster math operations. | |
| [-fipa-cp-clone](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fipa-cp-clone) | Off      | On, Off | enum | Enables interprocedural constant propagation with function cloning. | |
| [-funsafe-math-optimizations](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-funsafe-math-optimizations) | Off      | On, Off | enum | Allows optimizations that may not be safe for all math computations. | |
| [-fno-tree-loop-optimize](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-ftree-loop-optimize) | Off      | On, Off | enum | Disables loop optimizations on the GIMPLE representation. | |
| [-fno-merge-constants](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fmerge-constants) | Off      | On, Off | enum | Prevents merging of identical constants, which can increase code size. | |
| [-fno-omit-frame-pointer](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fomit-frame-pointer) | Off      | On, Off | enum | Retains the frame pointer in functions, which can aid in debugging. | |
| [-fno-align-labels](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-falign-labels) | Off      | On, Off | enum | Disables alignment of labels, which can reduce code size. | |
| [-fno-tree-dse](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-ftree-dse) | Off      | On, Off | enum | Disables dead store elimination on the GIMPLE representation. | |
| [-fwrapv](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fwrapv) | Off      | On, Off | enum | Assumes signed arithmetic overflow wraps around using two's complement. | |
| [-fgcse-after-reload](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fgcse-after-reload) | Off      | On, Off | enum | Enables global common subexpression elimination after register allocation. | |
| [-fno-align-jumps](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-falign-jumps) | Off      | On, Off | enum | Disables alignment of jump targets, which can reduce code size. | |
| [-fno-asynchronous-unwind-tables](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fasynchronous-unwind-tables) | Off      | On, Off | enum | Disables generation of unwind tables for exception handling. | |
| [-fno-cse-follow-jumps](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fcse-follow-jumps) | Off      | On, Off | enum | Disables common subexpression elimination across branches. | |
| [-fno-ivopts](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fivopts) | Off      | On, Off | enum | Disables induction variable optimizations. | |
| [-fno-guess-branch-probability](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fguess-branch-probability) | Off      | On, Off | enum | Disables guessing of branch probabilities. | |
| [-fprefetch-loop-arrays](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fprefetch-loop-arrays) | Off      | On, Off | enum | Enables prefetching of arrays in loops, which can improve cache performance. | |
| [-fno-tree-coalesce-vars](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-ftree-coalesce-vars) | Off      | On, Off | enum | Disables coalescing of variables, which can increase register pressure. | |
| [-fno-common](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html#index-fcommon) | Off      | On, Off | enum | Places uninitialized global variables in the BSS section, which can prevent multiple definitions. | |