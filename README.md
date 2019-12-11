# NEON, AVX, SSE and C implementations for L2 distance function

## input 
two vectors of float have the same length.

## notes:
-  in AVX we process 8 by 8 each shot while in SSE and NEON we process 4 by 4.
-  implementation of int is much faster than float and i will add it later on.