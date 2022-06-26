## Week 1 - Cuda Computation
- The problem is to compute the output matrix C defined as tran(A+B) - tran(B-A) using gpu kernels
- Three different kernels are to be written, each of which is invoked with respective configuration
	- 1D grid, 1D block
	- 1D grid, 2D block
	- 2D grid, 2D block
- Tiling is not allowed while calculating the transpose.
