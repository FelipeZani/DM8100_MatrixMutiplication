# Tasks DM8100


## Serial 
    [] Add clock to record duration
    [] Optimisation?
## OpenMP
    [] Implementation
        [] Explore different scheduling, collapse and other OpenMP options for maximum perfor-mance gains.
        [] Explore different loop orders, fx change the naive IJK to IKJ ordering.
    [] Optimisations
## OpenMPI
    [] Devise a strategy to do your computations. Think about how to block your matrices or how to separate your matrices onto different cores.
    [] Distribute your matrices onto different cores, e.g., by using MPI Scatter.
    [] Collect matrix C in the main process, fx using MPI Gather.
## CUDA
    [] Offload the computation to a GPU accelerator.
    [] Write a CUDA kernel where each thread computes one element of the output matrix.
    [] Manage memory transfers between the host and the device.
## Scaling
    [] Check the strong scaling of your matrix-multiplication codes.
    [] Check the weak scaling of your matrix-multiplication codes. (Keep in mind that naive matrix multiplications scale with N^3).
    more info on the matter : http://www.kth.se/blogs/pdc/2018/11/scalability-strong-and-weak-scaling/
