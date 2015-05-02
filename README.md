# GeneticAlgorithmTSP_CUDA
A genetic algorithm to find optimal solutions for TSP (Travelling Salesman Problem) using the CUDA Architecture (GPU). This is done as a final project for my Parallel and Distributed Processing class at USF in conjunction with Berkeley's Applications of Parallel Computers, Spring 2015. The purpose of this project is to familiarize myself with designing and implementing genetic algorithms, with a focus on learning how to write Parallel code on the CUDA architecture, as well as the OpenMP libraries for C++. 

# Overview:
This project is meant to be run on the Stampede.tacc super computer at University of Texas, Austin. The code runs on a single node on the stampede cluster, calling a single GPU at the moment, using Nvidia's CUDA. In CUDA, a kernel (GPU function) is executed and is composed of Blocks and Threads. Each block can have up to 1024 threads, and each kernel can have ~65000 blocks. I have designed the algorithm to create one population per GPU, made up of NTHREADS * NBLOCKS that the kernel is launched with. For instance, in a popular TSP problem "Berlin52", the total solution space is 8.0658175 x 10^67, so by launching 65000 blocks, each with 1024 threads, at each generation the solution space will be ~65000000 individuals. This is a fraction of the number of possible solutions to the TSP of 52 cities. I will also use OpenMP to utilize the 16 cores per node (spawn 16 CPU threads), to call up to 16 different GPUs to run 16 different populations, and select the best population out of the 16 individual populations. 

# To run this code:
1) Login to stampede:
2) $ sbatch job-cudaGA
or
3) $ sbatch job-cudaGa-Opt
or
4) $ sbatch job-cudaGA-islands

To compile locally with an NVIDIA GPU:

1) nvcc geneticTSP.cu -o tsp.out
2) do something to run an mpi program on your machine...

