# GeneticAlgorithmTSP_CUDA
A genetic algorithm to find optimal solutions for TSP (Travelling Salesman Problem) using the CUDA Architecture (GPU). 

# Overview:
This project is meant to be run on the Stampede.tacc super computer at University of Texas, Austin. The code runs on a single node on the stampede cluster, calling a single GPU at the moment, using Nvidia's CUDA. I will also use OpenMP to utilize the 16 cores per node (spawn 16 threads), to call up to 16 different GPUs to run 16 different populations, and select the best population out of the 16 individual populations.

# To run this code:
Login to stampede:
sbatch job-cudaGA

# To compile locally with an NVIDIA GPU:
nvcc geneticTSP.cu -o tsp.out

