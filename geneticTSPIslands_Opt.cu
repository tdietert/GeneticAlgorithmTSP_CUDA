#include <algorithm>
#include <ctime>
#include <iostream>
#include <fstream>
#include <math.h>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <utility>
#include <time.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define NUM_TESTS 5

// CUDA CONFIG
//--------------------
#define NUM_THREADS 256     						// this is also the number of tours (ie, population size)
#define BLOCKS 1024									// Number of sub populations (islands)
//--------------------

// POPULATION CONTROL
//-------------------
#define NUM_CITIES 194									// must be number of cities being read in file
#define MAX_COORD 250		
#define POPULATION_SIZE NUM_THREADS						// this should match #threads, at least in the beginning
#define NUM_POPULATIONS BLOCKS
#define NUM_EVOLUTIONS 100

#define MUTATION_RATE 0.33							
#define ELITISM	true
#define TOURNAMENT_SIZE 12
//--------------------

#include "headers/city.h"
#include "headers/tour.h"
#include "headers/population.h"
#include "headers/hostUtils.h"
#include "headers/gaUtils.h"

using namespace std;

__global__ void initCuRand(curandState *randState)
{	
	// Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= NUM_THREADS*BLOCKS) return;

	curand_init((unsigned long long)clock(), tid, 0, &randState[tid]);
}

__device__ void evaluatePopulation(population_t &population, float *costTable, const int &tIdx)
{
    evalTour(population.tours[tIdx], costTable);
}

// Maybe add cost table to reference during "tournamentSelection()"
__device__ void selection(population_t &population, curandState *randState, tour_t *parents, const int &blkIdx, const int &tid)
{
    if (ELITISM && (blkIdx == 0))
    {
    	parents[tid*2] = getFittestTour(population.tours, POPULATION_SIZE);
        parents[tid*2+1] = getFittestTour(population.tours, POPULATION_SIZE);
    }
    else
    {
	    parents[tid*2] = tournamentSelection(population, randState, tid);
    	parents[tid*2+1] = tournamentSelection(population, randState, tid);
    }

}

__device__ void crossover(population_t &population, tour_t *parents, curandState *randState, float *costTable, const int &tIdx, const int &tid, const int &index)
{
	// initializes first city of child
	if (index == 1)
    	population.tours[tIdx].cities[0] = parents[2*tid].cities[0];

	city_t c1 = getValidNextCity(parents[tid*2], population.tours[tIdx], population.tours[tIdx].cities[index-1], index);
	city_t c2 = getValidNextCity(parents[tid*2+1], population.tours[tIdx], population.tours[tIdx].cities[index-1], index);

	// compare the two cities from parents to the last city that was chosen in the child
	city_t currentCity = population.tours[tIdx].cities[index-1];
	if (costTable[c1.n*NUM_CITIES + currentCity.n] <= costTable[c2.n*NUM_CITIES + currentCity.n])
		population.tours[tIdx].cities[index] = c1;
	else
		population.tours[tIdx].cities[index] = c2;

}

__device__ void mutate(population_t &population, curandState *d_state, const int &blkIdx, const int &tIdx)
{    
    // pick random number between 0 and 1 
	curandState localState = d_state[blkIdx*NUM_THREADS + tIdx];

	// if random num is less than mutation_rate, perform mutation (swap two cities in tour)
    if (curand_uniform(&localState) < MUTATION_RATE && tIdx > 0)
    {
		int randNum1 = 1 + curand_uniform(&localState) * (NUM_CITIES - 2);
		int randNum2 = 1 + curand_uniform(&localState) * (NUM_CITIES - 2);

		city_t temp = population.tours[tIdx].cities[randNum1];
		population.tours[tIdx].cities[randNum1] = population.tours[tIdx].cities[randNum2];
		population.tours[tIdx].cities[randNum2] = temp;
		    
		d_state[blkIdx*NUM_THREADS + tIdx] = localState;
    }

}

__global__ void moveWorstToLastPos(population_t *populations, float *costTable)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= NUM_THREADS*BLOCKS) return;

	int indexOfLeastFit = getIndexOfLeastFit(populations[blockIdx.x]);
	tour_t worstTour = populations[blockIdx.x].tours[indexOfLeastFit];
	populations[blockIdx.x].tours[indexOfLeastFit] = populations[blockIdx.x].tours[POPULATION_SIZE-1];
    populations[blockIdx.x].tours[POPULATION_SIZE-1] = worstTour;
}

__global__ void migrate(population_t *populations)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= NUM_THREADS*BLOCKS) return;

	if (blockIdx.x == NUM_POPULATIONS-1)
		populations[0].tours[POPULATION_SIZE-1] = getFittestTour(populations[blockIdx.x].tours, POPULATION_SIZE);
	else
		populations[blockIdx.x+1].tours[POPULATION_SIZE-1] = getFittestTour(populations[blockIdx.x].tours, POPULATION_SIZE);
}

__global__ void evolvePopulation(population_t *populations, tour_t *parents, curandState *randState, float *costTable)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= NUM_THREADS*BLOCKS) return;

 //    if (!checkTour(populations[blockIdx.x].tours[threadIdx.x]))
	// printf("evalPop broke on thread %d, %d\n", blockIdx.x, threadIdx.x);

    selection(populations[blockIdx.x], randState, parents, blockIdx.x, tid);
    __syncthreads();

 //    if (!checkTour(populations[blockIdx.x].tours[threadIdx.x]))
	// printf("Selection broke on thread %d, %d\n", blockIdx.x, threadIdx.x);

    for (int i = 1; i < NUM_CITIES; ++i)
    	crossover(populations[blockIdx.x], parents, randState, costTable, threadIdx.x, tid, i);
    __syncthreads();

 //    if (!checkTour(populations[blockIdx.x].tours[threadIdx.x]))
	// printf("Crossover broke on thread %d, %d\n", blockIdx.x, threadIdx.x);

    mutate(populations[blockIdx.x], randState, blockIdx.x, threadIdx.x);
    __syncthreads();

    if (!checkTour(populations[blockIdx.x].tours[threadIdx.x]))
	printf("Mutate broke on thread %d, %d\n", blockIdx.x, threadIdx.x);

    evaluatePopulation(populations[blockIdx.x], costTable, threadIdx.x);
    __syncthreads();
}

int main(int argc, char **argv)
{
	printf("\nIslands:\n----------\n");
	printf("THREADS:			%d\n", NUM_THREADS);
	printf("BLOCKS:				%d\n", BLOCKS);
	printf("TOURNAMENT_SIZE:		%d\n", TOURNAMENT_SIZE);
	printf("NUM_EVOLUTIONS:			%d\n", NUM_EVOLUTIONS);

	// -----------------
	// MAIN LOOP 
	// -----------------
	tour_t tours[NUM_TESTS];
	float runtime = 0;
	for (int k = 0; k < NUM_TESTS; ++k)
	{
		// Build city distances table
		tour_t initialTour;
	    float costTable[NUM_CITIES*NUM_CITIES];
		population_t populations[BLOCKS];
		tour_t parents[NUM_POPULATIONS*POPULATION_SIZE*2];

	    // READS INITIAL TOUR FROM FILE
		ifstream file("quatar194.txt");
	    readTourFromFile(initialTour, file);

	    // Build cost table to save time computing distance between cities
	    // 	- array lookups are cheaper than squaring, adding, and sqrting
	   	buildCostTable(initialTour, costTable);

	    // ---------------------------
	    //		GPU Mem allocation 
		// ---------------------------
		population_t *d_populations;
		cudaMalloc((void **) &d_populations, BLOCKS * sizeof(population_t));

		// array to store parents selected from tournament selection
	    tour_t *d_parents;
		cudaMalloc((void **) &d_parents, sizeof(tour_t) * BLOCKS * NUM_THREADS * 2);

		// cost table for crossover function (SCX crossover)
		float *d_costTable;
		cudaMalloc((void **) &d_costTable, sizeof(float) * NUM_CITIES * NUM_CITIES);
	    cudaMemcpy(d_costTable, &costTable, sizeof(float) * NUM_CITIES * NUM_CITIES, cudaMemcpyHostToDevice);

		curandState *d_state;
	    cudaMalloc((void**)&d_state, BLOCKS * NUM_THREADS * sizeof(curandState));

     	// Initializes all populations to NUMTHREADS number of individuals, randomized
     	// Done on CPU (host)
	   	for (int i = 0; i < BLOCKS; ++i)
	   		initializePop(populations[i], initialTour, costTable);

	    // copies data from host to device for evolution
	    cudaMemcpy(d_populations, &populations, NUM_POPULATIONS * sizeof(population_t), cudaMemcpyHostToDevice);

	    // ----------------------------------------------
	    // 	Times execution of evolve population on gpu
	    // -----------------------------------------------
		float milliseconds = 0;
		cudaEvent_t start, stop;
		cudaEventCreate (&start);
		cudaEventCreate (&stop);
		cudaEventRecord (start);
		   
		// -----------
		// MAIN LOOP
		// -----------

		// initialize random numbers array for tournament selection
		initCuRand <<< BLOCKS, NUM_THREADS >>> (d_state);
		for (int i = 0; i < NUM_EVOLUTIONS; ++i)
		{	
		    evolvePopulation <<< BLOCKS, NUM_THREADS >>> (d_populations, d_parents, d_state, d_costTable);
		    
		    // only migrate every 5 iterations
		    if (i % 4 == 0)
		    {
		    	moveWorstToLastPos <<< BLOCKS, 1 >>> (d_populations, d_costTable);
		    	migrate	<<< BLOCKS, 1 >>> (d_populations);
		    }
		}
		// -----------------------------------
		// 		END MAIN LOOP
		// -----------------------------------

		cudaEventRecord (stop);
		cudaEventSynchronize (stop);
		cudaEventElapsedTime (&milliseconds, start, stop);

		// copy memory back to device!
	    cudaMemcpy(&populations, d_populations, NUM_POPULATIONS * sizeof(population_t), cudaMemcpyDeviceToHost);
	    cudaDeviceSynchronize();
	    checkForError();
		
		//printPopulation(initialPopulation);
		tour_t bestIndivs[NUM_POPULATIONS];
		for (int i = 0; i < NUM_POPULATIONS; ++i)
			bestIndivs[i] = getFittestTour(populations[i].tours, NUM_THREADS);

		tour_t fittest = getFittestTour(bestIndivs, NUM_POPULATIONS);

		// ---------------------
		// PRINTS OUTPUT TO FILE
		// ---------------------
		//printf("%f %f\n", milliseconds/1000, fittest.distance);
		tours[k] = fittest;
		runtime += milliseconds/1000;
		//printf("Program execution time: %f sec", timeInitGPUPop+timeInitHostPop+(milliseconds/1000)+evalPopTime);
 		//tours[k] = fittest;

		cudaFree(d_populations);
		cudaFree(d_parents);
		cudaFree(d_costTable);
		cudaFree(d_state);

    }

	tour_t mostFittest = getFittestTour(tours, NUM_TESTS);
	printf("%f %f\n", runtime/NUM_TESTS, mostFittest.distance);

	return 0;
}

