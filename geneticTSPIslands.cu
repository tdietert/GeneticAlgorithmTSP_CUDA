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

// SET NUMBER OF TESTS
#define NUM_TESTS 20

// CUDA CONFIG
//--------------------
#define NUM_THREADS 256							// this is also the number of tours (ie, population size)
#define BLOCKS 1024									// Number of sub populations
//--------------------

// POPULATION CONTROL
//-------------------
#define NUM_CITIES 52									// must be number of cities being read in file
#define MAX_COORD 250		
#define POPULATION_SIZE NUM_THREADS						// this should match #threads, at least in the beginning
#define NUM_POPULATIONS BLOCKS
#define NUM_EVOLUTIONS 50

#define MUTATION_RATE 0.05							// used to be 0.0015
#define ELITISM	true
#define TOURNAMENT_SIZE 16
//--------------------

#include "headers/city.h"
#include "headers/tour.h"
#include "headers/population.h"
#include "headers/hostUtils.h"
#include "headers/gaUtils.h"

using namespace std;

__global__ void initCuRand(curandState *randState)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= NUM_THREADS*BLOCKS) return;

	curand_init(1337, tid, 0, &randState[tid]);
}

__global__ void evaluatePopulations(population_t *populations, const float *costTable)
{
    // Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= NUM_THREADS*BLOCKS) return;

    evalTour(populations[blockIdx.x].tours[threadIdx.x], costTable);
}

// Maybe add cost table to reference during "tournamentSelection()"
__global__ void selection(population_t *populations, curandState *randState, tour_t *parents)
{
	// Get thread (particle) ID
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= NUM_THREADS*BLOCKS) return;

    if (ELITISM && blockIdx.x == 0)
    	parents[tid*2] = getFittestTour(populations[blockIdx.x].tours, POPULATION_SIZE);
    else
	    parents[tid*2] = tournamentSelection(populations[blockIdx.x], randState, tid);

    parents[tid*2+1] = tournamentSelection(populations[blockIdx.x], randState, tid);
}

__global__ void crossover(population_t *populations, tour_t *parents, curandState *randState, float *costTable, int index)
{
	// Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= NUM_THREADS*BLOCKS) return;

    populations[blockIdx.x].tours[threadIdx.x].cities[0] = parents[2*tid].cities[0];

	city_t c1 = getValidNextCity(parents[tid*2], populations[blockIdx.x].tours[threadIdx.x], populations[blockIdx.x].tours[threadIdx.x].cities[index-1], index);
	city_t c2 = getValidNextCity(parents[tid*2+1], populations[blockIdx.x].tours[threadIdx.x], populations[blockIdx.x].tours[threadIdx.x].cities[index-1], index);

	// compare the two cities from parents to the last city that was chosen in the child
	city_t currentCity = populations[blockIdx.x].tours[threadIdx.x].cities[index-1];
	if (costTable[c1.n*NUM_CITIES + currentCity.n] <= costTable[c2.n*NUM_CITIES + currentCity.n])
		populations[blockIdx.x].tours[threadIdx.x].cities[index] = c1;
	else
		populations[blockIdx.x].tours[threadIdx.x].cities[index] = c2;
}

__global__ void mutate(population_t *populations, curandState *d_state)
{
	// Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= NUM_THREADS*BLOCKS) return;
    
    // pick random number between 0 and 1 
	curandState localState = d_state[tid];

	// if random num is less than mutation_rate, perform mutation (swap two cities in tour)
    if (curand_uniform(&localState) < MUTATION_RATE)
    {
		int randNum1 = 1 + curand_uniform(&localState) * (NUM_CITIES - 1.0000001);
		int randNum2 = 1 + curand_uniform(&localState) * (NUM_CITIES - 1.0000001);

		city_t temp = populations[blockIdx.x].tours[threadIdx.x].cities[randNum1];
		populations[blockIdx.x].tours[threadIdx.x].cities[randNum1] = populations[blockIdx.x].tours[threadIdx.x].cities[randNum2];
		populations[blockIdx.x].tours[threadIdx.x].cities[randNum2] = temp;
		    
		d_state[tid] = localState;
    }
}

__global__ void migrate(population_t *populations)
{
	// Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= NUM_THREADS*BLOCKS || threadIdx.x != 0) return;

    int indexOfLeastFitInNeighbor;
    if (blockIdx.x == BLOCKS-1)
    {
    	indexOfLeastFitInNeighbor = getIndexOfLeastFit(populations[0]);
    	populations[0].tours[indexOfLeastFitInNeighbor] = getFittestTour(populations[blockIdx.x].tours, POPULATION_SIZE);
    }
    else
    {
    	indexOfLeastFitInNeighbor = getIndexOfLeastFit(populations[blockIdx.x+1]);
    	populations[blockIdx.x+1].tours[indexOfLeastFitInNeighbor] = getFittestTour(populations[blockIdx.x].tours, POPULATION_SIZE);
    }
}

int main(int argc, char **argv)
{
	printf("THREADS:			%d\n", NUM_THREADS);
	printf("BLOCKS:				%d\n", BLOCKS);
	printf("TOURNAMENT_SIZE:		%d\n", TOURNAMENT_SIZE);
	printf("NUM_EVOLUTIONS:			%d\n", NUM_EVOLUTIONS);

	// Build city distances table
	tour_t initialTour;
    float costTable[NUM_CITIES*NUM_CITIES];
	population_t populations[BLOCKS];
	tour_t parents[NUM_POPULATIONS*POPULATION_SIZE*2];

	// READS INITIAL TOUR FROM FILE
	ifstream file("berlin52.txt");
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

    // collects run results
	tour_t tours[NUM_TESTS];
	// -----------------
	// MAIN LOOP 
	// -----------------
	for (int k = 0; k < NUM_TESTS; ++k)
	{

     	// Initializes all populations to NUMTHREADS number of individuals, randomized
     	// Done on CPU (host)
	   	for (int i = 0; i < BLOCKS; ++i)
	   		initializePop(populations[i], initialTour);

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
		// figure out distance and fitness for each individual in population
		evaluatePopulations <<< BLOCKS, NUM_THREADS >>> (d_populations, d_costTable);		
		for (int i = 0; i < NUM_EVOLUTIONS; ++i)
		{	
		    selection <<< BLOCKS, NUM_THREADS >>> (d_populations, d_state, d_parents);

		    // breed the population with tournament selection and SCX crossover
		    // perform computation parallelized, build children iteratively
		    for (int j = 1; j < NUM_CITIES; ++j)
		        crossover <<< BLOCKS, NUM_THREADS >>> (d_populations, d_parents, d_state, d_costTable, j);

			mutate <<< BLOCKS, NUM_THREADS >>> (d_populations, d_state);

			evaluatePopulations <<< BLOCKS, NUM_THREADS >>> (d_populations, d_costTable);

			// migrate every 5 evolutions
			if (i % 3 == 0)
				migrate <<< BLOCKS, NUM_THREADS >>> (d_populations);
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
		printf("%f %f\n", milliseconds/1000, fittest.distance);
		//printf("Program execution time: %f sec", timeInitGPUPop+timeInitHostPop+(milliseconds/1000)+evalPopTime);

		tours[k] = fittest;
    }

	cudaFree(d_populations);
	cudaFree(d_parents);
	cudaFree(d_costTable);
	cudaFree(d_state);

	// tour_t mostFittest = getFittestTour(tours, NUM_TESTS);
	// printf("\nThe fittest tour OVERALL has length %f\n\n", mostFittest.distance);
	// printf("Winning Tour:\n");
	// printTour(mostFittest);

	return 0;
}
