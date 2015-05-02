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
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

// CUDA CONFIG
//--------------------
#define NUM_THREADS 256								
#define BLOCKS 1024									
//--------------------

// POPULATION CONTROL
//-------------------
#define NUM_CITIES 52								// *NUM_CITIES must be less than POPULATION_SIZE
#define MAX_COORD 250
#define POPULATION_SIZE NUM_THREADS*BLOCKS			// this should match #threads, at least in the beginning
#define NUM_EVOLUTIONS 100

#define MUTATION_RATE 0.05							// used to be 0.0015
#define ELITISM	true
#define TOURNAMENT_SIZE 128
//--------------------

#include "headers/city.h"
#include "headers/tour.h"
#include "headers/population.h"
#include "headers/hostUtils.h"
#include "headers/gaUtils.h"

using namespace std;

__global__  void initCuRand(curandState *randState)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= POPULATION_SIZE) return;

	curand_init(1337, tid, 0, &randState[tid]);
}

__global__ void evaluatePopulation(population_t *population, float *costTable)
{
    // Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= POPULATION_SIZE) return;

    evalTour(population->tours[tid], costTable);
}

// Maybe add cost table to reference during "tournamentSelection()"
__global__ void selection(population_t *population, curandState *randState, tour_t *parents)
{
	// Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= POPULATION_SIZE) return;

    parents[tid*2] = tournamentSelection(*population, randState, tid);
    parents[tid*2+1] = tournamentSelection(*population, randState, tid);
}

__global__ void crossover(population_t *population, tour_t *parents, curandState *randState, float *costTable, int index)
{
	// Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= POPULATION_SIZE) return;

    population->tours[tid].cities[0] = parents[2*tid].cities[0];

	city_t c1 = getValidNextCity(parents[tid*2], population->tours[tid], population->tours[tid].cities[index-1], index);
	city_t c2 = getValidNextCity(parents[tid*2+1], population->tours[tid], population->tours[tid].cities[index-1], index);

	// compare the two cities from parents to the last city that was chosen in the child
	if (costTable[c1.n*NUM_CITIES + population->tours[tid].cities[index-1].n] <= costTable[c2.n*NUM_CITIES + population->tours[tid].cities[index-1].n])
		population->tours[tid].cities[index] = c1;
	else
		population->tours[tid].cities[index] = c2;
}

__global__ void mutate(population_t *population, curandState *d_state)
{
	// Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= POPULATION_SIZE) return;
    
    // pick random number between 0 and 1 
	curandState localState = d_state[tid];

	// if random num is less than mutation_rate, perform mutation (swap two cities in tour)
    if (curand_uniform(&localState) < MUTATION_RATE)
    {
		int randNum1 = 1 + curand_uniform(&localState) * (NUM_CITIES - 1.0000001);
		int randNum2 = 1 + curand_uniform(&localState) * (NUM_CITIES - 1.0000001);

		city_t temp = population->tours[tid].cities[randNum1];
		population->tours[tid].cities[randNum1] = population->tours[tid].cities[randNum2];
		population->tours[tid].cities[randNum2] = temp;
		    
		d_state[tid] = localState;
    }

}

int main(int argc, char **argv)
{
	printf("\nSingle:\n---------\n");
	printf("THREADS:			%d\n", NUM_THREADS);
	printf("BLOCKS:				%d\n", BLOCKS);
	printf("TOURNAMENT_SIZE:		%d\n", TOURNAMENT_SIZE);
	printf("NUM_EVOLUTIONS:			%d\n", NUM_EVOLUTIONS);

    // -----------------------------------------------
    //		Population initialization on host (CPU)
	// -----------------------------------------------

	tour_t initialTour;
    float costTable[NUM_CITIES*NUM_CITIES];
	population_t initialPopulation;

	// READS INITIAL TOUR FROM FILE
	ifstream file("berlin52.txt");
    readTourFromFile(initialTour, file);

    // Build cost table to save time computing distance between cities
    // 	- array lookups are cheaper than squaring, adding, and sqrting
   	buildCostTable(initialTour, costTable);

   	// Initialize population by generating POPULATION_SIZE number of
   	// permutations of the initial tour, all starting at the same city
    initializePop(initialPopulation, initialTour, costTable);

    // ---------------------------
    //		GPU Mem allocation 
	// ---------------------------
    // ---------------------------

	population_t *d_Population;
	cudaMalloc((void **) &d_Population, sizeof(population_t));
	// array to store parents selected from tournament selection
    tour_t *d_Parents;
	cudaMalloc((void **) &d_Parents, sizeof(tour_t) * POPULATION_SIZE * 2);
	// cost table for crossover function (SCX crossover)
	float *d_CostTable;
	cudaMalloc((void **) &d_CostTable, sizeof(float) * NUM_CITIES * NUM_CITIES);
	// array for random numbers
	curandState *d_state;
    cudaMalloc((void**)&d_state, POPULATION_SIZE * sizeof(curandState));  
    cudaDeviceSynchronize();
    checkForError();

    // copies data to device for evolution
    cudaMemcpy(d_Population, &initialPopulation, sizeof(population_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_CostTable, &costTable, sizeof(float) * NUM_CITIES * NUM_CITIES, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    checkForError();

    // ---------------------------
    //		END GPU MEM ALLOC
    // ---------------------------


    // ----------------------------------------------
    // 	Times execution of evolve population on gpu
    // -----------------------------------------------
	float milliseconds;
	cudaEvent_t start, stop;
	cudaEventCreate (&start);
	cudaEventCreate (&stop);
	cudaEventRecord (start);
	   
	cudaDeviceSynchronize();
    checkForError();
	// -----------
	// MAIN LOOP
	// -----------

	// initialize random numbers array for tournament selection
	initCuRand <<< BLOCKS, NUM_THREADS >>> (d_state);
	cudaDeviceSynchronize();
    checkForError();

	// figure out distance and fitness for each individual in population
	evaluatePopulation <<< BLOCKS, NUM_THREADS >>> (d_Population, d_CostTable);
	
	for (int i = 0; i < NUM_EVOLUTIONS; ++i)
	{	
	    selection <<< BLOCKS, NUM_THREADS >>> (d_Population, d_state, d_Parents);

	    // breed the population with tournament selection and SCX crossover
	    // perform computation parallelized, build children iteratively
	    for (int j = 1; j < NUM_CITIES; ++j)
	        crossover <<< BLOCKS, NUM_THREADS >>> (d_Population, d_Parents, d_state, d_CostTable, j);

		mutate <<< BLOCKS, NUM_THREADS >>> (d_Population, d_state);

		evaluatePopulation <<< BLOCKS, NUM_THREADS >>> (d_Population, d_CostTable);
	}

	cudaEventRecord (stop);
	cudaEventSynchronize (stop);
	cudaEventElapsedTime (&milliseconds, start, stop);
	// -----------------------------------
	// 		END Timed GPU algorithm
	// -----------------------------------

	// copy memory back to device!
    cudaMemcpy(&initialPopulation, d_Population, sizeof(population_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    checkForError();
	
	//------------------------------
	//		OUTPUT
	//------------------------------

	tour_t fittest = getFittestTour(initialPopulation.tours, POPULATION_SIZE);

	printf("%f %f\n", milliseconds/1000, fittest.distance);

	//------------------------------
	//		END OUTPUT
	//------------------------------
	

	cudaFree(d_Population);;
	cudaFree(d_CostTable);
	cudaFree(d_Parents);
	cudaFree(d_state);

	return 0;
}

