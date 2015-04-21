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
#define NUM_THREADS 256							// this is also the number of tours (ie, population size)
#define BLOCKS 2048									// Number of sub populations
//--------------------

// POPULATION CONTROL
//-------------------
#define NUM_CITIES 52								// *NUM_CITIES must be less than POPULATION_SIZE
#define MAX_COORD 250
#define POPULATION_SIZE NUM_THREADS*BLOCKS			// this should match #threads, at least in the beginning
#define NUM_EVOLUTIONS 1500

#define MUTATION_RATE 0.15							// used to be 0.0015
#define ELITISM	true
#define TOURNAMENT_SIZE 13
//--------------------

#include "city.h"
#include "tour.h"
#include "population.h"
#include "hostUtils.h"
#include "gaUtils.h"

using namespace std;

__global__  void initCuRand(curandState *randState)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= POPULATION_SIZE) return;

	curand_init(1337, tid, 0, &randState[tid]);
}

__global__ void evaluatePopulation(population_t *population)
{
    // Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= POPULATION_SIZE) return;

    evalTour(population->tours[tid]);
}

// Maybe add cost table to reference during "tournamentSelection()"
__global__ void selection(population_t *population, curandState *randState, tour_t *parents)
{
	// Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= POPULATION_SIZE) return;

    parents[2*tid] = tournamentSelection(population, randState, tid);
    parents[2*tid+1] = tournamentSelection(population, randState, tid);
}

__global__ void crossover(population_t *newPopulation, tour_t *parents, curandState *randState, int index)
{
	// Get thread (particle) ID
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid >= POPULATION_SIZE) return;

    newPopulation->tours[tid].cities[0] = parents[2*tid].cities[0];

	city_t currentCity = newPopulation->tours[tid].cities[index-1];
	city_t c1 = getValidNextCity(parents, newPopulation->tours[tid], currentCity, index, tid*2);
	city_t c2 = getValidNextCity(parents, newPopulation->tours[tid], currentCity, index, tid*2+1);

	if (distBetweenCities(c1, currentCity) <= distBetweenCities(c2, currentCity))
		newPopulation->tours[tid].cities[index] = c1;
	else
		newPopulation->tours[tid].cities[index] = c2;
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
	srand(time(NULL));
	cudaDeviceSynchronize();

    // -----------------------------------------------
    //		Population initialization on host (CPU)
	// -----------------------------------------------

	tour_t initialTour;
    int costTable[NUM_CITIES*NUM_CITIES];
	population_t initialPopulation;
	tour_t parents[POPULATION_SIZE*2];

	// READS INITIAL TOUR FROM FILE
	ifstream file("cityData.txt");
    readTourFromFile(initialTour, file);

    // Build cost table to save time computing distance between cities
    // 	- array lookups are cheaper than squaring, adding, and sqrting
   	buildCostTable(initialTour, costTable);

   	// Initialize population by generating POPULATION_SIZE number of
   	// permutations of the initial tour, all starting at the same city
    initializePop(initialPopulation, initialTour);

    // ---------------------------
    //		GPU Mem allocation 
	// ---------------------------
    // population on device
	population_t *d_Population;
	cudaMalloc((void **) &d_Population, sizeof(population_t));
	// newPopulation on device to hold offspring of current generation
	population_t *d_NewPopulation;
	cudaMalloc((void **) &d_NewPopulation, sizeof(population_t));
	// array to store parents selected from tournament selection
    tour_t *d_Parents;
	cudaMalloc((void **) &d_Parents, sizeof(tour_t) * POPULATION_SIZE * 2);
	
	// COST TABLE:
	// -----------
	// cost table for crossover function (SCX crossover)
	// int *d_CostTable;
	// cudaMalloc((void **) &d_CostTable, sizeof(int) * NUM_CITIES * NUM_CITIES);

	curandState *d_state;
    cudaMalloc((void**)&d_state, POPULATION_SIZE * sizeof(curandState));

    cudaDeviceSynchronize();
    checkForError();

    // copies data to device for evolution
    cudaMemcpy(d_Population, &initialPopulation, sizeof(population_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Parents, &parents, sizeof(tour_t) * POPULATION_SIZE * 2, cudaMemcpyHostToDevice);
    //cudaMemcpy(d_CostTable, &costTable, sizeof(int) * NUM_CITIES * NUM_CITIES, cudaMemcpyHostToDevice);
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

	// if error > 5, mod it by 4 and that's the iteration it occurs on
	for (int i = 0; i < NUM_EVOLUTIONS; ++i)
	{	
		// figure out distance and fitness for each individual in population
	    evaluatePopulation <<< BLOCKS, NUM_THREADS >>> (d_Population);

	    selection <<< BLOCKS, NUM_THREADS >>> (d_Population, d_state, d_Parents);

	    // breed the population with tournament selection and SCX crossover
	    // perform computation parallelized, build children iteratively
	    for (int j = 1; j < NUM_CITIES; ++j)
	        crossover <<< BLOCKS, NUM_THREADS >>> (d_NewPopulation, d_Parents, d_state, j);

		// update population to children produced by crossover
		d_Population = d_NewPopulation;

		mutate <<< BLOCKS, NUM_THREADS >>> (d_Population, d_state);
	}

	// one last population eval
	evaluatePopulation <<< BLOCKS, NUM_THREADS >>> (d_Population);

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
    
    // 		DEBUGGING TOURNAMENT SELECTION
    // tour_t tournamentTest[2*POPULATION_SIZE];
    // cudaMemcpy(&tournamentTest, d_Parents, 2 * sizeof(tour_t) * POPULATION_SIZE, cudaMemcpyDeviceToHost);
	// for (int i = 0; i < 2*POPULATION_SIZE; ++i)
	// {
	// 	cout << tournamentTest[i].distance << endl;
	// }
	
	//------------------------------
	//		OUTPUT
	//------------------------------

	//printPopulation(initialPopulation);
	printf("Timed evolvePopulation: %d ms\n", milliseconds);

	tour_t fittest = getFittestTour(initialPopulation.tours, POPULATION_SIZE);
	printf("THREADS:	%d\n", NUM_THREADS);
	printf("BLOCKS: 	%d\n", BLOCKS);
	printf("POPULATION_SIZE:	 %d\n", POPULATION_SIZE);
	printf("NUM_EVOLUTIONS: 	 %d\n", NUM_EVOLUTIONS);
	printf("\nThe fittest tour has length %f\n\n", fittest.distance);
	printf("Winning Tour:\n");
	printTour(fittest);

	//------------------------------
	//		END OUTPUT
	//------------------------------

	cudaFree(d_Population);
	cudaFree(d_NewPopulation);
	//cudaFree(d_CostTable);
	cudaFree(d_Parents);
	cudaFree(d_state);
	return 0;
}

