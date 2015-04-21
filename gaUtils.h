//--------------------------
//
//  This file contains all helper functions neede for
//  manipulation of population on GPU
//
//--------------------------

// need to implement pow function for CUDA (can't use stl)
__host__ __device__ float cudaPow(float x, float y)
{
	float temp = 1;
	for (int i = 0; i < y; ++i)
	{
		temp *= x; 
	}
	return temp;
}

//----------------------
// BUILD COST TABLE FOR SCX Crossover
//----------------------
void buildCostTable(tour_t &tour, int (&cost_table)[NUM_CITIES*NUM_CITIES])
{ 
    for (int i = 0; i < NUM_CITIES; ++i)
    {
        for (int j = 0; j < NUM_CITIES; ++j)
        {
            if (i != j)
                cost_table[i*NUM_CITIES+j] = distBetweenCities(tour.cities[i], tour.cities[j]);
            else
                cost_table[i*NUM_CITIES+j] = MAX_COORD * MAX_COORD;
        }
    }
}

// given an array of tours and an array to store the selection, select a tour and store it in parent array
__host__ __device__ tour_t getFittestTour(tour_t *tours, const int &tourneySize)
{
    tour_t fittest = tours[0];
    for (int i = 1; i < tourneySize; ++i)
    {
        if (tours[i].fitness >= fittest.fitness)
            fittest = tours[i];
    }
    return fittest;
}

// ---------------------
// TOURNAMENT SELECTION:
// ---------------------
// note: could improve this by assigning probabilities for tours to be picked
//       based on their fitness level. P = tour_fitness / sum(of all tours fitness)

// chooses best two tours out of randomly selected subpopulation of size TOURNAMENT_SIZE, then puts them in parents array
__device__ tour_t tournamentSelection(population_t *population, curandState *d_state, const int &tid)
{
    tour_t tournament[TOURNAMENT_SIZE];
    
    int randNum;
    for (int i = 0; i < TOURNAMENT_SIZE; ++i)
    {   
        // gets random number from global random state on GPU
        randNum = curand_uniform(&d_state[tid]) * (POPULATION_SIZE - 0.00000001);
        tournament[i] = population->tours[randNum];
    }

    tour_t fittest = getFittestTour(tournament, TOURNAMENT_SIZE);
    return fittest;
}

__device__ int getIndexOfCity(const city_t &city, const tour_t &tour, const int &tourSize)
{
    for (int i = 0; i < tourSize; ++i)
    {
        if (city == tour.cities[i])
            return i;
    }
    return -1;
}

__device__ city_t getCityN(const int &n, const tour_t &tour)
{
    for (int i = 0; i < NUM_CITIES; ++i)
    {
        if (tour.cities[i].n == n)
            return tour.cities[i];
    }

    printf("could not find city %d in this tour: ", n);
    printTour(tour);
    return city_t();
}

__device__ city_t getValidNextCity(tour_t *parents, const tour_t &child, const city_t &currentCity, const int &childSize, const int &pIndex)
{   
    city_t validCity;
    int indexOfCurrentCity = getIndexOfCity(currentCity, parents[pIndex], NUM_CITIES);

    // search for first valid city (not already in child) 
    // occurring after currentCities location in parent tour
    for (int i = indexOfCurrentCity+1; i < NUM_CITIES; ++i)
    {
        // if not in child already, select it!
        if (getIndexOfCity(parents[pIndex].cities[i], child, childSize) == -1)
            return parents[pIndex].cities[i];
    }

    // loop through city id's [1.. NUM_CITIES] and find first valid city
    // to choose as a next point in construction of child tour
    for (int i = 1; i < NUM_CITIES; ++i)
    {
        bool inTourAlready = false;
        for (int j = 1; j < childSize; ++j)
        {
            if (child.cities[j].n == i)
            {
                inTourAlready = true;
                break;
            }
        }

        if (!inTourAlready)
            return getCityN(i,parents[pIndex]);
    }
    
    // THIS SHOULD NOT HAPPEN:
    // ||||||||||||||||||||||
    // VVVVVVVVVVVVVVVVVVVVVV

    // if there is an error:
    printf("no valid city was found for tid %d\nIndex of currentCity: %d\n\n", pIndex/2, indexOfCurrentCity);
    return city_t();
}   

__host__ void checkForError()
{
    // check for error
    cudaError_t errorVar = cudaGetLastError();
    if(errorVar != cudaSuccess)
    {
        // print the CUDA error message and exit
        printf("CUDA error: %s\n", cudaGetErrorString(errorVar));
        exit(-1);
    }   
}