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
__host__ __device__ void buildCostTable(tour_t &tour, float (&cost_table)[NUM_CITIES*NUM_CITIES])
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

// given an array of tours, choose fittest and return it
__host__ __device__ tour_t getFittestTour(tour_t *tours, const int &populationSize)
{
    tour_t fittest = tours[0];
    for (int i = 1; i < populationSize; ++i)
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
__device__ tour_t tournamentSelection(population_t &population, curandState *d_state, const int &tid)
{
    tour_t tournament[TOURNAMENT_SIZE];
    
    int randNum;
    for (int i = 0; i < TOURNAMENT_SIZE; ++i)
    {   
        // gets random number from global random state on GPU
        randNum = curand_uniform(&d_state[tid]) * (POPULATION_SIZE - 1);
        tournament[i] = population.tours[randNum];
    }

    tour_t fittest = getFittestTour(tournament, TOURNAMENT_SIZE);
    return fittest;
}

__device__ int getIndexOfCity(city_t &city, tour_t &tour, const int &tourSize)
{
    for (int i = 0; i < tourSize; ++i)
    {
        if (city == tour.cities[i])
            return i;
    }
    return -1;
}

__device__ city_t getCityN(int &n, tour_t &tour)
{
    for (int i = 0; i < NUM_CITIES; ++i)
    {
        if (tour.cities[i].n == n)
            return tour.cities[i];
    }

    printf("%d, %d", blockIdx.x, threadIdx.x);
    printf("could not find city %d in this tour: ", n);
    printTour(tour);
    return city_t();
}

__device__ city_t getValidNextCity(tour_t &parent, tour_t &child, city_t &currentCity, const int &childSize)
{   
    city_t validCity;
    int indexOfCurrentCity = getIndexOfCity(currentCity, parent, NUM_CITIES);

    // search for first valid city (not already in child) 
    // occurring after currentCities location in parent tour
    for (int i = indexOfCurrentCity+1; i < NUM_CITIES; ++i)
    {
        // if not in child already, select it!
        if (getIndexOfCity(parent.cities[i], child, childSize) == -1)
            return parent.cities[i];
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
            return getCityN(i,parent);
    }
    
    // THIS SHOULD NOT HAPPEN:
    // ||||||||||||||||||||||
    // VVVVVVVVVVVVVVVVVVVVVV

    // if there is an error:
    printf("no valid city was found!\n\n");
    return city_t();
}   

__device__ int getIndexOfLeastFit(population_t &population)
{
    tour_t leastFit = population.tours[0];
    int index = 0;
    for (int i = 1; i < POPULATION_SIZE; ++i)
    {
        if (population.tours[i].fitness <= leastFit.fitness)
        {
            leastFit = population.tours[i];
            index = i;
        }
    }
    return index; 
}

__device__ bool checkTour(const tour_t &tour)
{
    bool seen[NUM_CITIES];
    for (int i = 0; i < NUM_CITIES; ++i)
        seen[i] = true;
    
    for (int i = 0; i < NUM_CITIES; ++i)
    {
        if (!seen[tour.cities[i].n])
            return false;
        else
            seen[tour.cities[i].n] = false;
    }

    return true;
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

//----------------------------
//         HEAPSORT
//----------------------------
__device__ void sortPopulation(tour_t *tours)  
{  
    unsigned int n = POPULATION_SIZE, i = n/2, parent, child;  
    tour_t t;  
  
    while(true) /* Loops until arr is sorted */  
    { 
        if (i > 0)
        {
            i--;           /* Save its index to i */  
            t = tours[i];    /* Save parent value to t */  
        } 
        else 
        {     /* Second stage - Extracting elements in-place */  
            n--;           /* Make the new heap smaller */  
            if (n == 0) return; /* When the heap is empty, we are done */  
            t = tours[0];    /* Save last value (it will be overwritten) */  
            tours[0] = tours[n]; /* Save largest value at the end of tours */  
        }  
  
        parent = i; /* We will start pushing down t from parent */  
        child = i*2 + 1; /* parent's left child */  
  
        /* Sift operation - pushing the value of t down the heap */  
        while (child < n) {  
            if (child + 1 < n  &&  tours[child + 1].distance > tours[child].distance)  
                child++; /* Choose the largest child */  
              
            if (tours[child].distance > t.distance) { /* If any child is bigger than the parent */  
                tours[parent] = tours[child]; /* Move the largest child up */  
                parent = child; /* Move parent pointer to this child */  
                //child = parent*2-1; /* Find the next child */  
                child = parent*2+1; /* the previous line is wrong*/  
            }
            else 
                break; /* t's place is found */  
              
        }  
        tours[parent] = t; /* We save t in the heap */  
    }  
} 