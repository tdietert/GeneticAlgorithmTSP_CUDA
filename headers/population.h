// DEFINES: Population as array of tours
struct population_t{
    tour_t tours[POPULATION_SIZE];

    __host__ __device__ population_t &operator=(const population_t &a)
    {
        for (int i = 0; i < POPULATION_SIZE; ++i)
        {
            tours[i] = a.tours[i];
        }
        return *this;
    }
};

__device__ int getIndexOfTour(tour_t *tours)
{
    for (int i = 0; i < POPULATION_SIZE; ++i)
    {
        
    }
    return 0;
}

// creates population of tours, all randomized versions of initialTour
void initializePop(population_t &initialPop, tour_t &initialTour, float *costTable) 
{
    initialPop.tours[0] = initialTour;
    for (int i = 1; i < POPULATION_SIZE; ++i)
    {
        for (int j = 1; j < NUM_CITIES; ++j)
        {
            int randPos = 1 + (rand() % (NUM_CITIES-1));
            std::swap(initialTour.cities[j], initialTour.cities[randPos]);
        }

        initialPop.tours[i] = initialTour;
        evalTour(initialPop.tours[i], costTable);
    }
}

void printPopulation(population_t &pop)
{
    for (int i = 0; i < POPULATION_SIZE; ++i)
    {
        std::cout << "Individual " << i << std::endl;
        std::cout << "> fitness: " << pop.tours[i].fitness << std::endl;
        std::cout << "> cities:  ";
        for (int j = 0; j < NUM_CITIES; ++j)
        {
            std::cout << pop.tours[i].cities[j].n << " ";
        }
        std::cout << std::endl;
    }
}