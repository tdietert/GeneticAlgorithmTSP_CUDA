// DEFINES: Tour data type (array of cities, with attrs fitness and dist)
struct tour_t{
    __host__ __device__ tour_t() 
    { 
        for (int i = 0; i < NUM_CITIES; ++i)
        {
            cities[i] = city_t(-1,-1,-1);
        } 
        fitness = 0;
        distance = 0;
    }


    __host__ __device__ tour_t &operator=(const tour_t &a)
    {
        for (int i = 0; i < NUM_CITIES; ++i)
        {
            cities[i] = a.cities[i];
        }
        fitness = a.fitness;
        distance = a.distance;
        return *this;
    }

    __host__ __device__ bool operator==(tour_t &rhs)
    { 
        for (int i = 0; i < NUM_CITIES; ++i)
        {
            if (cities[i].x != rhs.cities[i].x || cities[i].y != rhs.cities[i].y)
                return false;
        }
        return true; 
    }

    city_t cities[NUM_CITIES];
    float fitness;
    float distance;
};

__host__ __device__ void evalTour(tour_t &tour, const float *costTable)
{
    tour.distance = 0;
    for (int i = 0; i < NUM_CITIES; ++i)
    {
        if (i < NUM_CITIES - 1)
            tour.distance += costTable[(tour.cities[i].n)*NUM_CITIES + (tour.cities[i+1]).n];
        else 
            tour.distance += costTable[(tour.cities[i].n)*NUM_CITIES + (tour.cities[0]).n];
    }

    if (tour.distance != 0)
        tour.fitness = 1/(tour.distance);
    else 
        tour.fitness = 0;
}

void initializeRandomTour(tour_t &tour)
{
    // only randomizes the tail of the tour, so all
    // tours start at the same city.
    tour.cities[0] = city_t(0, 0, 0);
    for (int i = 1; i < NUM_CITIES; ++i)
    { 
        int randX = rand() % MAX_COORD;
        int randY = rand() % MAX_COORD;
        tour.cities[i] = city_t(randX, randY, i);
    }
}

__host__ __device__ void printTour(const tour_t &tour)
{  
    for (int i = 0; i < NUM_CITIES; ++i)
    {
        printf("%d ", tour.cities[i].n + 1);
    }
    printf("\n");
}
