// definition for individual city (really just 2d point)

// DEFINES: City data type
struct city_t{
    float x;
    float y;
    int n;
    __host__ __device__ city_t() { x = -1; y = -1; n = -1; }
    __host__ __device__ city_t(int xPos, int yPos, int num) { x = xPos; y = yPos; n = num; }
    __host__ __device__ city_t &operator=(const city_t &a)
    {
		x=a.x;    	
	    y=a.y;
        n=a.n;
        return *this;
    }
    __host__ __device__ bool operator==(const city_t &rhs)const{ return (x == rhs.x && y == rhs.y && n == rhs.n); }
};

__host__ __device__ float distBetweenCities(const city_t &city1, const city_t &city2)
{
    float xDist = pow(city1.x - city2.x,2);
    float yDist = pow(city1.y - city2.y,2);
    return sqrt(xDist+yDist);
}

