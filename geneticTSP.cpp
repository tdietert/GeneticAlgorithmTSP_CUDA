#include <stdlib.h>
#include <stdio.h>
#include <ctime>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <vector>

#include "city.h"
#include "tour.h"
#include "population.h"

using namespace std;

#define NUM_THREADS 256
#define NUM_CITIES 15
#define MAX_COORD 200
#define POPULATION_SIZE 50

int main(int argc, char **argv)
{
	srand(time(NULL));

	Tour initialTour = Tour(NUM_CITIES, MAX_COORD);
	Population population(POPULATION_SIZE, initialTour);
	population.printPopulation();

}

