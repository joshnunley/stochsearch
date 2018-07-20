Evolutionary Algorithm
=========================
An evolutionary algorithm is a stochastic search based optimization technique. It is a population based method, where the optimization starts with a population of random solutions (individuals or genotypes). Each individual is assigned a fitness score based on how well they perform in the task at hand. Based on this fitness, a fraction of the best performing individuals are retained for the next iteration (generation). a new population of solutions is then created for the next generation with these 'elite' individuals and copies of them that have been subjected to mutation noise. This process is repeated either for a fixed number of generations, or until a desired fitness value is reached by the best individual in the population. In a non-stochastic system, this procedure will cause the fitness to be non-decreasing over generations. For those familiar with hill climbing, this approach can be seen as multiple hill climbers searching in parallel, where the number of hill climbers would be given by the elitist fraction of the population that are retained generation after generation. This same implementation can be used to perform hill-climbing if the elitist fraction is set such that elitist_fraction*population_size = 1.

This Python stochastic search package, stochsearch, includes an implementation of evolutionary algorithms in a class called EvolSearch using the Python multiprocessing framework. Fitness evaluation of individuals in a population is carried out in parallel across CPUs in a multiprocessing.pool.Pool with the number of processes defined by the user or by os.cpu_count() of the system. This package can be imported as follows "from stochsearch import EvolSearch".

Installation
---------------
        >> pip install stochsearch

Usage
---------------
The EvolSearch class is defined as follows

        class EvolSearch()
         |
         |  __init__(self, evol_params)
         |      Initialize evolutionary search
         |      ARGS:
         |      evol_params: dict
         |          required keys -
         |              pop_size: int - population size,
         |              genotype_size: int - genotype_size,
         |              fitness_function: function - a user-defined function that takes a genotype as arg and returns a float fitness value
         |              elitist_fraction: float - fraction of top performing individuals to retain for next generation
         |              mutation_variance: float - variance of the gaussian distribution used for mutation noise
         |          optional keys -
         |              num_processes: int -  pool size for multiprocessing.pool.Pool - defaults to os.cpu_count()
         |
         |  elitist_selection(self)
         |      from fitness select top performing individuals based on elitist_fraction
         |
         |  evaluate_fitness(self, individual_index)
         |      Call user defined fitness function and pass genotype
         |
         |  execute_search(self, num_gens)
         |      runs the evolutionary algorithm for given number of generations, num_gens
         |
         |  get_best_individual(self)
         |      returns 1D array of the genotype that has max fitness
         |
         |  get_best_individual_fitness(self)
         |      return the fitness value of the best individual
         |
         |  get_fitness_variance(self)
         |      returns variance of the population's fitness
         |
         |  get_mean_fitness(self)
         |      returns the mean fitness of the population
         |
         |  mutation(self)
         |      create new pop by repeating mutated copies of elitist individuals
         |
         |  step_generation(self)
         |      evaluate fitness of pop, and create new pop after elitist_selection and mutation

In order to use this package
1. from stochsearch import EvolSearch
2. Write your own fitness evaluation function that takes a genotype as argument and return its fitness
3. Create an object of EvolSearch by passing a dict of evol_params with all required keys, say es
4. Do one of the following:
    a. call es.execute_search(number_of_generations)
    b. while es.get_best_individual_fitness() < desired_fitness and num_gens less than man_num_gens, call es.step_generation()
5. Access best individual using es.get_best_individual() to perhaps save solution to a file
