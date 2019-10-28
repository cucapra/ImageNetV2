# -*- coding: utf-8 -*-
"""
    pyeasyga module

"""

import random
import copy
from operator import attrgetter

from six.moves import range
import utils
import numpy as np

class GeneticAlgorithm(object):
    """Genetic Algorithm class.

    This is the main class that controls the functionality of the Genetic
    Algorithm.

    A simple example of usage:

    >>> # Select only two items from the list and maximise profit
    >>> from pyeasyga.pyeasyga import GeneticAlgorithm
    >>> input_data = [('pear', 50), ('apple', 35), ('banana', 40)]
    >>> easyga = GeneticAlgorithm(input_data)
    >>> def fitness (member, data):
    >>>     return sum([profit for (selected, (fruit, profit)) in
    >>>                 zip(member, data) if selected and
    >>>                 member.count(1) == 2])
    >>> easyga.fitness_function = fitness
    >>> easyga.run()
    >>> print easyga.best_individual()

    """

    def __init__(self,
                 seed_data,
                 population_size=50,
                 generations=100,
                 crossover_probability=1,
                 mutation_probability=0.3,
                 elitism=True,
                 maximise_fitness=True):
        """Instantiate the Genetic Algorithm.

        :param seed_data: input data to the Genetic Algorithm
        :type seed_data: list of objects
        :param int population_size: size of population
        :param int generations: number of generations to evolve
        :param float crossover_probability: probability of crossover operation
        :param float mutation_probability: probability of mutation operation

        """

        self.seed_data = seed_data
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.elitism = elitism
        self.maximise_fitness = maximise_fitness

        self.current_generation = []

        def create_individual(seed_data):
            """Create a candidate solution representation.

            e.g. for a bit array representation:

            >>> return [random.randint(0, 1) for _ in range(len(data))]

            :param seed_data: input data to the Genetic Algorithm
            :type seed_data: list of objects
            :returns: candidate solution representation as a list

            """
            return [random.randint(0, 1) for _ in range(len(seed_data))]

        def crossover(parent_1, parent_2):
            """Crossover (mate) two parents to produce two children.

            :param parent_1: candidate solution representation (list)
            :param parent_2: candidate solution representation (list)
            :returns: tuple containing two children

            """
            index = random.randrange(1, len(parent_1))
            child_1 = parent_1[:index] + parent_2[index:]
            child_2 = parent_2[:index] + parent_1[index:]
            return child_1, child_2

        def mutate(individual):
            """Reverse the bit of a random index in an individual."""
            mutate_index = random.randrange(len(individual))
            individual[mutate_index] = (0, 1)[individual[mutate_index] == 0]

        def random_selection(population):
            """Select and return a random member of the population."""
            return random.choice(population)

        def tournament_selection(population):
            """Select a random number of individuals from the population and
            return the fittest member of them all.
            """
            if self.tournament_size == 0:
                self.tournament_size = 2
            members = random.sample(population, self.tournament_size)
            members.sort(
                key=attrgetter('fitness'), reverse=self.maximise_fitness)
            return members[0]

        self.fitness_function = None
        self.tournament_selection = tournament_selection
        self.tournament_size = self.population_size // 10
        self.random_selection = random_selection
        self.create_individual = create_individual
        self.crossover_function = crossover
        self.mutate_function = mutate
        self.selection_function = self.tournament_selection

    def create_initial_population(self):
        """Create members of the first population randomly.
        """
        initial_population = []
        for j in range(self.population_size):
            genes,rank = self.create_individual(self.seed_data,j)
            individual = Chromosome(genes, rank)
            initial_population.append(individual)
        self.current_generation = initial_population

    def calculate_population_fitness(self):
        """Calculate the fitness of every member of the given population using
        the supplied fitness_function.
        """
        for individual in self.current_generation:
            if individual.fitness == 0:
                individual.fitness,individual.info = self.fitness_function(
                    individual.genes, self.seed_data)

    def rank_population(self):
        """Sort the population by fitness according to the order defined by
        maximise_fitness.
        """
        scores = []
        for c in self.current_generation:
            scores.append(c.info)
            c.rank=-1
        scores=np.array(scores)
        for rank in range(5):
            indexs = utils.identify_pareto(scores)
            for ind in indexs:
                self.current_generation[ind].rank = rank
            scores[indexs] = 0
            print(scores)
            if np.count_nonzero(scores)/2 == len(scores): break

        for c in self.current_generation:
            if c.rank==-1: c.rank=5

        self.current_generation.sort(
            key=attrgetter('rank'))#reverse=self.maximise_fitness)
        self.current_generation = self.current_generation[:self.population_size+1]
        #r = 0
        #cnt = 1
        #bins = int((self.population_size-1)/6)+1#6
        #for p in range(0,self.population_size):
        #    self.current_generation[p].rank = r
        #    if cnt == bins:
        #        r += 1
        #        cnt = 0
        #    cnt += 1
        #self.current_generation = self.current_generation[:self.population_size+1]
       

    def create_new_population(self,i):
        """Create a new population using the genetic operators (selection,
        crossover, and mutation) supplied.
        """
        new_population = []
        elite = copy.deepcopy(self.current_generation[0])
        selection = self.selection_function
        j = 0
        while len(new_population) < self.population_size:
            parent_1 = copy.deepcopy(selection(self.current_generation))
            parent_2 = copy.deepcopy(selection(self.current_generation))
            
            child_1, child_2 = parent_1, parent_2
            child_1.fitness, child_2.fitness = 0, 0
            child_1.rank, child_2.rank = 0,0 
            
            can_mutate = random.random() < self.mutation_probability
            if can_mutate:
                qtable_1 = self.mutate_function(parent_1.genes,parent_1.rank,(i,j))
                qtable_2 = self.mutate_function(parent_2.genes,parent_2.rank,(i,j))
            else:
                qtable_1,qtable_2 = parent_1.genes,parent_2.genes
            #can_crossover = random.random() < self.crossover_probability
            #if can_crossover:
            child_1.genes, child_2.genes = self.crossover_function(
                qtable_1, qtable_2, i,j)
            
            new_population.append(child_1)
            if len(new_population) < self.population_size:
                new_population.append(child_2)
            j+=2
        #if self.elitism:
        #    new_population[0] = elite
        for p in range(len(self.current_generation)):
           new_population.append(copy.deepcopy(self.current_generation[p]))

        self.current_generation = new_population

    def create_first_generation(self):
        """Create the first population, calculate the population's fitness and
        rank the population by fitness according to the order specified.
        """
        self.create_initial_population()
        self.calculate_population_fitness()
        self.rank_population()

    def create_next_generation(self,i):
        """Create subsequent populations, calculate the population fitness and
        rank the population by fitness in the order specified.
        """
        self.create_new_population(i)
        self.calculate_population_fitness()
        self.rank_population()

    def run(self):
        """Run (solve) the Genetic Algorithm."""
        self.create_first_generation()

        for i in range(1, self.generations):
            self.create_next_generation(i)

    def best_individual(self):
        """Return the individual with the best fitness in the current
        generation.
        """
        best = self.current_generation[0]
        return (best.fitness, best.info, best.genes)

    def last_generation(self):
        """Return members of the last generation as a generator function."""
        return ((member.fitness, member.info ,member.genes) for member
                in self.current_generation)


class Chromosome(object):
    """ Chromosome class that encapsulates an individual's fitness and solution
    representation.
    """
    def __init__(self, genes,rank):
        """Initialise the Chromosome."""
        self.genes = genes
        self.fitness = 0
        self.rank = rank
        self.info = ()

    def __repr__(self):
        """Return initialised Chromosome representation in human readable form.
        """
        return repr((self.fitness, self.rank, self.info,self.genes))

