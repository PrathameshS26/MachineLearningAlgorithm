# -*- coding: utf-8 -*-
"""
Created on Mon Nov 6 22:31:23 2023
@author: Prathamesh
"""
from bs4 import BeautifulSoup as bs  # Import BeautifulSoup from bs4 library with alias bs
import lxml  # Import lxml library for XML parsing
# Import ElementTree module from xml.etree.ElementTree with alias ET
import xml.etree.ElementTree as ET
from random import choice  # Import the choice function from the random module
import random  # Import the random module
import numpy as np  # Import numpy library with alias np
import time  # Import the time module
import glob  # Import the glob module for file path manipulation
# Import the pyplot module from matplotlib with alias plt
import matplotlib.pyplot as plt
import pandas as pd  # Import the pandas library with alias pd

EXPERIMENT_RUNS = 1  # Number of times the experiment should run

# Custom exception class to handle the termination of a run based on the number of fitness evaluations


class TerminationException(Exception):
    pass


# Parse the XML file for Burma
burma_tree = ET.parse(
    "C:/Users/Prathamesh/OneDrive/Desktop/NIC/CSVFiles/burma14.xml")
burma_root = burma_tree.getroot()

# Find the graph element
burma_graph = burma_root.find('graph')

# Initialize variables for Burma
burma_num_vertices = int(burma_root.find('description').text.split('-')[0])
burma_precision = int(burma_root.find('doublePrecision').text)
burma_cost_matrix = np.zeros((burma_num_vertices, burma_num_vertices))

# Initialize vertex_id for Burma
vertex_id = 0

# Iterate through the edge elements and populate the cost matrix for Burma
for vertex in burma_graph.findall('vertex'):
    for edge in vertex.findall('edge'):
        target_vertex = int(edge.text)
        cost = float(edge.attrib['cost'])
        burma_cost_matrix[vertex_id, target_vertex] = cost

    # Increment vertex_id
    vertex_id += 1

# Print the cost matrix for Burma (optional)
print(burma_cost_matrix)

num_burma_cities = burma_num_vertices

# Parse the XML file for Brazil
brazil_tree = ET.parse(
    "C:/Users/Prathamesh/OneDrive/Desktop/NIC/CSVFiles/brazil58.xml")
brazil_root = brazil_tree.getroot()

# Find the graph element
brazil_graph = brazil_root.find('graph')

# Initialize variables for Brazil
brazil_num_vertices = int(brazil_root.find('description').text.split()[0])
brazil_precision = int(brazil_root.find('doublePrecision').text)
brazil_cost_matrix = np.zeros((brazil_num_vertices, brazil_num_vertices))

# Initialize vertex_id for Brazil
vertex_id = 0

# Iterate through the edge elements and populate the cost matrix for Brazil
for vertex in brazil_graph.findall('vertex'):
    for edge in vertex.findall('edge'):
        target_vertex = int(edge.text)
        cost = float(edge.attrib['cost'])
        brazil_cost_matrix[vertex_id, target_vertex] = cost

    # Increment vertex_id
    vertex_id += 1

# Print the cost matrix for Brazil (optional)
print(brazil_cost_matrix)

num_brazil_cities = brazil_num_vertices

# Generate random tour depending on the number of cities from each country


def generateTour(num_cities):
    tour = random.sample(range(1, num_cities + 1), num_cities)
    return tour


# Set population size and mutation rate
population_size = 100
mutation_rate = 0.1

# Generating based on the population size


def generateRandomPopulation(num_cities):
    population = []
    for _ in range(population_size):
        population.append(generateTour(num_cities))
    return population

# Evaluating the fitness value for the individual gene


def generatefitness(individual, precision, country):
    global generation

    if country == "Burma":
        if generation > 0:
            cost_matrix = burma_cost_matrix
            cost = 0
            for i in range(len(individual) - 1):
                cost += cost_matrix[individual[i] - 1][individual[i + 1] - 1]
            cost += cost_matrix[individual[-1] - 1][individual[0] - 1]
            generation -= 1
            return round(1/cost, precision)
        else:
            raise TerminationException
    elif country == "Brazil":
        if generation > 0:
            cost_matrix = brazil_cost_matrix
            cost = 0
            for i in range(len(individual) - 1):
                cost += cost_matrix[individual[i] - 1][individual[i + 1] - 1]
            cost += cost_matrix[individual[-1] - 1][individual[0] - 1]
            generation -= 1
            return round(1/cost, precision)
        else:
            raise TerminationException

# Function to generate the fitness of the entire population


def generatePopulationFitness(population, precision, country):
    fitness = []
    for element in population:
        fitness.append(generatefitness(element, precision, country))
    return fitness

# Function to perform swap mutation on the input tour


def swapMutate(tour, num_cities):
    if random.random() < mutation_rate:
        # Swap two random cities in the tour
        idx1, idx2 = random.sample(range(num_cities), 2)
        tour[idx1], tour[idx2] = tour[idx2], tour[idx1]
    return tour

# Function to perform inversion mutation on the input tour


def inversionMutation(tour):
    # Select two random positions in the chromosome
    pos1, pos2 = sorted(random.sample(range(len(tour)), 2))
    print(pos1, pos2)
    # Perform inversion mutation
    segment_to_invert = tour[pos1:pos2 + 1]
    inverted_segment = segment_to_invert[::-1]

    # Update the chromosome with the inverted segment
    tour[pos1:pos2 + 1] = inverted_segment

    return tour

# Function to perform single-point crossover using two parents


def singlePointCrossover(parent1, parent2):
    # Choosing a random point in the parent
    point = np.random.randint(1, len(parent1))
    # Splitting the chromosome and adding the distinct part together
    parent1_left = parent1[:point]
    parent1_right = parent1[point:]
    parent2_left = parent2[:point]
    parent2_right = parent2[point:]
    child1 = parent1_left + parent2_right
    child2 = parent2_left + parent1_right
    # Find missing and conflicting data in parent1
    parent1_missing = [i for i in parent1 if i not in child1]
    parent1_conflict = [i for i in parent1_left if i in parent2_right]
    # Fix parent1 by replacing the conflicting data with the missing elements
    for element in child1:
        if element in parent1_conflict:
            copy = element
            child1[child1.index(element)] = choice(
                [i for i in parent1_missing if i not in child1])
            parent1_conflict.remove(copy)

    # Find missing and conflicting data in parent2
    parent2_missing = [i for i in parent2 if i not in child2]
    parent2_conflict = [i for i in parent2_left if i in parent1_right]
    # Fix parent2 by replacing the conflicting data with the missing elements
    for element in child2:
        if element in parent2_conflict:
            copy = element
            child2[child2.index(element)] = choice(
                [i for i in parent2_missing if i not in child2])
            parent2_conflict.remove(copy)

    return child1, child2

# Function to perform ordered crossover using two parents


def orderedCrossover(parent1, parent2):
    size = len(parent1)

    # Choose random start/end position for crossover
    child1, child2 = [-1] * size, [-1] * size
    start, end = sorted([random.randrange(size) for _ in range(2)])

    # Replicate parent1's sequence for child1, parent2's sequence for child2
    child1_inherited = []
    child2_inherited = []
    for i in range(start, end + 1):
        child1[i] = parent1[i]
        child2[i] = parent2[i]
        child1_inherited.append(parent1[i])
        child2_inherited.append(parent2[i])

    print(child1, child2)
    # Fill the remaining positions with the other parents' entries
    current_parent2_position, current_parent1_position = 0, 0

    fixed_pos = list(range(start, end + 1))
    i = 0
    while i < size:
        if i in fixed_pos:
            i += 1
            continue

        test_child1 = child1[i]
        if test_child1 == -1:
            parent2_trait = parent2[current_parent2_position]
            while parent2_trait in child1_inherited:
                current_parent2_position += 1
                parent2_trait = parent2[current_parent2_position]
            child1[i] = parent2_trait
            child1_inherited.append(parent2_trait)

        test_child2 = child2[i]
        if test_child2 == -1:
            parent1_trait = parent1[current_parent1_position]
            while parent1_trait in child2_inherited:
                current_parent1_position += 1
                parent1_trait = parent1[current_parent1_position]
            child2[i] = parent1_trait
            child2_inherited.append(parent1_trait)

        i += 1

    return child1, child2

# Function to perform tournament selection


def tournament_selection(population, size, country, precision):
    tournament = []
    # Choose the number (size) from the population and add that to the tournament list
    for i in range(size):
        tournament.append(choice([solution for solution in population]))
    # The winner of the tournament is the solution with the best fitness
    # parent = max(tournament, key=lambda solution: generatefitness(
    #   solution, precision, country))

    parent = max(tournament)
    return parent

# Function to perform rank-based selection


def rankBasedSelection(population, pop_fitness):
    # Combine the population and its fitness
    fitness_population = sorted(zip(pop_fitness, population))
    # Sort the population and fitness
    sorted_population = [x for y, x in fitness_population]
    sorted_pop_fitness = [y for y, x in fitness_population]
    rank = []
    # Assign ranks to the solutions in the population
    for i in range(len(sorted_population)):
        rank.append(i+1)
    rank_total = sum(rank)
    probability = []
    # Find the probabilities of choosing a solution from the population based on its rank
    for r in rank:
        probability.append(r/rank_total)
    indeces = np.arange(len(sorted_population))
    # Choose 2 parents randomly based on the probabilities
    choice_1, choice_2 = np.random.choice(
        indeces, 2, replace=True, p=probability)
    parent_1, parent_2 = sorted_population[choice_1], sorted_population[choice_2]
    return parent_1, parent_2

# Function to perform roulette wheel selection


def rouletteWheelSelection(population, pop_fitness):
    total_fitness = sum(pop_fitness)

    # Select the first parent
    selection_point1 = random.uniform(0, total_fitness)
    cumulative_fitness = 0
    for i, fitness in enumerate(pop_fitness):
        cumulative_fitness += fitness
        if cumulative_fitness >= selection_point1:
            parent1 = population[i]
            break

    # Select the second parent
    selection_point2 = random.uniform(0, total_fitness)
    cumulative_fitness = 0
    for i, fitness in enumerate(pop_fitness):
        cumulative_fitness += fitness
        if cumulative_fitness >= selection_point2:
            parent2 = population[i]
            break

    return parent1, parent2

# Function to replace the weakest individuals in the population


def replaceWeakest(population, population_fitness, child, child_fitness, precision):
    # Search for the worst fitness in the population and store it
    worst_fitness_ind = min(population_fitness)
    # Replace the worst individual if the new candidate individual is better or the same
    if round(child_fitness, precision) >= round(worst_fitness_ind, precision):
        list_index = population_fitness.index(worst_fitness_ind)
        del population_fitness[list_index]
        del population[list_index]
        population.append(child)
        population_fitness.append(child_fitness)
    return population, population_fitness

# Function to calculate the average fitness of the population


def calculateAverageFitness(population_fitness, size):
    return sum(population_fitness) / size


# Function to calculate the best solution and its fitness from a given population and its fitness values
def calculateBestSolution(population, population_fitness):
    best_fitness = max(population_fitness)
    list_index = population_fitness.index(best_fitness)
    return population[list_index], best_fitness


# Lists to store historical data for tournament selection
exp1_best_fitness_tournament_selection_history = []

for run in range(EXPERIMENT_RUNS):
    POPULATION_SIZE = 100
    Tournament_size = 10
    generation = 10000

    try:
        population = generateRandomPopulation(num_burma_cities)
        population_fitness = generatePopulationFitness(
            population, burma_precision, "Burma")
        while True:

            a = tournament_selection(
                population, Tournament_size,  "Burma", burma_precision)
            b = tournament_selection(
                population, Tournament_size,  "Burma", burma_precision)
            c, d = singlePointCrossover(a, b)
            e = swapMutate(c, num_burma_cities)
            e_fitness = generatefitness(e, burma_precision, "Burma")
            f = swapMutate(d, num_burma_cities)
            f_fitness = generatefitness(f, burma_precision, "Burma")
            population, population_fitness = replaceWeakest(
                population, population_fitness, e, e_fitness, burma_precision)
            population, population_fitness = replaceWeakest(
                population, population_fitness, f, f_fitness, burma_precision)
            best_solution, best_fitness = calculateBestSolution(
                population, population_fitness)
            best_fitness = str(
                "{:." + str(burma_precision) + "f}").format(best_fitness)
            best_solution = ' '.join(str(g) for g in best_solution)

            exp1_best_fitness_tournament_selection_history.append(
                float(best_fitness))

    except TerminationException:
        print("---TERMINATION CONDITION REACHED---")

    print(calculateBestSolution(population, population_fitness))

# Lists to store historical data for tournament selection
exp2_best_fitness_tournament_selection_history = []

for run in range(EXPERIMENT_RUNS):
    POPULATION_SIZE = 200
    Tournament_size = 20
    generation = 10000

    try:
        population = generateRandomPopulation(num_burma_cities)
        population_fitness = generatePopulationFitness(
            population, burma_precision, "Burma")
        while True:

            a = tournament_selection(
                population, Tournament_size,  "Burma", burma_precision)
            b = tournament_selection(
                population, Tournament_size,  "Burma", burma_precision)
            c, d = singlePointCrossover(a, b)
            e = swapMutate(c, num_burma_cities)
            e_fitness = generatefitness(e, burma_precision, "Burma")
            f = swapMutate(d, num_burma_cities)
            f_fitness = generatefitness(f, burma_precision, "Burma")
            population, population_fitness = replaceWeakest(
                population, population_fitness, e, e_fitness, burma_precision)
            population, population_fitness = replaceWeakest(
                population, population_fitness, f, f_fitness, burma_precision)
            best_solution, best_fitness = calculateBestSolution(
                population, population_fitness)
            best_fitness = str(
                "{:." + str(burma_precision) + "f}").format(best_fitness)
            best_solution = ' '.join(str(g) for g in best_solution)

            exp2_best_fitness_tournament_selection_history.append(
                float(best_fitness))

    except TerminationException:
        print("---TERMINATION CONDITION REACHED---")

    print(calculateBestSolution(population, population_fitness))

 # Lists to store historical data for tournament selection
exp3_best_fitness_tournament_selection_history = []

for run in range(EXPERIMENT_RUNS):
    POPULATION_SIZE = 500
    Tournament_size = 50
    generation = 10000

    try:
        population = generateRandomPopulation(num_burma_cities)
        population_fitness = generatePopulationFitness(
            population, burma_precision, "Burma")
        while True:

            a = tournament_selection(
                population, Tournament_size,  "Burma", burma_precision)
            b = tournament_selection(
                population, Tournament_size,  "Burma", burma_precision)
            c, d = singlePointCrossover(a, b)
            e = swapMutate(c, num_burma_cities)
            e_fitness = generatefitness(e, burma_precision, "Burma")
            f = swapMutate(d, num_burma_cities)
            f_fitness = generatefitness(f, burma_precision, "Burma")
            population, population_fitness = replaceWeakest(
                population, population_fitness, e, e_fitness, burma_precision)
            population, population_fitness = replaceWeakest(
                population, population_fitness, f, f_fitness, burma_precision)

            best_solution, best_fitness = calculateBestSolution(
                population, population_fitness)
            best_fitness = str(
                "{:." + str(burma_precision) + "f}").format(best_fitness)
            best_solution = ' '.join(str(g) for g in best_solution)

            exp3_best_fitness_tournament_selection_history.append(
                float(best_fitness))

    except TerminationException:
        print("---TERMINATION CONDITION REACHED---")

    print(calculateBestSolution(population, population_fitness))

   # Lists to store historical data for tournament selection

exp4_best_fitness_tournament_selection_history = []

for run in range(EXPERIMENT_RUNS):
    POPULATION_SIZE = 1000
    Tournament_size = 100
    generation = 10000

    try:
        population = generateRandomPopulation(num_burma_cities)
        population_fitness = generatePopulationFitness(
            population, burma_precision, "Burma")
        while True:

            a = tournament_selection(
                population, Tournament_size,  "Burma", burma_precision)
            b = tournament_selection(
                population, Tournament_size,  "Burma", burma_precision)
            c, d = singlePointCrossover(a, b)
            e = swapMutate(c, num_burma_cities)
            e_fitness = generatefitness(e, burma_precision, "Burma")
            f = swapMutate(d, num_burma_cities)
            f_fitness = generatefitness(f, burma_precision, "Burma")
            population, population_fitness = replaceWeakest(
                population, population_fitness, e, e_fitness, burma_precision)
            population, population_fitness = replaceWeakest(
                population, population_fitness, f, f_fitness, burma_precision)

            best_solution, best_fitness = calculateBestSolution(
                population, population_fitness)
            best_fitness = str(
                "{:." + str(burma_precision) + "f}").format(best_fitness)
            best_solution = ' '.join(str(g) for g in best_solution)

            exp4_best_fitness_tournament_selection_history.append(
                float(best_fitness))

    except TerminationException:
        print("---TERMINATION CONDITION REACHED---")

    print(calculateBestSolution(population, population_fitness))

# Plotting the best fitness over generations
plt.figure(figsize=(10, 5))
plt.plot(range(len(exp1_best_fitness_tournament_selection_history)),
         exp1_best_fitness_tournament_selection_history, label='Experient 1', color='red')
plt.plot(range(len(exp2_best_fitness_tournament_selection_history)),
         exp2_best_fitness_tournament_selection_history, label='Experient 2', color='blue')
plt.plot(range(len(exp3_best_fitness_tournament_selection_history)),
         exp3_best_fitness_tournament_selection_history, label='Experient 3', color='green')
plt.plot(range(len(exp4_best_fitness_tournament_selection_history)),
         exp4_best_fitness_tournament_selection_history, label='Experient 4', color='yellow')
plt.title('Best Fitness Over Generations (Burma DataSet)')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend()
plt.show()


exp1_brazil_best_fitness_tournament_selection_history = []

for run in range(EXPERIMENT_RUNS):
    POPULATION_SIZE = 100
    Tournament_size = 10
    generation = 10000

    try:
        population = generateRandomPopulation(num_brazil_cities)
        population_fitness = generatePopulationFitness(
            population, burma_precision, "Brazil")
        while True:

            a = tournament_selection(
                population, num_burma_cities,  "Brazil", brazil_precision)
            b = tournament_selection(
                population, num_burma_cities,  "Brazil", brazil_precision)
            c, d = singlePointCrossover(a, b)
            e = swapMutate(c, num_brazil_cities)
            e_fitness = generatefitness(e, brazil_precision, "Brazil")
            f = swapMutate(d, num_brazil_cities)
            f_fitness = generatefitness(f, brazil_precision, "Brazil")
            population, population_fitness = replaceWeakest(
                population, population_fitness, e, e_fitness, brazil_precision)
            population, population_fitness = replaceWeakest(
                population, population_fitness, f, f_fitness, brazil_precision)

            best_solution, best_fitness = calculateBestSolution(
                population, population_fitness)
            best_fitness = str(
                "{:." + str(burma_precision) + "f}").format(best_fitness)
            best_solution = ' '.join(str(g) for g in best_solution)

            exp1_brazil_best_fitness_tournament_selection_history.append(
                float(best_fitness))

    except TerminationException:
        print("---TERMINATION CONDITION REACHED---")

    print(calculateBestSolution(population, population_fitness))

exp2_brazil_best_fitness_tournament_selection_history = []

for run in range(EXPERIMENT_RUNS):
    POPULATION_SIZE = 200
    Tournament_size = 20
    generation = 10000

    try:
        population = generateRandomPopulation(num_brazil_cities)
        population_fitness = generatePopulationFitness(
            population, burma_precision, "Brazil")
        while True:

            a = tournament_selection(
                population, num_burma_cities,  "Brazil", brazil_precision)
            b = tournament_selection(
                population, num_burma_cities,  "Brazil", brazil_precision)
            c, d = singlePointCrossover(a, b)
            e = swapMutate(c, num_brazil_cities)
            e_fitness = generatefitness(e, brazil_precision, "Brazil")
            f = swapMutate(d, num_brazil_cities)
            f_fitness = generatefitness(f, brazil_precision, "Brazil")
            population, population_fitness = replaceWeakest(
                population, population_fitness, e, e_fitness, brazil_precision)
            population, population_fitness = replaceWeakest(
                population, population_fitness, f, f_fitness, brazil_precision)

            best_solution, best_fitness = calculateBestSolution(
                population, population_fitness)
            best_fitness = str(
                "{:." + str(burma_precision) + "f}").format(best_fitness)
            best_solution = ' '.join(str(g) for g in best_solution)

            exp2_brazil_best_fitness_tournament_selection_history.append(
                float(best_fitness))

    except TerminationException:
        print("---TERMINATION CONDITION REACHED---")

    print(calculateBestSolution(population, population_fitness))

exp3_brazil_best_fitness_tournament_selection_history = []

for run in range(EXPERIMENT_RUNS):
    POPULATION_SIZE = 500
    Tournament_size = 50
    generation = 10000

    try:
        population = generateRandomPopulation(num_brazil_cities)
        population_fitness = generatePopulationFitness(
            population, burma_precision, "Brazil")
        while True:

            a = tournament_selection(
                population, num_burma_cities,  "Brazil", brazil_precision)
            b = tournament_selection(
                population, num_burma_cities,  "Brazil", brazil_precision)
            c, d = singlePointCrossover(a, b)
            e = swapMutate(c, num_brazil_cities)
            e_fitness = generatefitness(e, brazil_precision, "Brazil")
            f = swapMutate(d, num_brazil_cities)
            f_fitness = generatefitness(f, brazil_precision, "Brazil")
            population, population_fitness = replaceWeakest(
                population, population_fitness, e, e_fitness, brazil_precision)
            population, population_fitness = replaceWeakest(
                population, population_fitness, f, f_fitness, brazil_precision)

            best_solution, best_fitness = calculateBestSolution(
                population, population_fitness)
            best_fitness = str(
                "{:." + str(burma_precision) + "f}").format(best_fitness)
            best_solution = ' '.join(str(g) for g in best_solution)

            exp3_brazil_best_fitness_tournament_selection_history.append(
                float(best_fitness))

    except TerminationException:
        print("---TERMINATION CONDITION REACHED---")

    print(calculateBestSolution(population, population_fitness))

exp4_brazil_best_fitness_tournament_selection_history = []

for run in range(EXPERIMENT_RUNS):
    POPULATION_SIZE = 1000
    Tournament_size = 100
    generation = 10000

    try:
        population = generateRandomPopulation(num_brazil_cities)
        population_fitness = generatePopulationFitness(
            population, burma_precision, "Brazil")
        while True:

            a = tournament_selection(
                population, num_burma_cities,  "Brazil", brazil_precision)
            b = tournament_selection(
                population, num_burma_cities,  "Brazil", brazil_precision)
            c, d = singlePointCrossover(a, b)
            e = swapMutate(c, num_brazil_cities)
            e_fitness = generatefitness(e, brazil_precision, "Brazil")
            f = swapMutate(d, num_brazil_cities)
            f_fitness = generatefitness(f, brazil_precision, "Brazil")
            population, population_fitness = replaceWeakest(
                population, population_fitness, e, e_fitness, brazil_precision)
            population, population_fitness = replaceWeakest(
                population, population_fitness, f, f_fitness, brazil_precision)

            best_solution, best_fitness = calculateBestSolution(
                population, population_fitness)
            best_fitness = str(
                "{:." + str(burma_precision) + "f}").format(best_fitness)
            best_solution = ' '.join(str(g) for g in best_solution)

            exp4_brazil_best_fitness_tournament_selection_history.append(
                float(best_fitness))

    except TerminationException:
        print("---TERMINATION CONDITION REACHED---")

    print(calculateBestSolution(population, population_fitness))


# Plotting the best fitness over generations
plt.figure(figsize=(10, 5))
plt.plot(range(len(exp1_brazil_best_fitness_tournament_selection_history)),
         exp1_brazil_best_fitness_tournament_selection_history, label='Experient 1', color='red')
plt.plot(range(len(exp2_brazil_best_fitness_tournament_selection_history)),
         exp2_brazil_best_fitness_tournament_selection_history, label='Experient 2', color='blue')
plt.plot(range(len(exp3_brazil_best_fitness_tournament_selection_history)),
         exp3_brazil_best_fitness_tournament_selection_history, label='Experient 3', color='green')
plt.plot(range(len(exp4_brazil_best_fitness_tournament_selection_history)),
         exp4_brazil_best_fitness_tournament_selection_history, label='Experient 4', color='yellow')
plt.title('Best Fitness Over Generations (Brazil DataSet)')
plt.xlabel('Generations')
plt.ylabel('Fitness')
plt.legend()
plt.show()
