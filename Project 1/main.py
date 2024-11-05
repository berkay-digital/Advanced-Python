import re
import math
import random
from typing import Dict, List, Tuple

#random.seed(42)

def parse_tsp_file(file_path: str) -> Tuple[Dict[int, Dict], Dict]:
    with open(file_path, 'r') as file:
        content = file.read()

    metadata = {}
    node_coord_section = re.search(r'NODE_COORD_SECTION\n([\s\S]*?)EOF', content)
    
    if node_coord_section:
        node_coords = node_coord_section.group(1).strip().split('\n')
        cities = {}
        for line in node_coords:
            ordering, x, y = map(float, line.split())
            cities[int(ordering)] = {'x_location': x, 'y_location': y}
    else:
        raise ValueError("NODE_COORD_SECTION not found in the file")

    metadata_section = content.split('NODE_COORD_SECTION')[0]
    for line in metadata_section.split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()

    return cities, metadata

def calculate_distance(city1: Dict, city2: Dict) -> float:
    x1, y1 = city1['x_location'], city1['y_location']
    x2, y2 = city2['x_location'], city2['y_location']
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def generate_random_solution(cities: Dict) -> List[int]:
    city_ids = list(cities.keys())
    random.shuffle(city_ids)
    return city_ids

def calculate_distance_matrix(cities: Dict) -> Dict[Tuple[int, int], float]:
    distance_matrix = {}
    for i in cities:
        for j in cities:
            if i != j and (j, i) not in distance_matrix:
                distance_matrix[(i, j)] = calculate_distance(cities[i], cities[j])
                distance_matrix[(j, i)] = distance_matrix[(i, j)]
    return distance_matrix

def calculate_route_distance(route: List[int], cities: Dict, distance_matrix: Dict[Tuple[int, int], float]) -> float:
    total_distance = 0
    for i in range(len(route)):
        city1, city2 = route[i], route[(i + 1) % len(route)]
        total_distance += distance_matrix[(city1, city2)]
    return total_distance

def tournament_selection(population: List[List[int]], cities: Dict, distance_matrix: Dict[Tuple[int, int], float], tournament_size: int = 5) -> List[int]:
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda x: calculate_route_distance(x, cities, distance_matrix))

def order_crossover(parent1: List[int], parent2: List[int]) -> List[int]:
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [-1] * size
    child[start:end + 1] = parent1[start:end + 1]
    
    remaining = [x for x in parent2 if x not in child]
    j = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = remaining[j]
            j += 1
    
    return child

def swap_mutation(solution: List[int], mutation_rate: float) -> List[int]:
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(solution)), 2)
        solution[i], solution[j] = solution[j], solution[i]
    return solution

def genetic_algorithm(cities: Dict, population_size: int = 200, generations: int = 2000,
                     elite_size: int = 20, tournament_size: int = 5) -> Tuple[List[int], float]:

    distance_matrix = calculate_distance_matrix(cities)
    
    population = [generate_random_solution(cities) for _ in range(population_size)]
    mutation_rate = 1.0 / len(cities)
    
    best_distance = float('inf')
    best_route = None
    

    route_distances = {}
    
    for gen in range(generations):

        for route in population:
            route_tuple = tuple(route)
            if route_tuple not in route_distances:
                route_distances[route_tuple] = calculate_route_distance(route, cities, distance_matrix)
        

        population.sort(key=lambda x: route_distances[tuple(x)])
        
        current_best_distance = route_distances[tuple(population[0])]
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = population[0].copy()
        
        if gen % 100 == 0:
            print(f"Generation {gen}: Distance = {best_distance:.2f}")
        
        new_population = []
        new_population.extend([x.copy() for x in population[:elite_size]])
        
        while len(new_population) < population_size:
            parent1 = tournament_selection(population, cities, distance_matrix, tournament_size)
            parent2 = tournament_selection(population, cities, distance_matrix, tournament_size)
            child = order_crossover(parent1, parent2)
            child = swap_mutation(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        
        if len(route_distances) > population_size * 2:
            new_cache = {}
            for route in population:
                route_tuple = tuple(route)
                new_cache[route_tuple] = calculate_route_distance(route, cities, distance_matrix)
            route_distances = new_cache

    return best_route, best_distance

def main():
    files = ['Project 1/kroA100.tsp', 'Project 1/berlin52.tsp', 'Project 1/berlin11_modified.tsp']
    
    for file in files:
        print(f"\nProcessing file: {file}")
        print("-" * 50)
        
        cities, metadata = parse_tsp_file(file)
        print(f"Number of cities: {len(cities)}")
        
        best_route, best_distance = genetic_algorithm(
            cities=cities,
            population_size=200,
            generations=2000,
            elite_size=20,
            tournament_size=4
        )
        
        print(f"\nBest solution found:")
        print(f"Total distance: {best_distance:.2f}")
        print(f"Route: {best_route}")
        print("\n" + "="*50)

if __name__ == "__main__":
    main()
