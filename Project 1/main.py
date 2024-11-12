import re
import math
import random
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional

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
    coords = {city_id: (data['x_location'], data['y_location']) 
             for city_id, data in cities.items()}
    
    distance_matrix = {}
    city_ids = list(cities.keys())
    
    for i in range(len(city_ids)):
        x1, y1 = coords[city_ids[i]]
        for j in range(i + 1, len(city_ids)):
            x2, y2 = coords[city_ids[j]]
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distance_matrix[(city_ids[i], city_ids[j])] = dist
            distance_matrix[(city_ids[j], city_ids[i])] = dist
    
    return distance_matrix

def calculate_route_distance(route: List[int], cities: Dict, distance_matrix: Dict[Tuple[int, int], float]) -> float:
    get_distance = distance_matrix.get
    return sum(get_distance((route[i], route[(i + 1) % len(route)])) 
              for i in range(len(route)))

def tournament_selection(population: List[List[int]], cities: Dict, distance_matrix: Dict[Tuple[int, int], float], tournament_size: int = 5) -> List[int]:
    tournament = random.sample(population, tournament_size)
    return min(tournament, 
              key=lambda x: calculate_route_distance(x, cities, distance_matrix))

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

def inversion_mutation(solution: List[int], mutation_rate: float) -> List[int]:
    if random.random() < mutation_rate:
        i, j = sorted(random.sample(range(len(solution)), 2))
        solution[i:j+1] = reversed(solution[i:j+1])
    return solution

def create_new_population(pop_size: int, cities: Dict, distance_matrix: Dict[Tuple[int, int], float], 
                         tournament_size: int, mutation_rate: float) -> List[List[int]]:
    population = [generate_random_solution(cities) for _ in range(pop_size)]
    
    new_population = []
    while len(new_population) < pop_size:
        p1 = tournament_selection(population, cities, distance_matrix, tournament_size)
        p2 = tournament_selection(population, cities, distance_matrix, tournament_size)
        
        offspring = order_crossover(p1, p2)
        
        if random.random() < 0.5:
            offspring = swap_mutation(offspring, mutation_rate)
        else:
            offspring = inversion_mutation(offspring, mutation_rate)
            
        new_population.append(offspring)
    
    return new_population

def genetic_algorithm(cities: Dict, population_size: int = 200, generations: int = 2000,
                     elite_size: int = 20, tournament_size: int = 5) -> Tuple[List[int], float, List[float], float]:
    start_time = time.time()
    
    distance_matrix = calculate_distance_matrix(cities)
    mutation_rate = 1.0 / len(cities)
    
    population = create_new_population(population_size, cities, distance_matrix, tournament_size, 
                                     mutation_rate)
    best_distances_history = []
    best_distances_history.extend([0] * generations)
    
    route_distances = {}
    route_key = tuple
    
    best_distance = float('inf')
    best_route = None
    
    for gen in range(generations):
        new_routes = set(route_key(route) for route in population 
                        if route_key(route) not in route_distances)
        for route_tuple in new_routes:
            route_distances[route_tuple] = calculate_route_distance(list(route_tuple), 
                                                                 cities, distance_matrix)

        population.sort(key=lambda x: route_distances[route_key(x)])
        
        current_best_distance = route_distances[route_key(population[0])]
        best_distances_history[gen] = current_best_distance
        
        if current_best_distance < best_distance:
            best_distance = current_best_distance
            best_route = population[0].copy()
        
        if gen % 100 == 0:
            print(f"Generation {gen}: Distance = {best_distance:.2f}")
        
        new_population = [None] * population_size
        
        for i in range(elite_size):
            new_population[i] = population[i].copy()
        
        for i in range(elite_size, population_size):
            parent1 = tournament_selection(population, cities, distance_matrix, tournament_size)
            parent2 = tournament_selection(population, cities, distance_matrix, tournament_size)
            child = order_crossover(parent1, parent2)
            
            if random.random() < 0.5:
                child = swap_mutation(child, mutation_rate)
            else:
                child = inversion_mutation(child, mutation_rate)
                
            new_population[i] = child
        
        population = new_population
        
        if len(route_distances) > population_size * 2:
            current_routes = set(route_key(route) for route in population)
            route_distances = {k: v for k, v in route_distances.items() 
                             if k in current_routes}

    execution_time = time.time() - start_time
    return best_route, best_distance, best_distances_history, execution_time

def plot_performance(distances_history: List[float], title: str, figure_num: int, execution_time: float):
    fig = plt.figure(figure_num, figsize=(10, 6))
    plt.clf()
    
    color = '#0077BB'
    plt.plot(distances_history, color=color, linewidth=2, 
            label=f'Execution time: {execution_time:.2f}s')
    
    final_value = distances_history[-1]
    plt.plot(len(distances_history)-1, final_value, 'o', color=color)
    plt.annotate(f'Final: {final_value:.2f}', 
                xy=(len(distances_history)-1, final_value),
                xytext=(10, 10), textcoords='offset points')
    
    plt.title(f'Best Distance Over Generations - {title}')
    plt.xlabel('Generation')
    plt.ylabel('Distance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.5)
    return fig

def compare_parameters(cities: Dict, parameter_sets: List[Dict], 
                      generations: int = 2000, figure_num: int = None) -> None:
    colors = ['#0077BB', '#EE7733', '#009988',
             '#CC3311', '#33BBEE', '#EE3377']
    
    fig = plt.figure(figure_num, figsize=(12, 6))
    plt.clf()
    
    all_histories = []
    all_labels = []
    
    for idx, params in enumerate(parameter_sets):
        best_route, best_distance, history, exec_time = genetic_algorithm(
            cities=cities,
            generations=generations,
            **params
        )
        
        final_value = history[-1]
        label = f"Pop:{params['population_size']}, Elite:{params['elite_size']}, Tour:{params['tournament_size']} (Final: {final_value:.2f}, Time: {exec_time:.2f}s)"
        
        all_histories.append((history, final_value))
        all_labels.append(label)
        
        plt.clf()
        for j in range(len(all_histories)):
            h, fv = all_histories[j]
            plt.plot(h, color=colors[j % len(colors)], 
                    label=all_labels[j], linewidth=2)
            plt.plot(len(h)-1, fv, 'o', 
                    color=colors[j % len(colors)])
        
        plt.title('Parameter Comparison')
        plt.xlabel('Generation')
        plt.ylabel('Best Distance')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)
    
    return fig

def main():
    plt.ion()
    plt.close('all')
    
    files = ['Project 1/berlin11_modified.tsp', 'Project 1/berlin52.tsp', 'Project 1/kroA100.tsp']
    
    parameter_sets = [
        {'population_size': 200, 'elite_size': 20, 'tournament_size': 2},
        {'population_size': 200, 'elite_size': 20, 'tournament_size': 3},
        {'population_size': 200, 'elite_size': 20, 'tournament_size': 4},
    ]
    
    figures = {}
    
    for file_idx, file in enumerate(files):
        print(f"\nProcessing file: {file}")
        print("-" * 50)
        
        cities, metadata = parse_tsp_file(file)
        print(f"Number of cities: {len(cities)}")
        
        best_route, best_distance, history, exec_time = genetic_algorithm(
            cities=cities,
            population_size=200,
            generations=2000,
            elite_size=20,
            tournament_size=3 
        )
        
        print(f"\nBest solution found:")
        print(f"Total distance: {best_distance:.2f}")
        print(f"Route: {best_route}")
        print(f"Execution time: {exec_time:.2f} seconds")
        
        perf_fig_num = file_idx * 2 + 1
        figures[f'perf_{file_idx}'] = plot_performance(history, f"File: {file}", 
                                                      figure_num=perf_fig_num, 
                                                      execution_time=exec_time)
        
        print("\nComparing different parameter sets...")
        comp_fig_num = file_idx * 2 + 2
        figures[f'comp_{file_idx}'] = compare_parameters(cities, parameter_sets, 
                                                       figure_num=comp_fig_num)
        
        print("\n" + "="*50)
    
    for fig_name, fig in figures.items():
        plt.figure(fig.number)
        plt.tight_layout()
        plt.draw()
    
    plt.show(block=True)

if __name__ == "__main__":
    main()
