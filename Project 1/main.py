import re
import math
import random

def parse_tsp_file(file_path):
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

def calculate_distance(city1, city2):
    x1, y1 = city1['x_location'], city1['y_location']
    x2, y2 = city2['x_location'], city2['y_location']
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def generate_random_solution(cities):
    city_ids = list(cities.keys())
    random.shuffle(city_ids)
    return city_ids

def test_all_functions():
    files = ['Project 1/kroA100.tsp', 'Project 1/berlin52.tsp', 'Project 1/berlin11_modified.tsp']
    
    for file in files:
        print(f"\nTesting file: {file}")
        print("=" * 50)
        
        cities, metadata = parse_tsp_file(file)
        print("Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        print(f"\nTotal number of cities: {len(cities)}")
        print("\nFirst 5 cities:")
        for i, (city_id, city_data) in enumerate(cities.items()):
            if i >= 5:
                break
            print(f"City {city_id}: {city_data}")
        
        city_ids = list(cities.keys())
        city1_id, city2_id = city_ids[0], city_ids[1]
        city1, city2 = cities[city1_id], cities[city2_id]
        distance = calculate_distance(city1, city2)
        print(f"\nDistance between city {city1_id} and city {city2_id}: {distance:.2f}")
        
        random_solution = generate_random_solution(cities)
        print(f"\nRandom solution (first 10 cities): {random_solution[:10]}")

        for i in range(len(random_solution) - 1):
            distance = calculate_distance(cities[random_solution[i]], cities[random_solution[i+1]])
            print(f"Distance between city {random_solution[i]} and city {random_solution[i+1]}: {distance:.2f}")

        print("\n" + "="*50)

if __name__ == "__main__":
    test_all_functions()
