import re

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

def test_parser():
    files = ['kroA100.tsp', 'berlin52.tsp', 'berlin11_modified.tsp']
    
    for file in files:
        print(f"Testing file: {file}")
        cities, metadata = parse_tsp_file(file)
        
        print("Metadata:")
        for key, value in metadata.items():
            print(f"{key}: {value}")
        
        print("\nFirst 5 cities:")
        for i, (city_id, city_data) in enumerate(cities.items()):
            if i >= 5:
                break
            print(f"City {city_id}: {city_data}")
        
        print(f"\nTotal number of cities: {len(cities)}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    test_parser()
