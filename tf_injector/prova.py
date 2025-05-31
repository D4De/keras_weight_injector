import json
import os

base_path = os.path.dirname(os.path.abspath(__file__))
target_dir = os.path.join(base_path, "loaders", "CIFAR10")

config_file_path = os.path.join(target_dir, "config.json")
with open(config_file_path, 'r') as file:
    config_data = json.load(file)
func_name = config_data["loader_name"]

