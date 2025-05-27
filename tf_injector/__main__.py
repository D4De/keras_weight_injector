import argparse

import tensorflow as tf  # type:ignore
import ast
import os
import shutil
import json

from tf_injector.campaign import Campaign

def get_avaliable_datasets() -> str:
    """
    Returns a string with the names of the avaliable datasets
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    loaders_path = os.path.join(base_path, "loaders")
    datasets = [d for d in os.listdir(loaders_path) if os.path.isdir(os.path.join(loaders_path, d))]
    return ", ".join(datasets)

def get_avaliable_metrics() -> str:
    '''
    Returns a string with the names of the avaliable metrics
    '''
    base_path = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_path, "new_metrics")
    metrics = [f[:-3] for f in os.listdir(metrics_path) if f.endswith('.py') and f != '__init__.py' and f != 'metric.py']
    return ", ".join(metrics)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform fault injections to weigths of a Tensorflow model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Different commands")

    # Load Commands : ld + <dataset/metric>
    load_parser = subparsers.add_parser('lddataset', help='load dataset')
    load_parser.add_argument(
        '--path_to_function', 
        '-fp', 
        required=True, 
        help='Path of the python file that contains the function to load the dataset'
        )
    load_parser.add_argument(
        '--function_name', 
        '-fn', 
        required=True, 
        help='Name of the top function of the loading file'
        )
    load_parser.add_argument(
        '--dataset_name', 
        '-n', 
        required=True, 
        help='Dataset name to load, it will be used to create a directory in the loaders folder'
        )
    
    # Comando LOAD METRIC : ldmetric
    load_parser = subparsers.add_parser('ldmetric', help='load metric')
    load_parser.add_argument(
        '--path', 
        '-p', 
        required=True, 
        help='Path of the loading python file with the class. Name of the class must be the same as the file name'
        )

    # Comando RUN (to confing a custom campaign)
    run_parser = subparsers.add_parser('run', help='Run campaign')
    run_parser.add_argument(
        '--dataset', 
        '-d', 
        required=True,
        help=f"Name of the dataset. Avaliables : {get_avaliable_datasets()}\n"
        ) 
    run_parser.add_argument(
        '--dataset_path', 
        '-dp', 
        help='Path to the dataset directory',
        default="lambda x: x"
    )
    run_parser.add_argument(
        '--postprocess', 
        '-p', 
        help='Lambda expression for the function to apply to the output of the inference of the model\nEXAMPLE: "lambda x: x" (default)',
        default="lambda x: x"
    )
    run_parser.add_argument(
        '--metrics', 
        '-met', 
        required=True, 
        help=f"Set of metrics to use for the campaign, separated by commas.\nEXAMPLE: metric1,mectric2,metric3.\nAvaliable metrics: {get_avaliable_metrics()}\n"
        )
    run_parser.add_argument(
        '--fault_list', 
        '-fl', 
        help='Path to the fault list'
        )
    run_parser.add_argument(
        '--model', 
        '-m', 
        required=True, 
        help='Path to the model model.keras, model.h or the model directory'
        )
    run_parser.add_argument(
        '--output_path', 
        '-o', 
        default='out',
        help='Path to the output directory where the results will be saved',
        )
    run_parser.add_argument(
        '--batch', 
        '-b', 
        type=int, 
        default=32, 
        help='Batch size for the campaign'
    )
    run_parser.add_argument(
        '--save_scores', 
        '-s',
        help='Save scores',
    )
    run_parser.add_argument(
        '--resume-from', 
        '-r', 
        type=int, 
        default=0, 
        help='Resume injection from this index'
    )
    run_parser.add_argument(
        '--validate', 
        '-v', 
        help='Validate the fault list, checking if the layers are present in the model'
    )
    run_parser.add_argument(
        '--seed', 
        '-se', 
        type=int, 
        default=None, 
        help='Set a seed for determinism'
    )
    
    return parser.parse_args()

def run(args):
    if args.metrics:
        metrics_list = args.metrics.replace(" ", "").split(",")

    save_scores = True if args.save_scores else False
    resume_from = args.resume_from if args.resume_from else 0
    validate_fault_list = True if args.validate else False

    campaign = Campaign(
        dataset_name = args.dataset,
        dataset_path = args.dataset_path,
        network_path = args.model,
        output_path = args.output_path,
        fautl_list_path = args.fault_list,
        preporcessig = args.postprocess,   # (tf.data.Dataset) -> tf.data.Dataset
        metrics_list = metrics_list,
        batch_size = args.batch,
        save_scores = save_scores,
        resume_from = resume_from,
        validate_fault_list = validate_fault_list,
        seed = args.seed
    )
    campaign.run()
    return

def lddataset(args):
    name = args.dataset_name 
    path = args.path_to_function
    function_name = args.function_name

    # check path leads to a python file
    if (not path.endswith(".py")) :
        raise ValueError(f"The path doesn't lead to a python (.py) file : {path}")

    # check if the function is in the file (Abstract Syntax Tree)
    with open(path, 'r') as file:
            source = file.read()
        
    tree = ast.parse(source)
    functions = {node.name: node for node in ast.walk(tree) 
                    if isinstance(node, ast.FunctionDef) and not node.name.startswith('_')}
        
    if (function_name not in functions.keys()):
        raise ValueError(f"The function {function_name} is not in the file {path}")
    
    # create name dir if non present
    base_path = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(base_path, "loaders", name)
    if os.path.exists(target_dir):
        print(f"Dataset {name} was already loaded\nUpdating loading function...")
    else:
        os.mkdir(target_dir)
        print(f"Loading new dataset {name}...")

    # copy the file in the name dir
    # (delete .py file in target_dir)
    for file in os.listdir(target_dir):
        if file.endswith('.py'):
            file_path = os.path.join(target_dir, file)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")

    destination_file = os.path.join(target_dir, os.path.basename(path))
    shutil.copy2(path, destination_file)
    print(f"Added new loading fuunction file: {path} → {destination_file}")

    # create/modify data.jsonv putting loader name of the function
    for file in os.listdir(target_dir):
        if file.endswith('.json'):
            file_path = os.path.join(target_dir, file)
            os.remove(file_path)
            #print(f"Deleted file: {file_path}")
    
    json_file_path = os.path.join(target_dir, f"config.json")
    loader_data = {"loader_name": function_name}

    with open(json_file_path, 'w') as json_file:
        json.dump(loader_data, json_file, indent=4)

    print(f"loading function for dataset {name}: {function_name}")

    return

def ldmetric(args):
    
    path = args.path
    metric_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "new_metrics")

    # check path leads to a python file
    if (not path.endswith(".py")) :
        raise ValueError(f"The path doesn't lead to a python (.py) file : {path}")

    # check file validity
    if (not os.path.exists(path) or not os.path.isfile(path)):
        raise FileNotFoundError(f"The file {path} doesn't exist or is not a file")

    # copy the file path in target_dir
    shutil.copy2(path, metric_dir)
    print(f"Added new metric file: {path} → {metric_dir}")

    return

def main(args):
    
    if args.command == "run":
        run(args)
    
    elif args.command == "lddataset":
        lddataset(args)
    
    elif args.command == "ldmetric":
        ldmetric(args)

    else:
        raise ValueError(f"Unknown command: {args.command}")
    
    
if __name__ == "__main__":
    main(parse_args())
