import argparse

import tensorflow as tf  # type:ignore
import ast
import os
import shutil
import json

from tf_injector.utils import SUPPORTED_MODELS, SUPPORTED_DATASETS, DEFAULT_REPORT_DIR, IMAGE_CLASSIFICATION_REPORT_HEADER
from tf_injector.loader import load_network
from tf_injector.injector import Injector
from tf_injector.metrics import ImageClassificationMetric
from tf_injector.writer import CampaignWriter



def parse_args():
    parser = argparse.ArgumentParser(
        description="Perform fault injections to weigths of a Tensorflow model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Different commands")

    # RUN_OLD (old implementation)
    run_default_parser = subparsers.add_parser('run_old', help='Run the old implementation')
    run_default_parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        choices=SUPPORTED_DATASETS,
        required=True,
        help="Dataset to use",
    )
    run_default_parser.add_argument(
        "--batch-size", "-b", type=int, default=512, help="Test set batch size"
    )
    run_default_parser.add_argument(
        "--network-name",
        "-n",
        type=str,
        required=True,
        help="Target network",
        choices=SUPPORTED_MODELS,
    )
    run_default_parser.add_argument(
        "--fault-list",
        "-f",
        type=str,
        required=False,
        help="Path to Fault list",
    )
    run_default_parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        required=False,
        default=DEFAULT_REPORT_DIR,
        help="Path to the generated output",
    )
    run_default_parser.add_argument(
        "--resume-from",
        "-r",
        type=int,
        required=False,
        default=0,
        help="Resume experiment from a certain injection id",
    )
    run_default_parser.add_argument(
        "--save-scores",
        "-s",
        action="store_true",
        help="Save Injection Data",
    )

    run_default_parser.add_argument(
        "--use-tf",
        "-u",
        action="store_true",
        help="Use TensorFlow instead of NumPy for dataset preprocessing"
    )

    run_default_parser.add_argument(
        "--seed", default=None, type=int, help="random seed for determinism"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the fault list before running the campaign",
    )

    # Comandi di LOAD : ld + {cosa da loadare}
    load_parser = subparsers.add_parser('lddataset', help='load dataset')
    load_parser.add_argument(
        '--path_to_function', 
        '-fp', 
        required=True, 
        help='path of the loading python file'
        )
    load_parser.add_argument(
        '--function_name', 
        '-fn', 
        required=True, 
        help='name of the top function of the loading file'
        )
    load_parser.add_argument(
        '--dataset_name', 
        '-n', 
        required=True, 
        help='dataset name, if present you substitute the python file with this one'
        )

    load_parser = subparsers.add_parser('ldmetric', help='load metric')
    load_parser.add_argument(
        '--path_to_class', 
        '-p', 
        required=True, 
        help='path of the loading python file with the class'
        )

    # Comando RUN (to confing a custom campaign)
    load_parser = subparsers.add_parser('run', help='Run campaign')
    load_parser.add_argument(
        '--dataset', 
        '-d', 
        required=True, 
        help='name of the dataset'
        ) 
    load_parser.add_argument(
        '--preprocessing', 
        '-p', 
        help='Preprocessing function'
        )
    load_parser.add_argument(
        '--metrics', 
        '-met', 
        required=True, 
        help='Metriche da utilizzare'
        )
    load_parser.add_argument(
        '--fault_list', 
        '-fl', 
        help='path to the fault list'
        )
    load_parser.add_argument(
        '--model', 
        '-m', 
        required=True, 
        help='path to the model'
        )

    # Comando SHOW
    show_parser = subparsers.add_parser('show', help='Show all the catalogue')
    show_parser.add_argument(
        '--all', 
        '-all', 
        help='Show all the catalogue'
        )
    show_parser.add_argument(
        '--dataset', 
        '-d', 
        help='Show the loaded dataset with associated functions'
        )

    # Comando RESET (delete all the user plugins)
    reset_parser = subparsers.add_parser('reset', help='delete anything the user plugged in')
    return parser.parse_args()


def run_old(args):
    if args.seed:
        tf.config.experimental.enable_op_determinism()
        tf.keras.utils.set_random_seed(args.seed)
        tf.keras.backend.manual_variable_initialization(True)

    network, dataset = load_network(args.network_name, args.dataset, use_tf=args.use_tf)
    injector = Injector(network, dataset)
    
    # add metric choice logic here
    metric = ImageClassificationMetric

    # add report header choice logic here
    report_header = IMAGE_CLASSIFICATION_REPORT_HEADER

    if args.fault_list is None:
        output, labels = injector.run_inference(args.batch_size)
        top_1, top_5 = metric(output, labels, labels).clean_metric()
        print(
            f"GOLD stats:\nimages: {len(dataset)}\ntop 1 accuracy: {top_1}\ntop 5 accuracy: {top_5}"
        )
        if args.save_scores:
            cw = CampaignWriter(args.dataset, args.network_name, args.output_path)
            cw.save_scores(output)
    else:
        injector.load_fault_list(args.fault_list, resume_from=args.resume_from)
        if args.validate:
            print("Running validation")
            injector.validate()

        with CampaignWriter(args.dataset, args.network_name, report_header, args.output_path) as cw:
            injector.run_campaign(
                batch=args.batch_size,
                metrics=[metric],
                outputter=cw,
                save_scores=args.save_scores,
            )

def run(args):
    if args.metrics:
        metrics_list = args.metrics.split(',')
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
    return

def show(args):
    return

def main(args):
    if args.command == "run_old":
        run_old(args)
    
    elif args.command == "run":
        run(args)
    
    elif args.command == "lddataset":
        lddataset(args)
    
    elif args.command == "ldmetric":
        ldmetric(args)
    
    elif args.command == "show":
        show(args)

    else:
        raise ValueError(f"Unknown command: {args.command}")
    
    
if __name__ == "__main__":
    main(parse_args())
