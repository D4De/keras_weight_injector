import os
import importlib.util
import inspect

def esegui(dataset, loader):
    """
    Importa ed esegue l'unica funzione definita nell'unico file presente nella cartella loaders/stringa
    
    Args:
        stringa (str): Nome della cartella da cercare in loaders/
        
    Returns:
        function: La funzione trovata nel file
        
    Raises:
        FileNotFoundError: Se la cartella non esiste o è vuota
        ValueError: Se ci sono più file o più funzioni
    """
    # Costruisci il percorso della cartella
    base_path = os.path.dirname(os.path.abspath(__file__))
    cartella_path = os.path.join(base_path, "loaders", dataset)
    
    # Verifica che la cartella esista
    if not os.path.exists(cartella_path) or not os.path.isdir(cartella_path):
        raise FileNotFoundError(f"La cartella {cartella_path} non esiste")
    
    # Elenca i file nella cartella (escludi i file nascosti e __pycache__)
    files = [f for f in os.listdir(cartella_path) 
             if os.path.isfile(os.path.join(cartella_path, f)) 
             and not f.startswith('.') 
             and not f.startswith('__')]
    
    # Verifica che ci sia esattamente un file con estensione .py
    python_files = [f for f in files if f.endswith('.py')]

    if len(python_files) == 0:
        raise FileNotFoundError(f"Nessun file Python (.py) trovato nella cartella {cartella_path}")
    elif len(python_files) > 1:
        raise ValueError(f"Trovati più file Python nella cartella {cartella_path}: {python_files}. Deve contenere un solo file Python.")

    file_name = python_files[0]
    file_path = os.path.join(cartella_path, file_name)
    
    
    # Importa il modulo dinamicamente
    module_name = f"loaders.{dataset}.{os.path.splitext(file_name)[0]}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Trova tutte le funzioni definite nel modulo
    funzioni = [f for name, f in inspect.getmembers(module, inspect.isfunction) 
                if not name.startswith('_')]
    
    for f in funzioni:
        # Verifica che la funzione abbia il nome specificato
        if f.__name__ == loader:
            return f
    print("Funzione non trovata")
    return None


l = esegui("GTSRB", "loader")
l()

