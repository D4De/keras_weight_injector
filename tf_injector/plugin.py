import os
import shutil
import argparse
import importlib
import sys

# Percorso della cartella user
USER_PLUGINS_DIR = os.path.join(os.path.dirname(__file__), 'user')

def ensure_user_directory():
    """Crea la directory 'user' se non esiste"""
    if not os.path.exists(USER_PLUGINS_DIR):
        os.makedirs(USER_PLUGINS_DIR)
        # Crea un file __init__.py per rendere la cartella un package Python
        with open(os.path.join(USER_PLUGINS_DIR, '__init__.py'), 'w') as f:
            f.write('# Package per plugin utente\n')

def add_user_function(file_path, function_name=None):
    """
    Aggiunge un file Python contenente una funzione utente alla cartella 'user'
    
    Args:
        file_path: Percorso del file Python da aggiungere
        function_name: Nome della funzione da verificare (opzionale)
        
    Returns:
        Il nome del modulo aggiunto
    """
    ensure_user_directory()
    
    # Controlla che il file esista
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Il file {file_path} non esiste")
        
    # Controlla che sia un file Python
    if not file_path.endswith('.py'):
        raise ValueError("Il file deve essere un file Python (.py)")
    
    # Ottieni il nome del file
    file_name = os.path.basename(file_path)
    module_name = file_name.replace('.py', '')
    
    # Copia il file nella cartella user
    destination = os.path.join(USER_PLUGINS_DIR, file_name)
    shutil.copy2(file_path, destination)
    
    # Verifica che la funzione esista nel file, se specificata
    if function_name:
        try:
            # Aggiungi la cartella user al path per poter importare il modulo
            sys.path.insert(0, os.path.dirname(USER_PLUGINS_DIR))
            
            # Importa il modulo
            user_module = importlib.import_module(f"user.{module_name}")
            
            # Verifica che la funzione esista
            if not hasattr(user_module, function_name):
                os.remove(destination)  # Rimuovi il file se la funzione non esiste
                raise AttributeError(f"La funzione {function_name} non esiste nel file {file_name}")
                
            sys.path.pop(0)  # Rimuovi il percorso aggiunto
            
        except Exception as e:
            os.remove(destination)  # Rimuovi il file in caso di errore
            raise e
    
    print(f"File {file_name} aggiunto con successo!")
    print(f"Puoi usare questo loader personalizzato nel tuo file YAML con:")
    print(f"""
campaign:
  custom_function:
    module: "user.{module_name}"
    name: "{function_name or 'nome_funzione'}"
    params:
      network_name: "ResNet20"
      dataset_name: "CIFAR10"
      # altri parametri opzionali
  metrics:
    - ImageClassificationMetric
  outputter: "CampaignWriter"
    """)
    
    return module_name

