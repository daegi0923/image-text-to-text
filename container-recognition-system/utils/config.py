import yaml
import os

def load_config(config_path="configs/settings.yaml"):
    if not os.path.exists(config_path):
        # Fallback or default
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
