import json
import os
from tqdm import tqdm
import numpy as np
from typing import Dict, Any
from accelerate import Accelerator
from maskgen.utils.save_utils import save_maskgen_results 


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Flatten config for easier access
    flat_config = {}
    flat_config.update(config['evaluation'])
    flat_config.update(config['model'])
    flat_config.update(config['dataset'])
    
    return flat_config

def main():
    accelerator = Accelerator()
    device = accelerator.device

    config = load_config('eval_config.json')
    save_maskgen_results(config, device)


if __name__ == '__main__':
    main()
