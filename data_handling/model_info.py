import numpy as np
import json
import logging
from typing import Dict, Any

logger = logging.getLogger('ModelInfo')

def load_model_info(dataset_path: str, oiu: int) -> Dict[str, Any]:
    """
    Load model information from a JSON file.

    Parameters:
    - dataset_path (str): The path to the dataset.
    - oiu (int): The object identifier.

    Returns:
    - model_info (dict): A dictionary containing the loaded model information.

    Raises:
    - AssertionError: If the object identifier is not found in the JSON data.

    """
    model_info: Dict[str, Any] = {}
    with open(f'{dataset_path}/models_eval/models_info.json') as f:
        jsondata = json.load(f)

        key = str(oiu)
        assert(key in jsondata)

        model_info["diameter"] = jsondata[key]["diameter"]
        model_info["mins"] = np.array([jsondata[key]["min_x"],jsondata[key]["min_y"],jsondata[key]["min_z"]])
        model_info["maxs"] = np.array([jsondata[key]["size_x"],jsondata[key]["size_y"],jsondata[key]["size_z"]]) +  model_info["mins"]

        model_info["symmetries_discrete"] = [np.array(_).reshape((4,4)) for _ in jsondata[key]["symmetries_discrete"]] if "symmetries_discrete" in jsondata[key] else []
        model_info["symmetries_continuous"] = "symmetries_continuous" in jsondata[key]

    logger.debug(f'Model info for object {oiu}:')
    for k in model_info:
        logger.debug(f'\t{k} : {model_info[k]}')
            
    return model_info
