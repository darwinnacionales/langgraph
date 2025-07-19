from langchain_core.tools import tool
import random
from typing import Dict, Any, List

@tool
def gather_data_tool(request: str):
    """
    Gather data based on the user's request.
    
    Args:
        request (str): The user's request for data.
    
    Returns:
        The response will be a JSON data structure containing the gathered data.
        Example:
        {
            "data": {
                "sales": 1000,
                "customers": 200,
                "items": [
                    {"id": 1, "name": "Item 1", "value": 10},
                    {"id": 2, "name": "Item 2", "value": 20},
                    {"id": 3, "name": "Item 3", "value": 30}
                ]
            },
            "status": "success"
        }
    """
    # Placeholder for data gathering logic
    return get_db_data()

def get_db_data() -> Dict[str, Any]:
    len = random.randint(3, 10) 

    data = []
    for i in range(len):
        data.append({
            "id": i,
            "name": f"Item {i}",
            "value": random.randint(1, 100)
        })

    return {
        "sales": random.randint(100, 1000),
        "customers": random.randint(50, 500),
        "items": data
    }

@tool
def min_tool(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate the minimum value from a list of dictionaries.
    
    Args:
        data (List[Dict[str, Any]]): A list of dictionaries containing numerical values.
    
    Returns:
        The minimum value found in the list of dictionaries.
    """
    if not data:
        return {"min_value": None}
    
    min_value = min(item.get("value", float('inf')) for item in data)
    return {"min_value": min_value}

@tool
def max_tool(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate the maximum value from a list of dictionaries.
    
    Args:
        data (List[Dict[str, Any]]): A list of dictionaries containing numerical values.
    
    Returns:
        The maximum value found in the list of dictionaries.
    """
    if not data:
        return {"max_value": None}
    
    max_value = max(item.get("value", float('-inf')) for item in data)
    return {"max_value": max_value}

@tool
def average_tool(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate the average value from a list of dictionaries.
    
    Args:
        data (List[Dict[str, Any]]): A list of dictionaries containing numerical values.
    
    Returns:
        The average value found in the list of dictionaries.
    """
    if not data:
        return {"average_value": None}
    
    total = sum(item.get("value", 0) for item in data)
    count = len(data)
    average_value = total / count if count > 0 else 0
    return {"average_value": average_value}

@tool
def sum_tool(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate the sum of values from a list of dictionaries.
    
    Args:
        data (List[Dict[str, Any]]): A list of dictionaries containing numerical values.
    
    Returns:
        The sum of all values found in the list of dictionaries.
    """
    if not data:
        return {"sum_value": 0}
    
    total = sum(item.get("value", 0) for item in data)
    return {"sum_value": total}
