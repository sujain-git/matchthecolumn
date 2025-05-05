import importlib.util
import inspect
import sys
import os
from typing import get_type_hints

def extract_function_info(func):
    info = {
        "function_name": func.__name__,
        "inputs": [],
        "outputs": []
    }

    # Get function signature and annotations
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    for param in sig.parameters.values():
        annotation = hints.get(param.name, 'Any')
        info["inputs"].append(f"{param.name}: {annotation.__name__ if hasattr(annotation, '__name__') else str(annotation)}")

    # Extract output info (assumes function has custom attributes set by decorators)
    if hasattr(func, "_output_definitions"):
        for output in func._output_definitions:
            name = output["name"]
            dtype = output["type"]
            info["outputs"].append(f"{name}: {dtype}")
    else:
        # Fallback if not decorated
        return_type = hints.get("return", None)
        if return_type:
            info["outputs"].append(str(return_type))

    return info

def load_module_from_file(filepath):
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def extract_all_functions(filepath):
    module = load_module_from_file(filepath)
    results = []

    for name, obj in inspect.getmembers(module, inspect.isfunction):
        if obj.__module__ == module.__name__:
            results.append(extract_function_info(obj))

    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python extract_from_file.py <python_file.py>")
        sys.exit(1)

    filepath = sys.argv[1]
    extracted = extract_all_functions(filepath)
    for item in extracted:
        print(item)
