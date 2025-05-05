import ast

def get_function_signatures(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    function_signatures = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [f"{arg.arg}: {ast.unparse(arg.annotation)}" for arg in node.args.args]
            return_type = get_return_type(node)
            signature = {
                "function_name": node.name,
                "inputs": args,
                "outputs": return_type
            }
            function_signatures.append(signature)
        elif isinstance(node, ast.ClassDef):
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef):
                    args = [f"{arg.arg}: {ast.unparse(arg.annotation)}" for arg in class_node.args.args]
                    return_type = get_return_type(class_node)
                    signature = {
                        "function_name": f"{node.name}.{class_node.name}",
                        "inputs": args,
                        "outputs": return_type
                    }
                    function_signatures.append(signature)
    
    return function_signatures

def get_return_type(function_node):
    return_types = []
    for node in ast.walk(function_node):
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Name):
                return_types.append(f"{node.value.id}: {type(node.value).__name__}")
            elif isinstance(node.value, ast.Constant):
                return_types.append(f"{type(node.value.value).__name__}: {type(node.value).__name__}")
            elif isinstance(node.value, ast.Call):
                return_types.append(f"{node.value.func.id}: {type(node.value).__name__}")
    return return_types if return_types else ['None']

# Example usage
file_path = "C:\\Users\\sujain\\Downloads\\pdfs\\measurements\measurement-plugin-python-main\\examples\\game_of_life\\measurement.py"
signatures = get_function_signatures(file_path)

if not signatures:
    print("No functions or methods found in the file.")
else:
    for signature in signatures:
        print(f"function_name: {signature['function_name']}, inputs: {signature['inputs']}, outputs: {signature['outputs']}")
