import ast

def get_function_signatures(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    function_signatures = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [arg.arg for arg in node.args.args]
            return_type = get_return_type(node)
            signature = f"{node.name}({', '.join(args)}) -> {return_type}"
            function_signatures.append(signature)
        elif isinstance(node, ast.ClassDef):
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef):
                    args = [arg.arg for arg in class_node.args.args]
                    return_type = get_return_type(class_node)
                    signature = f"{node.name}.{class_node.name}({', '.join(args)}) -> {return_type}"
                    function_signatures.append(signature)
    
    return function_signatures

def get_return_type(function_node):
    for node in ast.walk(function_node):
        if isinstance(node, ast.Return):
            if isinstance(node.value, ast.Name):
                return node.value.id
            elif isinstance(node.value, ast.Constant):
                return type(node.value.value).__name__
            elif isinstance(node.value, ast.Call):
                return node.value.func.id
    return 'None'

# Example usage
file_path = "C:\\Users\\sujain\\Downloads\\pdfs\\measurements\measurement-plugin-python-main\\examples\\game_of_life\\measurement.py"
signatures = get_function_signatures(file_path)

if not signatures:
    print("No functions or methods found in the file.")
else:
    print("Function and method signatures found:", signatures)
