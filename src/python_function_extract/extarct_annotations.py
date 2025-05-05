import ast

def get_function_signatures(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    function_signatures = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [f"{arg.arg}: {ast.unparse(arg.annotation)}" for arg in node.args.args]
            return_type = ast.unparse(node.returns) if node.returns else 'None'
            signature = f"{node.name}({', '.join(args)}) -> {return_type}"
            function_signatures.append(signature)
        elif isinstance(node, ast.ClassDef):
            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef):
                    args = [f"{arg.arg}: {ast.unparse(arg.annotation)}" for arg in class_node.args.args]
                    return_type = ast.unparse(class_node.returns) if class_node.returns else 'None'
                    signature = f"{node.name}.{class_node.name}({', '.join(args)}) -> {return_type}"
                    function_signatures.append(signature)
    
    return function_signatures

# Example usage
file_path = "C:\\Users\\sujain\\Downloads\\pdfs\\measurements\measurement-plugin-python-main\\examples\\game_of_life\\measurement.py"
signatures = get_function_signatures(file_path)

if not signatures:
    print("No functions or methods found in the file.")
else:
    print("Function and method signatures found:", signatures)
