import ast
import os

def get_function_signatures(file_path):
    with open(file_path, 'r') as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    function_signatures = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            args = [f"{arg.arg}: {ast.unparse(arg.annotation)}" for arg in node.args.args]
            return_type = ast.unparse(node.returns) if node.returns else 'None'
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
                    return_type = ast.unparse(class_node.returns) if class_node.returns else 'None'
                    signature = {
                        "function_name": f"{node.name}.{class_node.name}",
                        "inputs": args,
                        "outputs": return_type
                    }
                    function_signatures.append(signature)
    
    return function_signatures


# Example usage
def printSignatures(signatures):
    if not signatures:
        print("No functions or methods found in the file.")
    else:
        for signature in signatures:
            print(f"function_name: {signature['function_name']}, inputs: {signature['inputs']}, outputs: {signature['outputs']}")


def iterate_folder(folder_path):
    all_signatures = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py'):
                if file == 'measurement.py':
                    file_path = os.path.join(root, file)
                    signatures = get_function_signatures(file_path)
                    printSignatures(signatures)
                    print("-------------------------")
                    all_signatures.extend(signatures)
    
    return all_signatures

# Example usage
folder_path = "C:\\Users\\sujain\\Downloads\\pdfs\\measurements\\measurement-plugin-python-main\\examples"
signatures = iterate_folder(folder_path)


