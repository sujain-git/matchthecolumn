import ast
import os

def extract_function_info_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)

    functions = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # This is a class, so we need to check for methods inside it
            class_name = node.name

            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef):
                    func_info = {
                        "function_name": class_node.name,
                        "inputs": [],
                        "outputs": [],
                        "class": class_name  # Add the class name to indicate this method belongs to the class
                    }

                    # Handle method arguments (including self for instance methods)
                    for arg in class_node.args.args:
                        arg_name = arg.arg
                        if arg.annotation:
                            arg_type = ast.unparse(arg.annotation)
                            func_info["inputs"].append(f"{arg_name}: {arg_type}")
                        else:
                            func_info["inputs"].append(arg_name)

                    # Handle return type annotation
                    if class_node.returns:
                        return_type = ast.unparse(class_node.returns)
                        func_info["outputs"].append(f"return: {return_type}")

                    # Extract output from decorators like @output("name", type)
                    for decorator in class_node.decorator_list:
                        if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'attr'):
                            if decorator.func.attr == "output" and len(decorator.args) >= 2:
                                try:
                                    output_name = ast.literal_eval(decorator.args[0])
                                except Exception:
                                    output_name = ast.unparse(decorator.args[0])
                                output_type = ast.unparse(decorator.args[1])
                                func_info["outputs"] = [f"{output_name}: {output_type}"]

                    functions.append(func_info)

        elif isinstance(node, ast.FunctionDef):
            # This is a regular function (not inside a class)
            func_info = {
                "function_name": node.name,
                "inputs": [],
                "outputs": []
            }

            # Handle function arguments
            for arg in node.args.args:
                arg_name = arg.arg
                if arg.annotation:
                    arg_type = ast.unparse(arg.annotation)
                    func_info["inputs"].append(f"{arg_name}: {arg_type}")
                else:
                    func_info["inputs"].append(arg_name)

            # Handle return type annotation
            if node.returns:
                return_type = ast.unparse(node.returns)
                func_info["outputs"].append(f"return: {return_type}")

            # Extract output from decorators like @output("name", type)
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'attr'):
                    if decorator.func.attr == "output" and len(decorator.args) >= 2:
                        try:
                            output_name = ast.literal_eval(decorator.args[0])
                        except Exception:
                            output_name = ast.unparse(decorator.args[0])
                        output_type = ast.unparse(decorator.args[1])
                        func_info["outputs"] = [f"{output_name}: {output_type}"]

            functions.append(func_info)

    return functions

# Example usage
def printSignatures(signatures):
    if not signatures:
        print("No functions or methods found in the file.")
    else:
        for signature in signatures:
            print(f"function_name: {signature['function_name']}, inputs: {signature['inputs']}, outputs: {signature['outputs']}")


def generate_descriptions(functions):
    descriptions = []
    for func in functions:
        function_name = func['function_name']
        inputs = ", ".join(func['inputs'])
        outputs = ", ".join(func['outputs'])

        # Create the description sentence
        description = f"Perform {function_name} where inputs are {inputs} and outputs are {outputs}."
        descriptions.append(description)

    return descriptions

def iterate_folder(folder_path):
    folder_path = os.path.expanduser(folder_path)
    all_signatures = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.py') and file == 'measurement.py':
                file_path = os.path.join(root, file)
                print(f"Processing file: {file_path}")
                try:
                    signatures = extract_function_info_from_file(file_path)
                    descriptions = generate_descriptions(signatures)
                    for desc in descriptions:
                        print(desc)
                    print("-------------------------")
                    all_signatures.extend(signatures)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return all_signatures

# Example usage
folder_path = "~/Desktop/dev/matchthecolumn/src/python_examples/"
signatures = iterate_folder(folder_path)