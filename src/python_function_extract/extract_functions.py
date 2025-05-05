import ast

class FunctionInfoExtractor(ast.NodeVisitor):
    def __init__(self):
        self.functions = []

    def visit_FunctionDef(self, node):
        func_name = node.name
        inputs = [arg.arg for arg in node.args.args]
        outputs = self.get_return_values(node)
        self.functions.append((func_name, inputs, outputs))
        self.generic_visit(node)

    def get_return_values(self, node):
        return_values = []
        for n in ast.walk(node):
            if isinstance(n, ast.Return):
                return_values.append(ast.dump(n.value))
        return return_values

def extract_function_info(file_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read())
    extractor = FunctionInfoExtractor()
    extractor.visit(tree)
    return extractor.functions

# Example usage
file_path = "C:\\Users\\sujain\\Downloads\\pdfs\\measurements\measurement-plugin-python-main\\examples\\game_of_life\\measurement.py"
functions_info = extract_function_info(file_path)
for func_name, inputs, outputs in functions_info:
    print(f"Function: {func_name}")
    print(f"Inputs: {inputs}")
    print(f"Outputs: {outputs}")
    print()