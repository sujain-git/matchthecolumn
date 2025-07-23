# folder_browser.py
import ast
import os
import csv
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel

class FolderBrowser(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Folder")
        self.setGeometry(100, 100, 400, 150)

        layout = QVBoxLayout()

        self.button = QPushButton("Browse Folder")
        self.button.clicked.connect(self.select_folder)
        layout.addWidget(self.button)

        self.label = QLabel("No folder selected.")
        layout.addWidget(self.label)

        self.selected_path = None
        

        self.extract_button = QPushButton("Extract Measurement MetaData")
        self.extract_button.clicked.connect(self.extract_measurements_info)
        layout.addWidget(self.extract_button)

        self.label1 = QLabel("No Measurement Metadata.")
        layout.addWidget(self.label1)
        layout.addWidget(self.label1)

        self.metadata_file_path = None
        self.setLayout(layout)

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.selected_path = folder_path
            self.label.setText(f"Selected Folder:\n{folder_path}")
        else:
            self.label.setText("No folder selected.")

    def extract_function_info_from_file(self, file_path):
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
    def printSignatures(self, signatures):
        if not signatures:
            print("No functions or methods found in the file.")
        else:
            for signature in signatures:
                print(f"function_name: {signature['function_name']}, inputs: {signature['inputs']}, outputs: {signature['outputs']}")


    def generate_descriptions(self, functions):
        descriptions = []
        for func in functions:
            function_name = func['function_name']
            inputs = ", ".join(func['inputs'])
            outputs = ", ".join(func['outputs'])

            # Create the description sentence
            description = f"Perform {function_name} where:\n inputs are {inputs} \n and \n outputs are {outputs}. \n"
            description = (
        f"Perform `{function_name}` where:\n"
        f"  • Inputs: {inputs}\n"
        f"  • Outputs: {outputs}\n"
    )

            descriptions.append(description)

        return descriptions

    # def iterate_folder(folder_path):
    #     folder_path = os.path.expanduser(folder_path)
    #     all_signatures = []

    #     for root, dirs, files in os.walk(folder_path):
    #         for file in files:
    #             if file.endswith('.py') and file == 'measurement.py':
    #                 file_path = os.path.join(root, file)
    #                 print(f"Processing file: {file_path}")
    #                 try:
    #                     signatures = extract_function_info_from_file(file_path)
    #                     descriptions = generate_descriptions(signatures)
    #                     for desc in descriptions:
    #                         print(desc)
    #                     print("-------------------------")
    #                     all_signatures.extend(signatures)
    #                 except Exception as e:
    #                     print(f"Error processing {file_path}: {e}")

    #     return all_signatures

    def iterate_folder(self, folder_path):
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.join(script_dir, 'data', 'parsed_functions.csv')

        folder_path = os.path.expanduser(folder_path)
        all_signatures = []

        # Open CSV file for writing
        with open(relative_path, mode='w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            # Write header row
            writer.writerow(['File Path', 'Function', 'Inputs', 'Outputs'])

            # Walk through the folder recursively
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.endswith('.py') and file == 'measurement.py':
                        file_path = os.path.join(root, file)
                        print(f"Processing file: {file_path}")
                        try:
                            signatures = self.extract_function_info_from_file(file_path)
                            for sig in signatures:
                                function_name = sig['function_name']
                                inputs = ", ".join(sig['inputs'])
                                outputs = ", ".join(sig['outputs'])
                                # Write each function's info to the CSV
                                writer.writerow([file_path, function_name, inputs, outputs])
                            all_signatures.extend(signatures)
                        except Exception as e:
                            print(f"Error processing {file_path}: {e}")
        
        return all_signatures, relative_path

    def point_to_python_measurements():  
        folder_path = input("Enter the folder path for python measurements: ")
        print(f"You entered: {folder_path}")
        return folder_path


    def extract_measurements_info(self, python_measurements_folder):
        # Example usage
        folder_path = self.selected_path #"C:/dev/git/matchthecolumn/measurement-plugin-python-main/examples"
        signatures, metadata_file = self.iterate_folder(folder_path)
        self.metadata_file_path = metadata_file

        self.label1.setText(f"MetaData file location:\n{metadata_file}")
        return signatures




def RunUI():
    app = QApplication(sys.argv)
    browser = FolderBrowser()
    browser.show()
    app.exec_()
    python_measurements_folder = browser.selected_path
    print(os.getcwd())




RunUI()