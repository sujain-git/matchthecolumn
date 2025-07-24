import sys
import csv
from sentence_transformers import SentenceTransformer, util
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel,
    QComboBox, QFileDialog, QPushButton, QTableWidget, QTableWidgetItem
)

class CSVViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CSV Viewer")
        self.setGeometry(200, 200, 760,630)

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.spec_data = {}
        self.measurements_sentences = {}
        self.measurement_dict = {}
        self.load_measurement_desc()
        self.measurement_embeddings = self.model.encode(self.measurements_sentences, convert_to_tensor=True)


        layout = QVBoxLayout()

        self.load_button = QPushButton("Load Spec Description File")
        self.load_button.width = 50;
        self.load_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_button)

        self.dropdown = QComboBox()
        self.dropdown.width = 50;
        self.dropdown.currentIndexChanged.connect(self.update_label)
        layout.addWidget(self.dropdown)

        self.label = QLabel("Select a specification to see the description.")
        self.label.width = 50;
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        # Add this inside your CSVViewer class __init__ method
        self.match_button = QPushButton("Find Matching Measurement")
        self.match_button.width = 50;
        self.match_button.clicked.connect(self.find_matching_measurements)
        layout.addWidget(self.match_button)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Rank", "Path", "Info"])
        layout.addWidget(self.table)
        self.setLayout(layout)

    def load_measurement_desc(self):
        # Define the file path
        file_path = "C:\\dev\\git\\matchthecolumn\\src\\matcher\\data\\parased_functions_with_desc_variations.csv"  # Replace with your actual file name

        # Read the CSV file into a list of dictionaries
        with open(file_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            rows = list(reader)

        i = 0;
        for row in rows:
        # Choose the row you want (e.g., the first row)
            target_row = row
        # Extract all columns that contain 'description' in their header
            description_values = [value for key, value in target_row.items() if 'description' in key.lower()]
            for list1 in description_values:
                self.measurements_sentences[i] = list1
                self.measurement_dict[i] = list1, target_row["File Path"], target_row["Function"], target_row["Input Parameters"], target_row["Output Parameters"]
                i = i + 1
    
    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            with open(file_path, newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                self.spec_data = {row["specifications"]: row["description"] for row in reader}
                self.dropdown.clear()
                self.dropdown.addItems(self.spec_data.keys())

    def update_label(self):
        key = self.dropdown.currentText()
        if key:
            self.label.setText(self.spec_data.get(key, "No description available."))

    def find_matching_measurements(self):
        key = self.dropdown.currentText()
        spec_sentence_to_match = self.spec_data.get(key, "No description available.")
        spec_embedding = self.model.encode(spec_sentence_to_match, convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(spec_embedding, self.measurement_embeddings)

        top3_indices = cosine_scores[0].topk(3).indices.tolist()
        measurements = []
        for rank, idx in enumerate(top3_indices, 1):
            print(f"{rank}. 🔁 Match: {self.measurements_sentences[idx]} \n")
            print(f"   🔢 Cosine Similarity: {cosine_scores[0][idx]:.3f} \n")
            print(f"   ℹ️ Info: {self.measurement_dict.get(idx)}\n")

            rank = f"{cosine_scores[0][idx]:.3f}"
            path = f"{self.measurement_dict.get(idx)[1]}"
            name = f"{self.measurement_dict.get(idx)}"
            measurements.append({"rank": rank, "path": path, "name": name})
        print(f"----------------------")

        print(measurements)

        # Sort and get top 3
        top_matches = sorted(measurements, key=lambda x: x["rank"], reverse=True)[:3]

        # Display in table
        self.table.setRowCount(len(top_matches))
        for row, match in enumerate(top_matches):
            
            rankitem = QTableWidgetItem(str(match["rank"]))
            rankitem.setFlags(Qt.ItemIsEnabled)
            rankitem.setToolTip(str(match["rank"]))  
            self.table.setItem(row, 0, rankitem)


            pathitem = QTableWidgetItem(match["path"])
            pathitem.setFlags(Qt.ItemIsEnabled)
            pathitem.setToolTip(match["path"])  
            self.table.setItem(row, 1, pathitem)

            nameitem = QTableWidgetItem(match["name"])
            nameitem.setFlags(Qt.ItemIsEnabled)
            nameitem.setToolTip(match["name"])  
            self.table.setItem(row, 2, nameitem)

            self.table.resizeColumnsToContents()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = CSVViewer()
    viewer.show()
    sys.exit(app.exec_())

# from sentence_transformers import SentenceTransformer, util

# # Load a lightweight and efficient model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Specifications and Measurements

# specs_sentence = [
#     "Measurements are performed via a selected hardware access point with data gathered at a particular rate across a defined number of samples; the excitation voltage is controlled within a specified range, culminating in a documented signal representation."
#     ,"leakage current, supply voltage range, propagation delay and capacitance will be passed as inputs and output voltage is expected to be between Vout(Min) and Vout(Max)"
#     ,"Get waveform data where inputs are channel number, rate of sampling, count, voltage range"
#     ,"in test session S1 on EQP R1, the target terminal was exposed to a controlled electrical stimulus—applying a preset voltage and regulated current within defined tolerance bands—following an initial delay, with the actual power metrics duly recorded."
#     ,"During test cycle S1 on equipment R1, the designated pin was supplied with a calibrated voltage (within its tolerance range) and a limited current—after an initial delay interval—with voltage and current readings logged for performance validation."
#     ,"In test session on resource , the designated terminal was energized with a predetermined voltage of (maintained within the acceptable range and a set current of values, following a scheduled source delay, with the resulting measurements recorded as vout and Iout"
#     ,"In test cycle on test system A, the corresponding pin was activated by applying a set voltage within the permitted tolerance and a regulated current within its allowed range, post a defined delay, with the actual voltage and current captured as outputs to confirm proper operation."
#     ,"The measurement process involves applying a fixed electrical potential and a controlled current—each maintained within predefined limits—after a brief delay, with the resulting power metrics then captured for evaluation."
#     ,"Initiating the measurement process, a set electrical potential is applied and a moderated current enforced within its safe range post an intentional pause, with the subsequent power values being logged for evaluation."
#     ,"The procedure entails choosing a specific physical channel for data acquisition, where measurements are taken at a preset sampling frequency for a predetermined number of samples, bounded within specified voltage limits, resulting in a recorded waveform."
# ]


# measurements_sentences = []
# measurement_dict = {}

# import csv

# # Define the file path
# file_path = "C:\\dev\\git\\matchthecolumn\\src\\matcher\\data\\parased_functions_with_desc_variations.csv"  # Replace with your actual file name

# # Read the CSV file into a list of dictionaries
# with open(file_path, mode='r', encoding='utf-8') as file:
#     reader = csv.DictReader(file)
#     rows = list(reader)

# i = 0;
# for row in rows:
# # Choose the row you want (e.g., the first row)
#     target_row = row
# # Extract all columns that contain 'description' in their header
#     description_values = [value for key, value in target_row.items() if 'description' in key.lower()]
#     for list1 in description_values:
#         measurements_sentences.append(list1)
#         measurement_dict[i] = list1, target_row["File Path"], target_row["Function"], target_row["Input Parameters"], target_row["Output Parameters"]
#         i = i + 1

# # Print the list of description values
# # print(description_values)

# # # Print each column and its values
# # for column in rows[0].keys():
# #     print(f"Column: {column}")
# #     for row in rows:
# #         print(row[column])
# #     print("-" * 40)  # Separator between columns



    



# # print(measurements_sentences[0])
# # print(measurements_sentences[1])
# # print(measurements_sentences[2])

# # Encode both sets
# spec_embeddings = model.encode(specs_sentence, convert_to_tensor=True)
# measurement_embeddings = model.encode(measurements_sentences, convert_to_tensor=True)

# # Match each spec to its best measurement
# print("Matching results:\n")
# for i, spec in enumerate(specs_sentence):
#     cosine_scores = util.pytorch_cos_sim(spec_embeddings[i], measurement_embeddings)
#     best_idx = cosine_scores.argmax().item()
#     print(f"✅ Spec: {spec}\n")
#     print(f"🔁 Best Measurement Match: {measurements_sentences[best_idx]}\n")
#     print(f"cosine similarity score: {cosine_scores[0][best_idx]:.3f}\n")
#     print(f"Info:{measurement_dict.get(best_idx)} \n")
#     print(f"----------------------")



