from sentence_transformers import SentenceTransformer, util

# Load a lightweight and efficient model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Specifications and Measurements

specs_sentence = [
    "Measurements are performed via a selected hardware access point with data gathered at a particular rate across a defined number of samples; the excitation voltage is controlled within a specified range, culminating in a documented signal representation."
    ,"leakage current, supply voltage range, propagation delay and capacitance will be passed as inputs and output voltage is expected to be between Vout(Min) and Vout(Max)"
    ,"Get waveform data where inputs are channel number, rate of sampling, count, voltage range"
    ,"in test session S1 on EQP R1, the target terminal was exposed to a controlled electrical stimulus—applying a preset voltage and regulated current within defined tolerance bands—following an initial delay, with the actual power metrics duly recorded."
    ,"During test cycle S1 on equipment R1, the designated pin was supplied with a calibrated voltage (within its tolerance range) and a limited current—after an initial delay interval—with voltage and current readings logged for performance validation."
    ,"In test session on resource , the designated terminal was energized with a predetermined voltage of (maintained within the acceptable range and a set current of values, following a scheduled source delay, with the resulting measurements recorded as vout and Iout"
    ,"In test cycle on test system A, the corresponding pin was activated by applying a set voltage within the permitted tolerance and a regulated current within its allowed range, post a defined delay, with the actual voltage and current captured as outputs to confirm proper operation."
    ,"The measurement process involves applying a fixed electrical potential and a controlled current—each maintained within predefined limits—after a brief delay, with the resulting power metrics then captured for evaluation."
    ,"Initiating the measurement process, a set electrical potential is applied and a moderated current enforced within its safe range post an intentional pause, with the subsequent power values being logged for evaluation."
    ,"The procedure entails choosing a specific physical channel for data acquisition, where measurements are taken at a preset sampling frequency for a predetermined number of samples, bounded within specified voltage limits, resulting in a recorded waveform."
]


measurements_sentences = {}
measurement_dict = {}

import csv

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
        measurements_sentences[i] = list1
        measurement_dict[i] = list1, target_row["File Path"], target_row["Function"], target_row["Input Parameters"], target_row["Output Parameters"]
        i = i + 1

# Print the list of description values
# print(description_values)

# # Print each column and its values
# for column in rows[0].keys():
#     print(f"Column: {column}")
#     for row in rows:
#         print(row[column])
#     print("-" * 40)  # Separator between columns



    



# print(measurements_sentences[0])
# print(measurements_sentences[1])
# print(measurements_sentences[2])

measurements_sentences_list = list(measurements_sentences.values())

# Encode both sets
spec_embeddings = model.encode(specs_sentence, convert_to_tensor=True)
measurement_embeddings = model.encode(measurements_sentences_list, convert_to_tensor=True)

# Match each spec to its best measurement
print("Matching results:\n")
for i, spec in enumerate(specs_sentence):
    cosine_scores = util.pytorch_cos_sim(spec_embeddings[i], measurement_embeddings)
    best_idx = cosine_scores.argmax().item()
    print(f"✅ Spec: {spec}\n")
    print(f"🔁 Best Measurement Match: {measurements_sentences_list[best_idx]}\n")
    print(f"cosine similarity score: {cosine_scores[0][best_idx]:.3f}\n")
    print(f"Info:{measurement_dict.get(best_idx)} \n")
    print(f"----------------------")
