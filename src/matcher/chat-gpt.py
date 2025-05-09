from sentence_transformers import SentenceTransformer, util

# Load a lightweight and efficient model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Specifications and Measurements
specs = [
    "Input leakage current",
    "Supply voltage range",
    "Output high voltage",
    "Output low voltage",
    "Propagation delay time",
    "Input capacitance"
]

specs_sentence = [
"Measurements are performed via a selected hardware access point with data gathered at a particular rate across a defined number of samples; the excitation voltage is controlled within a specified range, culminating in a documented signal representation."
    ,"leakage current, supply voltage range, propagation delay and capacitance will be passed as inputs and output voltage is expected to be between Vout(Min) and Vout(Max)"
    ,"Get waveform data where inputs are channel number, rate of sampling, count, voltage range"
    ,"in test session S1 on EQP R1, the target terminal was exposed to a controlled electrical stimulus—applying a preset voltage and regulated current within defined tolerance bands—following an initial delay, with the actual power metrics duly recorded."
    ,"During test cycle S1 on equipment R1, the designated pin was supplied with a calibrated voltage (within its tolerance range) and a limited current—after an initial delay interval—with voltage and current readings logged for performance validation."
    ,"In test session on resource , the designated terminal was energized with a predetermined voltage of (maintained within the acceptable range and a set current of values, following a scheduled source delay, with the resulting measurements recorded as vout and Iout"
    ,"In test cycle on test system A, the corresponding pin was activated by applying a set voltage within the permitted tolerance and a regulated current within its allowed range, post a defined delay, with the actual voltage and current captured as outputs to confirm proper operation."
,"The measurement process involves applying a fixed electrical potential and a controlled current—each maintained within predefined limits—after a brief delay, with the resulting power metrics then captured for evaluation."
,"nitiating the measurement process, a set electrical potential is applied and a moderated current enforced within its safe range post an intentional pause, with the subsequent power values being logged for evaluation."
, "The procedure entails choosing a specific physical channel for data acquisition, where measurements are taken at a preset sampling frequency for a predetermined number of samples, bounded within specified voltage limits, resulting in a recorded waveform."
]

measurements = [
    "IIN: ±1 µA max",
    "VCC: 2V to 5.5V",
    "VOH: 4.8V min (at IOH = -20 µA)",
    "VOL: 0.1V max (at IOL = 20 µA)",
    "tpd: 12 ns typical",
    "CIN: 4 pF typical"
]


lol_words = [
["voltage_level", "current_limit", "voltage_level_range", "current_limit_range", "source_delay", "voltage_measurement", "current_measurement"],
["pin_name", "voltage_level", "current_limit", "voltage_level_range", "current_limit_range", "source_delay", "voltage_measurement", "current_measurement", "resource_name", "session_name"],
["name", "num_responses", "data_size", "cumulative_data", "response_interval_in_ms", "error_on_index", "name", "index", "data"],

["physical_channel", "sample_rate", "number_of_samples", "minimum_voltage", "maximum_voltage", "waveform"],

  [ "voltage_level",
    "voltage_level_range" ,
    "current_limit"       ,
    "current_limit_range" ,
    "source_delay"        ,
    "input_pin"           ,
    "measurement_type"    ,
    "range"               ,
    "resolution_digits"   ,
    "output_pin"
  ]
]

lol_sentence = [
["perform a measurement with inputs: voltage_level, current_limit, voltage_level_range, current_limit_range, source_delay and outputs: voltage_measurement, current_measurement"]
,["perform a measurement with inputs: pin_name, voltage_level, current_limit, voltage_level_range, current_limit_range, source_delay and  outputs: voltage_measurement, current_measurement, resource_name, session_name"]

,["perform a measurement with inputs: name, num_responses, data_size, cumulative_data, response_interval_in_ms, error_on_index and outputs: name, index, data"]


,["perform a measurement with inputs: physical_channel, sample_rate, number_of_samples, minimum_voltage, maximum_voltage and outputs: waveform"] 

,["perform a measurement with inputs: voltage_level, voltage_level_range , current_limit, current_limit_range, source_delay, input_pin, measurement_type and outputs: range, resolution_digits, output_pin"]
,["perform a measurment with inputs: voltage level, capacitance, leakage in milli amperes, delay and output is voltage in mV"]
]

measurements2 = []

for list1 in lol_sentence:
    measurements2.append(list1[0])

print(measurements2[0])
print(measurements2[1])
print(measurements2[2])

# Encode both sets
spec_embeddings = model.encode(specs_sentence, convert_to_tensor=True)
measurement_embeddings = model.encode(measurements2, convert_to_tensor=True)

# Match each spec to its best measurement
print("Matching results:\n")
for i, spec in enumerate(specs_sentence):
    cosine_scores = util.pytorch_cos_sim(spec_embeddings[i], measurement_embeddings)
    best_idx = cosine_scores.argmax()
    print(f"✅ Spec: {spec}")
    print(f"🔁 Best Match: {measurements2[best_idx]} (score: {cosine_scores[0][best_idx]:.3f})\n")
