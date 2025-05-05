import spacy
from spacy.pipeline import EntityRuler
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Create a custom spaCy pipeline with added rules for your technical domain.
# -----------------------------------------------------------------------------

# Load spaCy's default small English model.
nlp = spacy.load("en_core_web_sm")

# Add an EntityRuler to capture custom technical entities.
# You can add as many patterns as needed to catch your domain-specific tokens.
ruler = nlp.add_pipe("entity_ruler", after="ner")
patterns = [
    # For measurements like voltage (e.g., "220V", "220 V", "220 volts")
    {
        "label": "MEASUREMENT",
        "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["v", "volts", "volt"]}}],
    },
    # For current measurements (e.g., "5A", "5 A", "5 amps")
    {
        "label": "MEASUREMENT",
        "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["a", "amps", "ampere", "amp"]}}],
    },
    # Add custom rules for device model names
    {
        "label": "DEVICE",
        "pattern": [{"LOWER": "iphone"}]  # This is a simple example.
    },
    # You can extend with attributes common in your documents (e.g., calibration, test ID, etc.)
]

ruler.add_patterns(patterns)

# -----------------------------------------------------------------------------
# 2. Define lists of sentences that might come from your technical documents.
# -----------------------------------------------------------------------------

list1 = [
    "Apple Inc. released the new iPhone in September with a battery voltage of 220V.",
    "The circuit test recorded a voltage measurement of 220 volts.",
    "Python programming is both fun and powerful."
]

list2 = [
    "In September, Apple unveiled its latest smartphone featuring a 220 V battery.",
    "The electrical test indicated the circuit voltage was 220 volts.",
    "I enjoy coding in Python every day."
]

# -----------------------------------------------------------------------------
# 3. Function to extract custom attributes (entities) from a sentence.
# -----------------------------------------------------------------------------

def extract_attributes(sentence):
    doc = nlp(sentence)
    # Collect both default entities and custom ones.
    return set([ent.text.lower() for ent in doc.ents])

# Extract attributes for every sentence in both lists.
attributes_list1 = [extract_attributes(sentence) for sentence in list1]
attributes_list2 = [extract_attributes(sentence) for sentence in list2]

# -----------------------------------------------------------------------------
# 4. Compute attribute similarity using Jaccard similarity.
# -----------------------------------------------------------------------------

def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# Build an attribute similarity matrix between the two lists.
attr_similarity_matrix = np.zeros((len(list1), len(list2)))
for i, attrs1 in enumerate(attributes_list1):
    for j, attrs2 in enumerate(attributes_list2):
        attr_similarity_matrix[i, j] = jaccard_similarity(attrs1, attrs2)

# -----------------------------------------------------------------------------
# 5. Compute semantic similarity using Sentence Transformers.
# -----------------------------------------------------------------------------

# Load a pre-trained Sentence Transformer model.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for both lists.
embeddings1 = model.encode(list1)
embeddings2 = model.encode(list2)

# Compute semantic similarity using cosine similarity.
semantic_similarity_matrix = cosine_similarity(embeddings1, embeddings2)

# -----------------------------------------------------------------------------
# 6. Combine the similarity scores.
# -----------------------------------------------------------------------------

# Define weights, e.g., 70% semantic similarity and 30% attribute similarity.
combined_similarity_matrix = 0.7 * semantic_similarity_matrix + 0.3 * attr_similarity_matrix

# -----------------------------------------------------------------------------
# 7. Display Results.
# -----------------------------------------------------------------------------

# Display the combined similarity matrix in a DataFrame.
df_combined = pd.DataFrame(combined_similarity_matrix, index=list1, columns=list2)
print("Combined Similarity Matrix:")
print(df_combined)

# For each sentence in list1, display the best matching sentence from list2
print("\nBest Matches (using combined score):")
for i, sentence1 in enumerate(list1):
    best_match_idx = np.argmax(combined_similarity_matrix[i])
    best_match_sentence = list2[best_match_idx]
    sem_score = semantic_similarity_matrix[i][best_match_idx]
    attr_score = attr_similarity_matrix[i][best_match_idx]
    combined_score = combined_similarity_matrix[i][best_match_idx]
    print(f"'{sentence1}'")
    print(f"  best matches with:")
    print(f"    '{best_match_sentence}'")
    print(f"  Semantic Score: {sem_score:.3f} | Attribute Score: {attr_score:.3f} | Combined: {combined_score:.3f}\n")
