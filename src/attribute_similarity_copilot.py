import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Load spaCy's small English model.
# (For more domain-specific entities, consider a larger or custom spaCy model.)
nlp = spacy.load("en_core_web_sm")

# Example lists of sentences
list1 = [
    "Apple Inc. released the new iPhone in September.",
    "The voltage in the circuit was measured as 220V.",
    "Python programming is both fun and powerful."
]

list2 = [
    "In September, Apple unveiled its latest smartphone.",
    "The electrical test showed a voltage of 220 volts.",
    "I enjoy coding in Python every day."
]

# Function to extract attributes (named entities) from a sentence.
def extract_attributes(sentence):
    doc = nlp(sentence)
    # We lowercase for easier matching.
    return set([ent.text.lower() for ent in doc.ents])

# Extract attributes for every sentence in both lists.
attributes_list1 = [extract_attributes(sentence) for sentence in list1]
attributes_list2 = [extract_attributes(sentence) for sentence in list2]

# Define a function for computing attribute similarity using Jaccard similarity.
def jaccard_similarity(set1, set2):
    if not set1 and not set2:
        # When both sets are empty, we return 0 similarity.
        return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

# Build an attribute similarity matrix between the two lists.
attr_similarity_matrix = np.zeros((len(list1), len(list2)))
for i, attrs1 in enumerate(attributes_list1):
    for j, attrs2 in enumerate(attributes_list2):
        attr_similarity_matrix[i, j] = jaccard_similarity(attrs1, attrs2)

# Load a pre-trained Sentence Transformer model for semantic similarity.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute sentence embeddings for both lists.
embeddings1 = model.encode(list1)
embeddings2 = model.encode(list2)

# Compute semantic similarity using cosine similarity.
semantic_similarity_matrix = cosine_similarity(embeddings1, embeddings2)

# Combine the scores.
# Here we choose a weighted strategy (e.g., 70% semantic and 30% attribute).
combined_similarity_matrix = 0.7 * semantic_similarity_matrix + 0.3 * attr_similarity_matrix

# For visualization, put the combined scores in a DataFrame.
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
