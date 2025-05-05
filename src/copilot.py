from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

# Example lists
list1 = [
    "I love apples",
    "The weather is sunny",
    "Python programming is fun"
]

list2 = [
    "It is raining all day",
    "I enjoy Python coding",
    "I adore oranges"
]

# Load a pre-trained sentence transformer model.
# 'all-MiniLM-L6-v2' is a popular model balancing speed and accuracy.
model = SentenceTransformer('all-MiniLM-L6-v2')

# Compute embeddings for both lists
embeddings1 = model.encode(list1)
embeddings2 = model.encode(list2)

# Compute cosine similarity between each sentence in list1 and each sentence in list2
similarity_matrix = cosine_similarity(embeddings1, embeddings2)

# Create a DataFrame to clearly see the similarity scores
df = pd.DataFrame(similarity_matrix, index=list1, columns=list2)
print("Cosine Similarity Matrix:")
print(df)

# For each sentence in list1, find the best matching sentence in list2
print("\nBest Matches:")
for i, sentence1 in enumerate(list1):
    best_match_idx = np.argmax(similarity_matrix[i])
    best_match_score = similarity_matrix[i][best_match_idx]
    best_match_sentence = list2[best_match_idx]
    print(f"'{sentence1}' best matches with '{best_match_sentence}' (score: {best_match_score:.3f})")
