import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
from typing import List, Dict, Tuple, Any

# --- Semantic Matching ---

# Load a pre-trained sentence transformer model
# 'all- ミニLM-L6-v2' is a good balance of speed and performance
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_semantic_similarity(name1: str, name2: str) -> float:
    """
    Calculates the semantic similarity between two column names.

    Args:
        name1: The first column name.
        name2: The second column name.

    Returns:
        A cosine similarity score between the sentence embeddings of the names.
        Ranges from -1 to 1, where 1 is maximum similarity.
    """
    embeddings = model.encode([name1, name2], convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity_score

# --- Domain Matching ---

def infer_domain(data: pd.Series) -> str:
    """
    Infers a basic domain/data type for a pandas Series.
    This is a simplified example and can be extended for more complex domain inference.

    Args:
        data: The pandas Series (column data).

    Returns:
        A string representing the inferred domain (e.g., 'number', 'text', 'date', 'boolean').
        Returns 'unknown' if a common domain cannot be easily inferred.
    """
    if pd.api.types.is_numeric_dtype(data):
        return 'number'
    elif pd.api.types.is_datetime64_any_dtype(data):
        return 'date'
    elif pd.api.types.is_bool_dtype(data):
        return 'boolean'
    elif pd.api.types.is_string_dtype(data):
        # Further checks could be added here for specific patterns (e.g., email, URL)
        if data.str.contains(r'@', na=False).any():
             return 'text (potentially email)'
        elif data.str.contains(r'http[s]?://', na=False).any():
             return 'text (potentially url)'
        else:
            return 'text'
    else:
        return 'unknown'

def are_domains_compatible(domain1: str, domain2: str) -> bool:
    """
    Checks for compatibility between two inferred domains.
    This is a basic compatibility check.

    Args:
        domain1: The first inferred domain.
        domain2: The second inferred domain.

    Returns:
        True if the domains are considered compatible, False otherwise.
    """
    if domain1 == domain2:
        return True
    # Consider number types compatible (int, float, etc.)
    if domain1 == 'number' and domain2 == 'number':
        return True
    # Consider different text types compatible for basic matching
    if domain1.startswith('text') and domain2.startswith('text'):
        return True
    # Add other compatibility rules as needed
    return False

# --- Combined Matching ---

def match_columns(
    list1_cols: List[str],
    list2_cols: List[str],
    df1: pd.DataFrame = None,
    df2: pd.DataFrame = None,
    semantic_similarity_threshold: float = 0.6,
    require_domain_compatibility: bool = True
) -> List[Tuple[str, str, float, bool]]:
    """
    Matches columns between two lists based on semantic similarity and domain compatibility.

    Args:
        list1_cols: A list of column names from the first source.
        list2_cols: A list of column names from the second source.
        df1: The first DataFrame (optional, needed for domain inference).
        df2: The second DataFrame (optional, needed for domain inference).
        semantic_similarity_threshold: The minimum semantic similarity score
                                       to consider a potential match.
        require_domain_compatibility: Whether to require domain compatibility
                                      for a match.

    Returns:
        A list of tuples, where each tuple contains:
        (column_from_list1, column_from_list2, semantic_similarity, domain_compatible).
        Only potential matches exceeding the semantic similarity threshold are returned.
    """
    matches = []

    for col1 in list1_cols:
        for col2 in list2_cols:
            # Perform semantic matching
            similarity = get_semantic_similarity(col1, col2)

            if similarity >= semantic_similarity_threshold:
                domain_compatible = True
                if require_domain_compatibility and df1 is not None and df2 is not None:
                    if col1 in df1.columns and col2 in df2.columns:
                        domain1 = infer_domain(df1[col1])
                        domain2 = infer_domain(df2[col2])
                        domain_compatible = are_domains_compatible(domain1, domain2)
                    else:
                        # Cannot check domain compatibility if column not in DataFrame
                        domain_compatible = False # Or handle as needed

                # Only add if domain is compatible (if required) or if not required
                if not require_domain_compatibility or domain_compatible:
                    matches.append((col1, col2, similarity, domain_compatible))

    # Sort matches by semantic similarity in descending order
    matches.sort(key=lambda x: x[2], reverse=True)

    return matches

# --- Example Usage ---

# Example DataFrames (replace with your actual data loading)
data1 = {'Name': ['Alice', 'Bob', 'Charlie'],
         'AgeYears': [25, 30, 35],
         'City_of_Residence': ['New York', 'London', 'Paris']}
df1 = pd.DataFrame(data1)

data2 = {'PersonName': ['David', 'Eve', 'Frank'],
         'AgeInYears': [40, 45, 50],
         'HomeTown': ['Tokyo', 'Sydney', 'Berlin'],
         'Salary': [50000, 60000, 70000]} # Added an extra column
df2 = pd.DataFrame(data2)

list1_columns = df1.columns.tolist()
list2_columns = df2.columns.tolist()

print("Matching columns based on semantic similarity and domain compatibility:")
matched_columns = match_columns(list1_columns, list2_columns, df1=df1, df2=df2, semantic_similarity_threshold=0.5)

for col1, col2, similarity, domain_compatible in matched_columns:
    print(f"'{col1}' <--> '{col2}' | Semantic Similarity: {similarity:.4f} | Domain Compatible: {domain_compatible}")

print("\nMatching columns based on semantic similarity only:")
matched_columns_semantic_only = match_columns(list1_columns, list2_columns, semantic_similarity_threshold=0.5, require_domain_compatibility=False)

for col1, col2, similarity, domain_compatible in matched_columns_semantic_only:
     print(f"'{col1}' <--> '{col2}' | Semantic Similarity: {similarity:.4f} | Domain Compatible: {domain_compatible}")

# Example with slightly different names and data types
data3 = {'Product_Title': ['Laptop', 'Keyboard', 'Mouse'],
         'Price_USD': [1200, 75, 25]}
df3 = pd.DataFrame(data3)

data4 = {'ItemName': ['Monitor', 'Desk', 'Laptop Computer'],
         'Cost_in_dollars': [300, 150, 1150]}
df4 = pd.DataFrame(data4)

list3_columns = df3.columns.tolist()
list4_columns = df4.columns.tolist()

print("\nMatching product columns:")
matched_product_columns = match_columns(list3_columns, list4_columns, df3=df3, df4=df4, semantic_similarity_threshold=0.4)

for col1, col2, similarity, domain_compatible in matched_product_columns:
    print(f"'{col1}' <--> '{col2}' | Semantic Similarity: {similarity:.4f} | Domain Compatible: {domain_compatible}")