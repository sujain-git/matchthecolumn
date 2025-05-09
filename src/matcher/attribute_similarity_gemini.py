import pandas as pd
from sentence_transformers import SentenceTransformer, util
import re
from typing import List, Dict, Tuple, Any
import numpy as np
from collections import Counter

# --- Semantic Matching ---

# Load a pre-trained sentence transformer model
# 'all-MiniLM-L6-v2' is a good balance of speed and performance
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
        A string representing the inferred domain (e.g., 'number', 'text', 'date', 'boolean', 'categorical').
        Returns 'unknown' if a common domain cannot be easily inferred.
    """
    if pd.api.types.is_numeric_dtype(data):
        return 'number'
    elif pd.api.types.is_datetime64_any_dtype(data):
        return 'date'
    elif pd.api.types.is_bool_dtype(data):
        return 'boolean'
    elif pd.api.types.is_object_dtype(data) or pd.api.types.is_string_dtype(data):
        # Check if it's likely categorical (many repeated values)
        if data.nunique() / len(data) < 0.1 and data.nunique() > 1: # Threshold for considering categorical
             return 'categorical'
        # Further checks could be added here for specific patterns (e.g., email, URL)
        if data.astype(str).str.contains(r'@', na=False).any():
             return 'text (potentially email)'
        elif data.astype(str).str.contains(r'http[s]?://', na=False).any():
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
     # Consider different text/categorical types compatible for basic matching
    if (domain1.startswith('text') or domain1 == 'categorical') and \
       (domain2.startswith('text') or domain2 == 'categorical'):
        return True
    # Add other compatibility rules as needed
    return False

# --- Attribute Matching ---

def get_attribute_similarity(data1: pd.Series, data2: pd.Series, inferred_domain: str) -> float:
    """
    Calculates attribute similarity between two columns based on their inferred domain.

    Args:
        data1: The first pandas Series (column data).
        data2: The second pandas Series (column data).
        inferred_domain: The inferred domain of the columns (assumed to be compatible).

    Returns:
        A similarity score between 0 and 1, or -1 if similarity cannot be calculated
        for the given domain type.
    """
    # Handle potential NaNs by dropping them for similarity calculation
    data1_clean = data1.dropna()
    data2_clean = data2.dropna()

    if data1_clean.empty or data2_clean.empty:
        return 0.0 # No similarity if one or both are empty after dropping NaNs

    if inferred_domain == 'number':
        # Basic statistical similarity (e.g., normalized difference in mean and std dev)
        # More advanced methods could compare distributions directly (e.g., Jensen-Shannon divergence)
        mean1, std1 = data1_clean.mean(), data1_clean.std()
        mean2, std2 = data2_clean.mean(), data2_clean.std()

        # Avoid division by zero if std dev is 0
        mean_diff = abs(mean1 - mean2) / max(abs(mean1), abs(mean2), 1e-9) if max(abs(mean1), abs(mean2)) > 1e-9 else 0
        std_diff = abs(std1 - std2) / max(abs(std1), abs(std2), 1e-9) if max(abs(std1), abs(std2)) > 1e-9 else 0

        # Combine differences (simple average of normalized differences)
        # Invert the difference to get a similarity score (1 - difference)
        statistical_difference = (mean_diff + std_diff) / 2
        attribute_sim = 1 - statistical_difference if statistical_difference <= 1 else 0

        return attribute_sim

    elif inferred_domain == 'categorical' or inferred_domain.startswith('text'):
        # Jaccard similarity of unique values for categorical/text
        set1 = set(data1_clean.astype(str).unique())
        set2 = set(data2_clean.astype(str).unique())

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 1.0 if intersection == 0 else 0.0 # Both empty or one empty after dropping NaNs
        else:
            return intersection / union

    # Add other domain types and their similarity measures here (e.g., date ranges)
    # elif inferred_domain == 'date':
    #     # Compare date ranges, frequency patterns, etc.
    #     pass

    return -1 # Indicate that attribute similarity calculation is not supported for this domain type

# --- Combined Matching ---

def match_columns(
    list1_cols: List[str],
    list2_cols: List[str],
    df1: pd.DataFrame = None,
    df2: pd.DataFrame = None,
    semantic_similarity_threshold: float = 0.6,
    require_domain_compatibility: bool = True,
    attribute_similarity_threshold: float = 0.0 # New threshold for attribute similarity
) -> List[Tuple[str, str, float, bool, float]]:
    """
    Matches columns between two lists based on semantic similarity, domain compatibility,
    and attribute similarity.

    Args:
        list1_cols: A list of column names from the first source.
        list2_cols: A list of column names from the second source.
        df1: The first DataFrame (optional, needed for domain and attribute inference).
        df2: The second DataFrame (optional, needed for domain and attribute inference).
        semantic_similarity_threshold: The minimum semantic similarity score
                                       to consider a potential match.
        require_domain_compatibility: Whether to require domain compatibility
                                      for a match.
        attribute_similarity_threshold: The minimum attribute similarity score
                                        to consider a potential match (only checked
                                        if df1 and df2 are provided and domains
                                        are compatible).

    Returns:
        A list of tuples, where each tuple contains:
        (column_from_list1, column_from_list2, semantic_similarity, domain_compatible, attribute_similarity).
        Only potential matches exceeding the semantic and attribute similarity thresholds (if applicable)
        are returned. attribute_similarity will be -1 if not calculated.
    """
    matches = []

    for col1 in list1_cols:
        for col2 in list2_cols:
            # Perform semantic matching
            similarity = get_semantic_similarity(col1, col2)

            if similarity >= semantic_similarity_threshold:
                domain_compatible = True
                attribute_sim = -1.0 # Default to -1 if not calculated

                if df1 is not None and df2 is not None:
                    if col1 in df1.columns and col2 in df2.columns:
                        domain1 = infer_domain(df1[col1])
                        domain2 = infer_domain(df2[col2])
                        domain_compatible = are_domains_compatible(domain1, domain2)

                        # Perform attribute matching if domains are compatible
                        if domain_compatible:
                            attribute_sim = get_attribute_similarity(df1[col1], df2[col2], domain1) # Use domain1 as they are compatible
                    else:
                         # Cannot check domain/attribute compatibility if column not in DataFrame
                         domain_compatible = False
                         attribute_sim = -1.0 # Cannot calculate attribute similarity

                # Check attribute similarity threshold if required and calculated
                if attribute_sim != -1.0 and attribute_sim < attribute_similarity_threshold:
                     continue # Skip this match if attribute similarity is below threshold

                # Only add if domain is compatible (if required) or if not required
                if not require_domain_compatibility or domain_compatible:
                    matches.append((col1, col2, similarity, domain_compatible, attribute_sim))

    # Sort matches primarily by semantic similarity, then by attribute similarity
    matches.sort(key=lambda x: (x[2], x[4]), reverse=True)

    return matches

# --- Example Usage ---

# Example DataFrames (replace with your actual data loading)
data1 = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
         'AgeYears': [25, 30, 35, 40],
         'City_of_Residence': ['New York', 'London', 'Paris', 'New York'],
         'Country': ['USA', 'UK', 'France', 'USA']} # Added Country
df1 = pd.DataFrame(data1)

data2 = {'PersonName': ['David', 'Eve', 'Frank', 'Grace'],
         'AgeInYears': [40, 45, 50, 22],
         'HomeTown': ['Tokyo', 'Sydney', 'Berlin', 'London'],
         'Salary': [50000, 60000, 70000, 55000], # Added an extra column
         'Nationality': ['Japan', 'Australia', 'Germany', 'UK']} # Added Nationality
df2 = pd.DataFrame(data2)

list1_columns = df1.columns.tolist()
list2_columns = df2.columns.tolist()

print("Matching columns based on semantic similarity, domain compatibility, and attribute similarity:")
matched_columns = match_columns(
    list1_columns,
    list2_columns,
    df1=df1,
    df2=df2,
    semantic_similarity_threshold=0.5,
    attribute_similarity_threshold=0.3 # Set an attribute similarity threshold
)

for col1, col2, semantic_sim, domain_compatible, attribute_sim in matched_columns:
    print(f"'{col1}' <--> '{col2}' | Semantic Sim: {semantic_sim:.4f} | Domain Compatible: {domain_compatible} | Attribute Sim: {attribute_sim:.4f}")

print("\nMatching columns based on semantic similarity and domain compatibility only (no attribute threshold):")
matched_columns_no_attribute_threshold = match_columns(
    list1_columns,
    list2_columns,
    df1=df1,
    df2=df2,
    semantic_similarity_threshold=0.5,
    require_domain_compatibility=True,
    attribute_similarity_threshold=0.0 # Attribute similarity doesn't filter if threshold is 0
)

for col1, col2, semantic_sim, domain_compatible, attribute_sim in matched_columns_no_attribute_threshold:
     print(f"'{col1}' <--> '{col2}' | Semantic Sim: {semantic_sim:.4f} | Domain Compatible: {domain_compatible} | Attribute Sim: {attribute_sim:.4f}")

# Example with slightly different names and data types, and less attribute overlap
data3 = {'Product_Title': ['Laptop', 'Keyboard', 'Mouse', 'Monitor'],
         'Price_USD': [1200, 75, 25, 300],
         'In_Stock': [True, True, False, True]}
df3 = pd.DataFrame(data3)

data4 = {'ItemName': ['Monitor', 'Desk', 'Laptop Computer', 'Printer'],
         'Cost_in_dollars': [300, 150, 1150, 200],
         'Available': [True, False, True, True]}
df4 = pd.DataFrame(data4)

list3_columns = df3.columns.tolist()
list4_columns = df4.columns.tolist()

print("\nMatching product columns with attribute matching:")
matched_product_columns = match_columns(
    list3_columns,
    list4_columns,
    df1=df3,
    df2=df4,
    semantic_similarity_threshold=0.4,
    attribute_similarity_threshold=0.2 # Set an attribute similarity threshold
)

for col1, col2, semantic_sim, domain_compatible, attribute_sim in matched_product_columns:
    print(f"'{col1}' <--> '{col2}' | Semantic Sim: {semantic_sim:.4f} | Domain Compatible: {domain_compatible} | Attribute Sim: {attribute_sim:.4f}")