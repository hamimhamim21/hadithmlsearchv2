from sentence_transformers import SentenceTransformer
import re
import faiss
import numpy as np
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# Function to remove bidirectional characters from text
import pickle
import os
# Define your base directory
base_dir = "tfidf"


def remove_bidirectional_characters(text):
    return re.sub(r'[\u200e\u200f\u202a-\u202e]', '', text)

# Function to generate an embedding for a single text


def string_to_bytes(string):
    return string.encode('latin1')


def generate_single_embedding(text, model):
    embedding = model.encode(text, convert_to_tensor=False)
    return embedding


def fetch_embeddings(db_conn, language_code):
    sql_query = text(
        "SELECT id, embedding FROM embeddings WHERE language_code = :language_code;")
    params = {'language_code': language_code}
    for chunk in pd.read_sql_query(sql_query, db_conn.connect(), params=params, chunksize=1000):
        yield chunk


# Function to calculate search and rank results using fetch_embeddings
def search_and_rank_results_onlyids(query, model, language_code, db_conn_embedding, top_n=10):
    query_embedding = generate_single_embedding(query, model)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    column_name = f"{language_code}_df_hadith_common_key"
    sql_query = text(
        f"SELECT {column_name}, embedding FROM embeddings WHERE {column_name} IS NOT NULL")

    embeddings_df = pd.read_sql(sql_query, db_conn_embedding.connect())
    if embeddings_df.empty:
        return [], []

    id_list = embeddings_df[column_name].tolist()
    embeddings = np.vstack(embeddings_df['embedding'].apply(
        lambda x: np.frombuffer(x, dtype=np.float32)))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    _, indices = index.search(query_embedding.reshape(1, -1), top_n)
    top_results = [id_list[idx] for idx in indices[0]]
    final_results = [convert_to_dict(item) for item in top_results]

    return final_results


book_map = {
    'abudawud': 'Sunan Abu Dawud',
    'bukhari': 'Sahih al Bukhari',
    'ibnmajah': 'Sunan Ibn Majah',
    'nasai': 'Sunan an Nasai',
    'tirmidhi': 'Jami At Tirmidhi',
    'muslim': 'Sahih Muslim',
    'malik': 'Muwatta Malik',
    'abudawud1': 'Sunan Abu Dawud',
    'bukhari1': 'Sahih al Bukhari',
    'ibnmajah1': 'Sunan Ibn Majah',
    'nasai1': 'Sunan an Nasai',
    'tirmidhi1': 'Jami At Tirmidhi',
    'muslim1': 'Sahih Muslim',
    'malik1': 'Muwatta Malik'
}


def parse_identifier(identifier):
    parts = identifier.split('_')
    book_key = parts[1]
    number = int(parts[2])
    book_name = book_map[book_key]
    return (book_name, 'eng', number)
# Function to calculate search and rank results
# Function to split each item and convert it into a dictionary


def convert_to_dict(item):
    parts = item.rsplit('_', 1)
    return {'bookcode': parts[0], 'arabicnumber': int(parts[1])}

# Function to calculate search and rank results


def search_and_rank_results_original(query, model, language_code, db_conn_embedding, db_conn_hadith, top_n=10):
    query_embedding = generate_single_embedding(query, model)
    # Normalize for cosine similarity
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Fetching embeddings using pandas
    sql_query = text(
        "SELECT id, embedding FROM embeddings WHERE language_code = :language_code;")
    params = {'language_code': language_code}
    embeddings_df = pd.read_sql(
        sql_query, db_conn_embedding.connect(), params=params)

    if embeddings_df.empty:
        return []

    # Preparing data for FAISS
    id_list = embeddings_df['id'].tolist()
    embeddings = np.vstack(embeddings_df['embedding'].apply(
        lambda x: np.frombuffer(x, dtype=np.float32)))
    # Normalize for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]

    # Creating a FAISS index
    dimension = embeddings.shape[1]
    # IndexFlatIP is for inner product (cosine similarity)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Searching the index
    distances, indices = index.search(query_embedding.reshape(1, -1), top_n)
    results_with_scores = [{'id': id_list[idx], 'score': float(distances[0][i]), 'algorithm': 'embedding model'}
                           for i, idx in enumerate(indices[0])]
    # top_results = [(id_list[idx], float(distances[0][i]))
    #                for i, idx in enumerate(indices[0])]

    # # Fetching original texts using pandas
    # conditions = [
    #     f"(bookname = '{book_map[identifier.split('_')[1]]}' AND language = '{language_code}' AND arabicnumber = {int(identifier.split('_')[2])})" for identifier in top_results]
    # where_clause = " OR ".join(conditions)
    # print(where_clause)

    # hadith_query = text(f"SELECT * FROM hadith WHERE {where_clause}")
    # final_result_df = pd.read_sql(hadith_query, db_conn_hadith)

    # order_map = {v: i for i, v in enumerate(top_results)}
    # print(order_map)
    # final_result_df['order'] = final_result_df.apply(lambda row: order_map.get(
    #     f"{row['language']}_{row['bookname']}_{row['arabicnumber']}"), axis=1)
    # final_result_df = final_result_df.sort_values(
    #     'order').drop('order', axis=1)
    # print(final_result_df)

    return results_with_scores


def search_documents(tfidf_matrix, vectorizer, document_ids, query, top_n=50):
    """
    Search for documents similar to the provided query using TF-IDF.

    Parameters:
    tfidf_matrix: The TF-IDF matrix of the document corpus.
    vectorizer: The TF-IDF vectorizer used to transform the query.
    document_ids: Array or list of document identifiers.
    query: The search query as a string.
    top_n: The number of top documents to return.

    Returns:
    Array of top document identifiers based on cosine similarity.
    """
    # Transform the query to its TF-IDF vector
    query_vector_tfidf = vectorizer.transform([query])

    # Compute cosine similarity between the query vector and TF-IDF vectors
    similarity_scores = cosine_similarity(
        query_vector_tfidf, tfidf_matrix).flatten()

    # Get the indices of the top_n most similar documents
    top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

    # Retrieve the original document IDs using the indices
    top_document_ids = document_ids[top_indices]

    return top_document_ids


def normalize_scores(scores_list, algorithm, weight=1.0):
    max_score = max(scores_list, key=lambda x: x['score'])['score']
    normalized_scores = [{'id': doc['id'], 'score': doc['score'] / max_score, 'algorithm': algorithm}
                         for doc in scores_list]
    for doc in normalized_scores:
        if algorithm == 'tfidf model':
            doc['score'] *= weight  # Apply the weight to TF-IDF scores
        elif algorithm == 'embedding model':
            # Apply the weight to Embedding scores
            doc['score'] *= (1 - weight)
    return normalized_scores


# def fetch_additional_data(row, db_conn_hadith):
#     # Construct the SQL query for the specific row
#     query = f"""
#     SELECT hadithnumber, text, grades, bookNumber, bookhadith, shortname
#     FROM hadith
#     WHERE language = '{row['language']}' AND
#           bookname = '{row['bookname']}' AND
#           arabicnumber = {row['arabicnumber']}
#     """
#     # Execute the query
#     result = pd.read_sql_query(query, db_conn_hadith)
#     return result


def search_and_rank_hybrid(query, model, language_code, db_conn_embedding, vectorizer, tfidf_matrix, document_ids, db_conn_hadith, top_n=1000, tfidf_weight=2):
    # Perform TF-IDF search
    tfidf_top_document_ids = search_documents(
        tfidf_matrix, vectorizer, document_ids, query, top_n)
    # Higher score for top ranks
    tfidf_scores = dict(zip(tfidf_top_document_ids, range(top_n, 0, -1)))
    tfidf_scores_list = [{'id': key, 'score': value, 'algorithm': 'tfidf model'}
                         for key, value in tfidf_scores.items()]

    # Perform embeddings search
    embeddings_results = search_and_rank_results_original(
        query, model, language_code, db_conn_embedding, top_n)

    normalized_tfidf_scores_list = normalize_scores(
        tfidf_scores_list, 'tfidf model', tfidf_weight)
    normalized_embeddings_results = normalize_scores(
        embeddings_results, 'embedding model', 1.0 - tfidf_weight)

    combined_results = normalized_embeddings_results+normalized_tfidf_scores_list
    combined_results_df = pd.DataFrame(combined_results)
    combined_results_df = combined_results_df.sort_values(
        by='score', ascending=False)

    combined_results_df['id'] = combined_results_df['id'].str.replace('-', '_')
    id_parts = combined_results_df['id'].str.split(
        '_').tolist()  # Convert Series to a list

    # Create new columns for 'language,' 'bookname,' and 'arabicnumber'
    combined_results_df['language'] = [parts[0] for parts in id_parts]
    combined_results_df['bookname'] = [book_map[parts[1]]
                                       for parts in id_parts]
    combined_results_df['arabicnumber'] = [parts[2] for parts in id_parts]
    combined_results_df['bookname'] = combined_results_df['bookname'].str.strip()
    combined_results_df['language'] = combined_results_df['language'].str.strip()
    combined_results_df['arabicnumber'] = combined_results_df['arabicnumber'].str.strip()

    combined_results_df = combined_results_df[combined_results_df['language'] == language_code]
    # combined_results_df.to_csv("combined_results.df", index=False)
    # Create a list of SQL queries, one for each row in combined_results_df
    combined_results_df

    queries = []
    for index, row in combined_results_df.iterrows():
        query = f"SELECT  hadithnumber,bookname,arabicnumber,text,grades,bookNumber,bookhadith, {row['score']} AS score, language FROM hadith WHERE language = '{row['language']}' AND bookname = '{row['bookname']}' AND arabicnumber = {row['arabicnumber']}"
        queries.append(query)

    # Join the queries with UNION ALL to combine the results
    final_query = " UNION ALL ".join(queries)

    # print(final_query)

    final_results_df = pd.read_sql(final_query, db_conn_hadith.connect())

    # # Convert 'arabicnumber' to string in combined_results_df
    # combined_results_df['arabicnumber'] = combined_results_df['arabicnumber'].astype(
    #     str)
    # # Or, convert 'arabicnumber' to string in final_results_df
    # final_results_df['arabicnumber'] = final_results_df['arabicnumber'].astype(
    #     str)
    # final_joined_df = pd.merge(combined_results_df, final_results_df,
    #                            on=['language', 'bookname', 'arabicnumber'],
    #                            how='left')
    # final_joined_df.drop('score_y', axis=1, inplace=True)
    # # Optionally, rename 'score_x' back to 'score'
    # final_joined_df.rename(columns={'score_x': 'score'}, inplace=True)

    return final_results_df
