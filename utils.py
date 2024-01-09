from sentence_transformers import SentenceTransformer
import re
import faiss
import numpy as np
import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text

# Function to remove bidirectional characters from text


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
    _, indices = index.search(query_embedding.reshape(1, -1), top_n)
    top_results = [id_list[idx] for idx in indices[0]]
    top_results = [element.replace('-', '_') for element in top_results]
    print(top_results)
    # Fetching original texts using pandas
    conditions = [
        f"(bookname = '{book_map[identifier.split('_')[1]]}' AND language = '{language_code}' AND arabicnumber = {int(identifier.split('_')[2])})" for identifier in top_results]
    where_clause = " OR ".join(conditions)
    print(where_clause)

    hadith_query = text(f"SELECT * FROM hadith WHERE {where_clause}")
    final_result_df = pd.read_sql(hadith_query, db_conn_hadith)

    order_map = {v: i for i, v in enumerate(top_results)}
    print(order_map)
    final_result_df['order'] = final_result_df.apply(lambda row: order_map.get(
        f"{row['language']}_{row['bookname']}_{row['arabicnumber']}"), axis=1)
    final_result_df = final_result_df.sort_values(
        'order').drop('order', axis=1)
    print(final_result_df)

    return final_result_df


def search_and_rank_results_full(query, model, language_code, db_conn_embedding, db_conn_hadith, top_n=20):
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
    print(top_results)

    conditions = [
        f"(bookname = '{book_map[identifier.split('_')[1]]}' AND language = '{language_code}' AND arabicnumber = {int(identifier.split('_')[2])})" for identifier in top_results]
    where_clause = " OR ".join(conditions)

    hadith_query = text(f"SELECT * FROM hadith WHERE {where_clause}")
    final_result_df = pd.read_sql(hadith_query, db_conn_hadith)

    order_map = {v: i for i, v in enumerate(top_results)}
    final_result_df['order'] = final_result_df.apply(lambda row: order_map.get(
        f"{row['language']}_{row['bookname']}_{row['arabicnumber']}"), axis=1)
    final_result_df = final_result_df.sort_values(
        'order').drop('order', axis=1)

    return final_result_df
