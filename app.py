from flask import Flask, request, jsonify, Response
from dotenv import load_dotenv
import traceback
from functools import wraps
import gc
import os
import json
import psutil
from flask_restx import Api, Resource, fields


import logging

# SQL Alchemy
from sqlalchemy import text
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func

# Data Manipulation
import pandas as pd
import time
from utils import search_and_rank_hybrid, search_and_rank_results_original
from translator import translate_eng
# ML Model
from sentence_transformers import SentenceTransformer

import pickle

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

#####################################################
#                INITIALIZATION                     #
#####################################################
# Loading the database and ML Model
connection_string_embedding = 'sqlite:///databases/hadith_embeddings.db'
connection_string_metadata = 'sqlite:///databases/hadith_search_full.db'

engine_embedding = create_engine(connection_string_embedding)
engine_metadata = create_engine(connection_string_metadata)
# Base path
base_dir = "tfidf"

# Join the paths
tfidf_matrix_path = os.path.join(base_dir, "tfidf_matrix.pkl")
vectorizer_path = os.path.join(base_dir, "tfidf_vectorizer.pkl")
document_ids_path = os.path.join(base_dir, "document_ids.pkl")

# Load the TF-IDF matrix and indices
with open(tfidf_matrix_path, 'rb') as f:
    tfidf_matrix, document_ids = pickle.load(f)  # Unpack the saved tuple

# Load the vectorizer
with open(vectorizer_path, 'rb') as f:
    vectorizer = pickle.load(f)

# Load document_ids
with open(document_ids_path, 'rb') as f:
    document_id = pickle.load(f)
model = None


def get_model():
    global model
    if model is None:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        gc.collect()  # Collect garbage after loading the model
    return model


language_code_map = {
    'ara': 'ar',  # Arabic
    'ben': 'bn',  # Bengali
    'fra': 'fr',  # French
    'ind': 'id',  # Indonesian (Bahasa Indonesia)
    'urd': 'ur',  # Urdu
    'tur': 'tr',  # Turkish
    'rus': 'ru',  # Russian
    'tam': 'ta',  # Tamil
    'eng': 'en'   # English
}
authorizations = {
    'apiKey': {
        'type': 'apiKey',
        'in': 'header',
        'name': 'X-API-KEY'
    }
}


#####################################################
#                FLASK APP INITALIZAION             #
#####################################################
app = Flask(__name__)

# Initialize API with Flask-RESTPlus
api = Api(app,
          version='1.0',
          title='Hadith Search API',
          description='A simple API for searching Hadiths',
          authorizations=authorizations,
          security='apiKey'
          )
# Define a namespace
ns = api.namespace('search', description='Search operations')

# Define the model for request parsing (if needed)
search_model = api.model('Search', {
    'query': fields.String(required=True, description='The search query'),
    'language_code': fields.String(required=True, description='Language code for the query')
})


def require_api_key(func):
    @wraps(func)
    def check_api_key(*args, **kwargs):
        try:
            # Ensure you have set this environment variable
            expected_api_key = os.getenv('API_KEY')
            received_api_key = request.headers.get('X-API-KEY')
            logger.info(
                f"Expected API Key: {expected_api_key}, Received API Key: {received_api_key}")

            if not expected_api_key or expected_api_key != received_api_key:
                logger.error(
                    "Unauthorized access attempt with API key: " + str(received_api_key))
                return {"error": "Unauthorized"}, 401

            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"An error occurred in API key check: {str(e)}")
            return {"error": "Internal Server Error"}, 500

    return check_api_key


@ns.route('/')
class Search(Resource):
    @api.doc(security='apiKey', params={
        'query': 'The search query',
        'language_code': 'Language code for the query'
    })
    @require_api_key
    def get(self):
        """
        Search Hadiths
        """
        try:
            model = get_model()
            query = request.args.get('query', None)
            language_code = request.args.get('language_code', None)
            logger.info(
                f'Received query: {query}, language code: {language_code}')
            english_translation = translate_eng(
                query, language_code_map[language_code])
            if query is None:
                logger.error('No query provided')
                return jsonify({"error": "No query provided"}), 400
            t1 = time.time()
            results = search_and_rank_hybrid(
                query=english_translation[0]['translations'][0]['text'],
                model=model, language_code=language_code,
                db_conn_hadith=engine_metadata,
                db_conn_embedding=engine_embedding,
                vectorizer=vectorizer,
                tfidf_matrix=tfidf_matrix,
                document_ids=document_ids,
                top_n=20,
                tfidf_weight=0.7)
            gc.collect()
            t2 = time.time()
            logger.info(f'Search completed in {t2-t1} seconds')
            # Check if results are found
            if results is None:
                return jsonify({"error": "No results found"}), 404
            # Computing results
            results_dict = results.to_dict(orient='records')
            response_data = json.dumps(results_dict, indent=4)
            return Response(response_data, mimetype='application/json', status=200)

        except Exception as e:
            print(e)
            logger.error(traceback.print_exc())
            logger.exception('An error occurred during the search process')
            # Log the exception, e.g., print(e) or log to a file
            return jsonify({"error": "An error occurred processing your request"}), 500


sys_ns = api.namespace('system', description='System operations')


@sys_ns.route('/app_memory_usage')
class AppMemoryUsage(Resource):
    def get(self):
        """
        Get Application Memory Usage
        This endpoint returns the memory usage of the application in megabytes.
        """
        gc.collect()
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_usage_mb = memory_info.rss / \
            (1024 ** 2)  # Convert from bytes to MB
        return jsonify({
            'memory_usage_MB': memory_usage_mb
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
