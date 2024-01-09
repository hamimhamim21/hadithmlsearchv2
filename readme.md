# Hadith Search API

### Overview

This project implements a Hadith Search API using Flask, providing an interface for querying Hadith texts. It utilizes a machine learning model (Sentence Transformers) for search and ranking, offering results in multiple languages. The API is secured with an API key authorization.

### Features

<ul>
   <li>Search Functionality: Allows users to search for Hadiths using a query in various languages.</li>
   <li>Machine Learning Integration: Employs a Sentence Transformer model for effective search and ranking.</li>
   <li>Multi-language Support: Translates queries and returns results in multiple languages including Arabic, Bengali, French, Indonesian, Urdu, Turkish, Russian, Tamil, and English.</li>
   <li>API Key Authorization: Ensures secure access to the API.
   Memory Usage Monitoring: Endpoint for monitoring the application's memory usage.</li>
</ul>

### Instructions

Follow these steps to set up and run the project:

1. **Create a .env File:**
   Create a `.env` file in the project root and add the following line to load the path to `TRANSLATOR_KEY` and `END_POINT` which is translator API from Azure Translate API.

```
TRANSLATOR_KEY=<TRANSLATOR_API_KEY>
END_POINT=https://api.cognitive.microsofttranslator.com/translate
API_KEY=<YOUR_CUSTOM_API_KEY>

```

2. **Database: SQLite**
   Contact the repo owner for the database
   `hadith_search_full.db` and `hadith_embeddings`

3. **Install Dependencies:**
   Run the following command to install the required dependencies:

```
    pip install -r requirements.txt
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cpu

```

4. **Run the Script:**
   Execute the script with the following command:

```
 python app.py

```

# Language Codes

Below is a table of language codes and their respective languages:

| Code | Language            |
| ---- | ------------------- |
| ara  | Arabic              |
| ben  | Bengali             |
| fra  | French              |
| ind  | Indonesian (Bahasa) |
| urd  | Urdu                |
| tur  | Turkish             |
| rus  | Russian             |
| tam  | Tamil               |
| eng  | English             |

#### Example usage browser

http://localhost:5000/search?query=how to pray Fajr&language_code=eng

Add the following to header call for the API

```
X-API-KEY:<YOUR_CUSTOM_API_KEY>

```

#### Example response

```
[
  {
    "hadithnumber": 500,
    "arabicnumber": 500,
    "text": "Abu Mahdhurah reported; I said; Messenger of Allah, teach me the method of ADHAN (how to pronounce the call to prayer). He wiped my forehead (with his hand) and asked me to pronounce; Allah is most great. Allah is most great. Allah is most great. Allah is most great, raising your voice while saying them (these words). Then you must raise your voice in making the testimony:I testify that there is no god but Allah, I testify that there is no god but Allah; I testify that Muhammad is the Messenger of Allah, I testify that Muhammad is the Messenger of Allah. Lowering your voice while saying them (these words). Then you must raise your voice in making the testimony: I testify that there is no god but Allah, I testify there is no god but Allah; I testify Muhammad is the Messenger of Allah, I testify Muhammad is the Messenger of Allah. Come to prayer, come to prayer; come to salvation, come to salvation. If it is the morning prayer, you must pronounce; prayer is better than sleep, prayer is better than sleep, Allah is most great; there is no god but Allah",
    "grades": "Al-Albani::Sahih && Muhammad Muhyi Al-Din Abdul Hamid::Sahih && Shuaib Al Arnaut::Daif && Zubair Ali Zai::Daif",
    "bookNumber": 2,
    "bookhadith": 110,
    "bookname": "Sunan Abu Dawud",
    "language": "eng",
    "shortname": "abudawud"
  }
]
```
