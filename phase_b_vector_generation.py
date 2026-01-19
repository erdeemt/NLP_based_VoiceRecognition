import json
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pickle

print("="*60)
print("PHASE B: OPTIMIZED VECTOR GENERATION")
print("="*60)

# 1. Load Model
print("---Loading Model ---")
try:
    model = Word2Vec.load('trmodel_finetuned')
    print("Model loaded successfully.")
except:
    print("ERROR: 'trmodel_finetuned' not found. Please run Phase A first.")
    exit()

#2. Load Optimized JSON Data
JSON_FILE = 'smart_home_corpus.json'
print(f"---Loading Data: {JSON_FILE} ---")

try:
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        corpus_data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: '{JSON_FILE}' not found. Please run dataset_optimized.py first.")
    exit()

#3. Tokenization
def tokenize(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence.split()

#4. TF-IDF Training (Across all variations)
all_sentences = []
for item in corpus_data:
    all_sentences.extend(item['variations'])

tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
tfidf_vectorizer.fit(all_sentences)
print(f"✓ TF-IDF trained ({len(tfidf_vectorizer.vocabulary_)} words)")

#5. Vectoring Functions
def get_sentence_vector_mean(sentence, model):
    tokens = tokenize(sentence)
    word_vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not word_vectors: return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def get_sentence_vector_tfidf(sentence, model, vectorizer):
    tokens = tokenize(sentence)
    word_vectors = []
    weights = []
    for token in tokens:
        if token in model.wv and token in vectorizer.vocabulary_:
            word_vectors.append(model.wv[token])
            weights.append(vectorizer.transform([sentence])[0, vectorizer.vocabulary_[token]])
    
    if not word_vectors: return np.zeros(model.vector_size)
    return np.average(word_vectors, axis=0, weights=np.array(weights))

#6. Creating Intent Vectors
# ATTENTION: Every 'target_command' is now an INTENT ID (eg: LIGHT_ON)
# For each Intent, we create the "Main Intent Vector" by averaging ALL its variations.

target_vectors_mean = {}
target_vectors_tfidf = {}

print("\n---Calculating Intent Vectors ---")
for item in corpus_data:
    intent_name = item['target_command'] # Ex: LIGHT_ON
    variations = item['variations']
    
# Let's take the average of the vectors of all sentences belonging to this intent
    # (This creates a more robust "Center" vector)
    
# Center for MEAN Method
    vectors_mean = [get_sentence_vector_mean(s, model) for s in variations]
    intent_center_mean = np.mean(vectors_mean, axis=0)
    target_vectors_mean[intent_name] = intent_center_mean
    
# Center for TF-IDF Method
    vectors_tfidf = [get_sentence_vector_tfidf(s, model, tfidf_vectorizer) for s in variations]
    intent_center_tfidf = np.mean(vectors_tfidf, axis=0)
    target_vectors_tfidf[intent_name] = intent_center_tfidf

print(f"✓ {len(target_vectors_mean)} Intent vectors have been created.")

#7. Saving
with open('target_vectors_mean.pkl', 'wb') as f:
    pickle.dump(target_vectors_mean, f)
    
with open('target_vectors_tfidf.pkl', 'wb') as f:
    pickle.dump(target_vectors_tfidf, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

print("\n✓ Updated files: target_vectors_mean.pkl, target_vectors_tfidf.pkl")
print("="*60)
print("NOW YOU CAN PROCEED THE PHASE C.")