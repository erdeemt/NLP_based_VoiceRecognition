import json
import numpy as np
import pickle
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import re

print("="*70)
print("PHASE C: IMPROVED COMMAND MATCHING")
print("="*70)

# ============================================================
# 0. TOKENIZATION
# ============================================================
def tokenize(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence.split()

# ============================================================
# 1.LOAD MODELS
# ============================================================
print("\n---Loading Models ---")

# Word2Vec
model = Word2Vec.load('trmodel_finetuned')
print(f"✓ Word2Vec modeli yüklendi")

# Vektörler
with open('target_vectors_mean.pkl', 'rb') as f:
    target_vectors_mean = pickle.load(f)
with open('target_vectors_tfidf.pkl', 'rb') as f:
    target_vectors_tfidf = pickle.load(f)

# TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
print("✓ All vectors loaded")

# ============================================================
# 2. VECTOR CALCULATION AIDS
# ============================================================
def get_sentence_vector_mean(sentence, model):
    tokens = tokenize(sentence)
    word_vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def get_sentence_vector_tfidf(sentence, model, tfidf_vectorizer):
    tokens = tokenize(sentence)
    word_vectors = []
    weights = []
    
    for token in tokens:
        if token in model.wv and token in tfidf_vectorizer.vocabulary_:
            word_vectors.append(model.wv[token])
            token_index = tfidf_vectorizer.vocabulary_[token]
            tfidf_matrix = tfidf_vectorizer.transform([sentence])
            tfidf_score = tfidf_matrix[0, token_index]
            weights.append(tfidf_score)
    
    if not word_vectors:
        return np.zeros(model.vector_size)
    
    weights = np.array(weights)
    if weights.sum() > 0:
        weights = weights / weights.sum()
    
    return np.average(word_vectors, axis=0, weights=weights)

# ============================================================
# 3.NEW MATCH COMMAND FUNCTION (OOV CONTROLLED)
# ============================================================
def match_command(input_sentence, target_vectors, model, tfidf_vectorizer=None, method='mean', threshold=0.75):
    """
Smart match: Checks OOV and returns the best candidate even if it is below threshold.
    """
    tokens = tokenize(input_sentence)
    
   #1. OOV (Unknown Word) Check
    known_tokens = [t for t in tokens if t in model.wv]
    
  # If there are no known words in the sentence
    if not known_tokens:
        return {
            'matched': False, 
            'candidate': None,     # No guesses
            'similarity': 0.0, 
            'reason': 'All OOV', 
            'all_scores': {}
        }
    
#2. Vector Calculation
    if method == 'mean':
        input_vector = get_sentence_vector_mean(input_sentence, model)
    else:
        if tfidf_vectorizer is None:
            raise ValueError("The tfidf_vectorizer parameter is required for the TF-IDF method!")
        input_vector = get_sentence_vector_tfidf(input_sentence, model, tfidf_vectorizer)
        
#3. Similarity Calculation
    similarities = {}
    for target_cmd, target_vec in target_vectors.items():
        sim = cosine_similarity([input_vector], [target_vec])[0][0]
        similarities[target_cmd] = sim
        
# Find the best candidate
    best_candidate = max(similarities, key=similarities.get)
    best_similarity = similarities[best_candidate]
    
# Threshold control
    matched = best_similarity >= threshold
    
    return {
        'matched': matched,
        'candidate': best_candidate, #<--IMPORTANT: Keep the candidate even if it doesn't match
        'similarity': best_similarity,
        'reason': 'Threshold Failed' if not matched else 'Success',
        'all_scores': similarities
    }

# ============================================================
# 4. INTERACTIVE TEST (CORRECTED)
# ============================================================

def interactive_test(threshold=0.65):
    print("\n" + "="*70)
    print("INTERACTIVE TEST MODE")
    print(f"Threshold: {threshold}")
    print("Type 'q' to exit")
    print("="*70)
    
    while True:
        user_input = input("\nEnter command: ").strip()
        
        if user_input.lower() == 'q':
            break
        
        if not user_input:
            continue
        
        print("-" * 50)
        
      # MEAN METHOD CALL (FIXED: model parameter added)
        result_mean = match_command(
            input_sentence=user_input, 
            target_vectors=target_vectors_mean, 
            model=model,          
            method='mean', 
            threshold=threshold
        )
       # TF-IDF METHOD CALL (FIXED)
        result_tfidf = match_command(
            input_sentence=user_input, 
            target_vectors=target_vectors_tfidf, 
            model=model,                
            tfidf_vectorizer=tfidf_vectorizer, 
            method='tfidf', 
            threshold=threshold
        )
        
    # PRINTING MEAN RESULT
        print(f"📊 MEAN METHOD:")
        if result_mean['matched']:
            print(f"  ✓ Matched: '{result_mean['candidate']}'") # 'candidate' instead of 'command'
            print(f" Similarity: {result_mean['similarity']:.3f}")
        else:
            reason = result_mean.get('reason', 'Unknown')
            cand = result_mean.get('candidate', 'Yok')
            sim = result_mean.get('similarity', 0.0)
            print(f"  ✗ Did not match ({reason})")
            print(f"   System prediction: '{cand}' (Score: {sim:.3f})") # Now we'll see what it predicts

        # PRINTING TF-IDF RESULT
        print(f"\n📊 TF-IDF METHOD:")
        if result_tfidf['matched']:
            print(f"  ✓ Matched: '{result_tfidf['candidate']}'")
            print(f"  Similarity: {result_tfidf['similarity']:.3f}")
        else:
            reason = result_tfidf.get('reason', 'Unknown')
            cand = result_tfidf.get('candidate', 'Yok')
            sim = result_tfidf.get('similarity', 0.0)
            print(f" Did not match ({reason})")
            print(f" System prediction: '{cand}' (Score: {sim:.3f})")

# Start Test
if __name__ == "__main__":
    interactive_test(threshold=0.65)