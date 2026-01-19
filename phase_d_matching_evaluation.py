import json
import numpy as np
import pickle
import gensim
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
import re
from collections import Counter

# ---SETTINGS ---
JSON_FILE = 'smart_home_corpus.json'
MODEL_FILE = 'trmodel_finetuned'
VECTOR_FILE = 'target_vectors_mean.pkl'

print("="*70)
print("PHASE D: HYBRID MATCHING EVALUATION")
print("="*70)

#1. Downloads
print("---Loading Data ---")
model = Word2Vec.load(MODEL_FILE)
with open(VECTOR_FILE, 'rb') as f:
    target_vectors = pickle.load(f)

with open(JSON_FILE, 'r', encoding='utf-8') as f:
    corpus_data = json.load(f)

def tokenize(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence.split()

# 2. KEYWORD EXTRACTION (Extract Critical Words for Each Intent)
# We find the most common words for each intent.
intent_keywords = {}

for item in corpus_data:
    intent = item['target_command']
    all_words = []
    for var in item['variations']:
        all_words.extend(tokenize(var))
    
# Most frequently occurring words (Stop words can be excluded, but let's keep it simple for now)
    # Word frequency analysis
    counts = Counter(all_words)
# Consider words that appear at least twice in that intent as "keywords"
    # or if the number of variations is small, take them all
    keywords = {word for word, count in counts.items()}
    intent_keywords[intent] = keywords

print(f"✓ {len(intent_keywords)} Keywords mapped for intent.")

# 3. AUXILIARY FUNCTIONS
def get_sentence_vector(sentence, model):
    tokens = tokenize(sentence)
    word_vectors = [model.wv[token] for token in tokens if token in model.wv]
    if not word_vectors: return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def calculate_keyword_score(input_tokens, target_intent_keywords):
    """
How many of the words in the input sentence are in the word pool of the target intent?
    """
    if not input_tokens: return 0.0
    
    match_count = sum(1 for t in input_tokens if t in target_intent_keywords)
    
# Simple ratio: Matching Word /Total Input Word
    return match_count / len(input_tokens)

#4. HYBRID MATCHING FUNCTION
def match_command_hybrid(input_sentence, threshold=0.60):
    tokens = tokenize(input_sentence)
    
#1. OOV Control (Word based)
    known_tokens = [t for t in tokens if t in model.wv]
    if not known_tokens:
        return {'matched': False, 'candidate': None, 'score': 0.0, 'reason': 'All OOV', 'vec_score': 0.0}

#2. Vector Similarity
    input_vec = get_sentence_vector(input_sentence, model)
    vector_scores = {}
    for intent, target_vec in target_vectors.items():
        sim = cosine_similarity([input_vec], [target_vec])[0][0]
        vector_scores[intent] = sim
        
# Recruit the top 3 candidates
    top_candidates = sorted(vector_scores, key=vector_scores.get, reverse=True)[:3]
    
    final_scores = {}
    
    for intent in top_candidates:
        vec_sim = vector_scores[intent]
        target_keywords = intent_keywords[intent]
# ---CRITICAL UPDATE: KEYWORD SCORING ---
# A) How many of the words in the sentence are in the target intent? (Coverage)
        matched_tokens = [t for t in tokens if t in target_keywords]
        coverage_score = len(matched_tokens) / len(tokens) if tokens else 0
        
# B) Alien Penalty
        # If the word is not in the intent, seriously reduce the score
        alien_word_count = len(tokens) - len(matched_tokens)
        penalty = 0.0
        if alien_word_count > 0:
          # Deduct 15% points for each foreign word (Sell the TV -> sell foreigner -> penalty)
            penalty = alien_word_count * 0.15
            
       #C) New Weighted Formula
        # Vector: 60%, Keyword Coverage: 40%
        base_score = (vec_sim * 0.6) + (coverage_score * 0.4)
        
      # Apply the punishment
        final_score = base_score - penalty
        
       # Block negative score
        final_scores[intent] = max(0.0, final_score)

    best_candidate = max(final_scores, key=final_scores.get)
    best_score = final_scores[best_candidate]
# Special Rule: If 'coverage_score' is 0 (no common words)
    # Dynamically raise threshold no matter how high the vector is
    # Ex: he said "red" but the target is "blue" (the vectors are close but the word doesn't fit)
    is_keyword_match = any(t in intent_keywords[best_candidate] for t in tokens)
    
# If no keywords are found, increase the threshold to 0.80 (Reject if not too sure)
    effective_threshold = threshold if is_keyword_match else 0.80
    
    matched = best_score >= effective_threshold
    
    return {
        'matched': matched,
        'candidate': best_candidate,
        'score': best_score,
        'vec_score': vector_scores[best_candidate],
        'reason': 'Success' if matched else 'Low Score'
    }

# 5. TEST DATA PREPARATION (same logic as Phase D)
test_dataset = []
#Positives
for item in corpus_data:
    intent = item['target_command']
# Her intenttain 3 variations go
    for var in item['variations'][:3]:
        test_dataset.append({'input': var, 'expected': intent, 'is_valid': True})

# Negatives (Challenging examples + FP creating ones)
negatives = [
    "araba çalıştır", "buzdolabını aç", "kapıyı kilitle", "pencereyi sil",
    "fırını yak", "televizyonu sat", "müzik indir", "yemek yap", "kedi maması ver",
    "hava durumu nasıl" # This might actually interfere with GET_TEMP, let's see
]
for neg in negatives:
    test_dataset.append({'input': neg, 'expected': None, 'is_valid': False})

# Extra Challenging "Missed" Examples (Taken from logs)
missed_cases = [
    {'input': 'aydınlatma sat', 'expected': 'LIGHT_ON', 'is_valid': True},
    {'input': 'az ışık', 'expected': 'LIGHT_DOWN', 'is_valid': True},
    {'input': 'lamba aç yak', 'expected': 'LIGHT_ON', 'is_valid': True},
    {'input': 'aydınlatmayı başlat', 'expected': 'LIGHT_ON', 'is_valid': True}
]
test_dataset.extend(missed_cases)

print(f"Total Test Data : {len(test_dataset)}")

# 6. TEST RUN
def run_test_hybrid(threshold=0.60):
    print(f"\n--- HYBRID TEST (Threshold: {threshold}) ---")
    
    y_true = []
    y_pred = []
    correct_intent = 0
    valid_count = 0
    
    false_positives = []
    false_negatives = []
    
    for sample in test_dataset:
        text = sample['input']
        expected = sample['expected']
        is_valid = sample['is_valid']
        
        res = match_command_hybrid(text, threshold)
        
        y_true.append(1 if is_valid else 0)
        y_pred.append(1 if res['matched'] else 0)
        
        if is_valid:
            valid_count += 1
            if res['matched'] and res['candidate'] == expected:
                correct_intent += 1
            elif res['matched'] and res['candidate'] != expected:
                false_negatives.append(f"WRONG: '{text}' -> Got '{res['candidate']}' (Sc: {res['score']:.3f}, Vec: {res['vec_score']:.3f}) Exp: {expected}")
            else:
                false_negatives.append(f"MISSED: '{text}' (Sc: {res['score']:.3f}, Vec: {res['vec_score']:.3f}) Best: {res['candidate']}")
        else:
            if res['matched']:
                false_positives.append(f"FP: '{text}' -> '{res['candidate']}' (Sc: {res['score']:.3f}, Vec: {res['vec_score']:.3f})")

    f1 = f1_score(y_true, y_pred)
    acc = correct_intent / valid_count if valid_count > 0 else 0
    
    print(f"F1-Score: {f1:.3f}")
    print(f"Intent Accuracy: {acc:.3f}")
    
    print("\nFalse Positives :")
    for fp in false_positives[:5]: print("  " + fp)
        
    print("\nFalse Negatives:")
    for fn in false_negatives[:5]: print("  " + fn)

# Run the test (we can lower Threshold a bit because hybrid scores are a bit more conservative)
run_test_hybrid(0.55) 
run_test_hybrid(0.59)