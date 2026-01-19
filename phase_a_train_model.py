import json
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 1. Load JSON data
with open('smart_home_corpus.json', 'r', encoding='utf-8') as f:
    corpus_data = json.load(f)

#   2. Prepare training sentences and target commands
training_sentences = []
command_variations = {}

for item in corpus_data:
    target = item['target_command']
    variations = item['variations']
    command_variations[target] = variations
    for variation in variations:
        training_sentences.append(variation)

print(f"Total training sentences: {len(training_sentences)}")
print(f"Total target commands: {len(command_variations)}")
print("\nFirst 5 example sentences:")
for i, sent in enumerate(training_sentences[:5]):
    print(f"{i+1}. {sent}")

#3. Tokenize sentences
def tokenize(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r'[^\w\s]', '', sentence)
    return sentence.split()
tokenized_sentences = [tokenize(sent) for sent in training_sentences]
print(f"\nNumber of tokenized sentences: {len(tokenized_sentences)}")

#4. Load pre-trained model
print("\n---Loading Word2Vec Model ---")
pretrained_vectors = None

# Load pre-trained vectors
try:
    pretrained_vectors = KeyedVectors.load_word2vec_format('trmodel', binary=True)
    print("✓ Pre-trained vectors loaded")
    print(f" Word count: {len(pretrained_vectors)}")
    print(f" Vector size: {pretrained_vectors.vector_size}")
except Exception as e:
    print(f"Failed to load pre-trained model: {e}")
    print("No problem, we will train from scratch!")

#5. Create and train new model
print("\n---Model Training Begins ---")

if pretrained_vectors is not None:
    print("Creating a new model with pre-trained vectors...")
    # Create a new Word2Vec model
    base_model = Word2Vec(
        vector_size=pretrained_vectors.vector_size,
        window=5,
        min_count=1,
        workers=4,
        sg=0
    )
    # Create Vocabulary
    base_model.build_vocab(tokenized_sentences)
    
    # Copy pre-trained vectors
    total_vec = len(base_model.wv)
    intersect_count = 0
    for word in base_model.wv.index_to_key:
        if word in pretrained_vectors:
            base_model.wv[word] = pretrained_vectors[word]
            intersect_count += 1
    
    print(f"Pre-trained vectors used for {intersect_count}/{total_vec} words")
    
 # Train the model (transfer learning)
    base_model.train(
        tokenized_sentences,
        total_examples=len(tokenized_sentences),
        epochs=30
    )
    print("Model trained with transfer learning")
else:
    # Train from scratch
    print("A new model is being trained from scratch...")
    base_model = Word2Vec(
        sentences=tokenized_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        epochs=50,
        sg=0
    )
    print("Model trained from scratch")

print(f"\nFinal Model:")
print(f"Word count: {len(base_model.wv)}")
print(f"Vector size: {base_model.vector_size}")

#6. Save the model
base_model.save('trmodel_finetuned')
print("\n✓ The model is saved as 'trmodel_finetuned'.")

# Test 7: Word similarities
print("\n--- Model Test ---")
test_words = ['ışık', 'lamba', 'karanlık', 'aydınlatma', 'klima', 'tv']

for word in test_words:
    if word in base_model.wv:
        try:
            similar = base_model.wv.most_similar(word, topn=3)
            print(f"Words most similar to '{word}':")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.3f}")
        except:
            print(f"Unable to calculate similarity for '{word}'")
    else:
        print(f"\n'{word}' was not found in the model")

print("\n" + "="*50)
print("Step 1 COMPLETED! You can now proceed to Phase B.")
print("="*50)