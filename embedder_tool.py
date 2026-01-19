import numpy as np
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

class ProjectEmbedder:
    """
    PDF Requirement: 
    - 2 Original English SBERT models
    - 2 Multilingual SBERT models
    - Word2Vec + TF-IDF (First term approach)
    """
    def __init__(self):
        print("🔧 Starting Embedder Tool...")
        
        # PDF REQUIREMENT: 4 SBERT Model
        print("  📥 SBERT Models are loading...")
        self.sbert_models = {
            # Original English Models
            "en_orig_1": SentenceTransformer('all-MiniLM-L6-v2'),      # 384 dim
            "en_orig_2": SentenceTransformer('all-mpnet-base-v2'),     # 768 dim
            
            # Multilingual Models
            "multi_1": SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2'),  # 384 dim
            "multi_2": SentenceTransformer('distiluse-base-multilingual-cased-v1')    # 512 dim
        }
        
        print("  ✅ SBERT Modelleri hazır!")
        print(f"     - en_orig_1: {self.sbert_models['en_orig_1'].get_sentence_embedding_dimension()} dim")
        print(f"     - en_orig_2: {self.sbert_models['en_orig_2'].get_sentence_embedding_dimension()} dim")
        print(f"     - multi_1: {self.sbert_models['multi_1'].get_sentence_embedding_dimension()} dim")
        print(f"     - multi_2: {self.sbert_models['multi_2'].get_sentence_embedding_dimension()} dim")
        
        # Word2Vec (First Term Approach)
        self.w2v_model = None
        self.vector_size = 100  # Increased for better representation
        
        # TF-IDF Vectorizer
        self.tfidf_vectorizer = None

    # ========================================================================
    # SBERT METHODS (Advanced NLP - PDF Requirement)
    # ========================================================================
    
    def get_sbert_vector(self, text, model_key="multi_1"):
        """
        SBERT kullanarak cümle vektörü üretir.
        
       Args:
            text (str): Menni Commune
            model_key (str): Hangi SBERT modeling
                -in_orig_1: MiniLM-L6-v2 (384 dim)
                -in_orig_2: all-studd-v2 (768 dim)
                -multi_1: paraphrace-MiniLM-L12-v2 (384 dim)
                -milti_2: distilled-multiling-vulist-v1 (512 dim)
        
        Returns:
            np.array: Sentence embedding vector
        """
        if model_key not in self.sbert_models:
            raise ValueError(f"Invalid model_key: {model_key}")
        
        return self.sbert_models[model_key].encode([text])[0]
    
    def get_all_sbert_embeddings(self, texts):
        """
        Returns embeddings for all BERT models. Used for model comparison.
        Args:
            texts (list): List of command texts
        
        Returns:
           dict: {model_key: embeddings_array}
        """
        embeddings = {}
        for key, model in self.sbert_models.items():
            print(f"  📊 {key} It is produced with embedding...")
            embeddings[key] = model.encode(texts)
        return embeddings

    # ========================================================================
    # WORD2VEC METHODS (First Term Approach - PDF Requirement)
    # ========================================================================
    
    def train_w2v(self, sentences, vector_size=100, window=5, min_count=1):
        """
       Trains the Word2Vec model.
        
        Args:
            sentences (list): Tokenized sentences [[word1, word2], ...]
            vector_size (int): Word vector size
            window (int): Context window size
            min_count (int): Minimum word frequency
        """
        print(f"  🧠Word2Vec is being trained(dim={vector_size})...")
        self.vector_size = vector_size
        self.w2v_model = Word2Vec(
            sentences, 
            vector_size=vector_size, 
            window=window, 
            min_count=min_count, 
            workers=4,
            seed=42
        )
        print("  ✅ Word2Vec eğitimi tamamlandı!")

    def get_word_mean_vector(self, text):
        """
      Takes the simple average of word vectors.
        
        Args:
            text (str): Command text
        
        Returns:
            np.array: Mean word vector
        """
        if not self.w2v_model:
            raise ValueError("The Word2Vec model has not been trained yet! Call train_w2v().")
        
        words = text.lower().split()
        vectors = [self.w2v_model.wv[w] for w in words if w in self.w2v_model.wv]
        
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(self.vector_size)
    
    def get_tfidf_weighted_vector(self, text, corpus=None):
        """
      TF-IDF weighted word vector (improved approach mentioned in the PDF).
        
        Args:
            text (str): Command text
            corpus (list): All documents for TF-IDF (required on first call)
        
        Returns:
            np.array: TF-IDF weighted word vector
        """
        if not self.w2v_model:
            raise ValueError("The Word2Vec model has not been trained yet!")
        
        # TF-IDF vectorizer'ı ilk seferde eğit
        if corpus is not None and self.tfidf_vectorizer is None:
            print("  📊 TF-IDF Vectorizer is being trained...")
            self.tfidf_vectorizer = TfidfVectorizer()
            self.tfidf_vectorizer.fit(corpus)
        
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer is not trained! corpus parameter required.")
        
        # TF-IDF skorlarını al
        tfidf_matrix = self.tfidf_vectorizer.transform([text])
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # TF-IDF skorları ile word vectorleri ağırlıklandır
        weighted_sum = np.zeros(self.vector_size)
        total_weight = 0.0
        
        for i, word in enumerate(feature_names):
            if word in self.w2v_model.wv and tfidf_matrix[0, i] > 0:
                weight = tfidf_matrix[0, i]
                weighted_sum += self.w2v_model.wv[word] * weight
                total_weight += weight
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return np.zeros(self.vector_size)

    # ========================================================================
    # COMPARISON UTILITIES
    # ========================================================================
    
    def compare_embeddings(self, text, include_w2v=True):
        """
        Compares all embedding methods for the same text.
        
        Args:
            text (str): Test text
            include_w2v (bool): Include Word2Vec?
        
        Returns:
            dict: {method_name: (vector, dimension)}
        """
        results = {}
        
        # SBERT modelleri
        for key in self.sbert_models.keys():
            vec = self.get_sbert_vector(text, model_key=key)
            results[f"SBERT_{key}"] = (vec, len(vec))
        
        # Word2Vec (eğer eğitilmişse)
        if include_w2v and self.w2v_model:
            vec_mean = self.get_word_mean_vector(text)
            results["Word2Vec_Mean"] = (vec_mean, len(vec_mean))
            
            if self.tfidf_vectorizer:
                vec_tfidf = self.get_tfidf_weighted_vector(text)
                results["Word2Vec_TF-IDF"] = (vec_tfidf, len(vec_tfidf))
        
        return results

# ============================================================================
# TEST & DEMO
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 EMBEDDER TOOL TEST")
    print("="*60)
    
    # Initialize
    embedder = ProjectEmbedder()
    
    # Test SBERT
    print("\n1️⃣ SBERT Test:")
    test_text = "ışığı aç"
    for key in ["en_orig_1", "en_orig_2", "multi_1", "multi_2"]:
        vec = embedder.get_sbert_vector(test_text, model_key=key)
        print(f"   {key}: {vec.shape} - Sample: {vec[:3]}")
    
    # Test Word2Vec
    print("\n2️⃣ Word2Vec Test:")
    sample_commands_tr = [
        "ışığı aç", "ışığı kapa", "klimayı aç", "klimayı kapa",
        "fanı arttır", "sıcaklığı düşür", "TV aç", "müzik kapa"
    ]
    
    sentences = [cmd.lower().split() for cmd in sample_commands_tr]
    embedder.train_w2v(sentences, vector_size=100)
    
    vec_mean = embedder.get_word_mean_vector("ışığı aç")
    print(f"   Mean Vector: {vec_mean.shape}")
    
    # TF-IDF weighted
    embedder.get_tfidf_weighted_vector("ışığı aç", corpus=sample_commands_tr)
    vec_tfidf = embedder.get_tfidf_weighted_vector("ışığı aç")
    print(f"   TF-IDF Weighted: {vec_tfidf.shape}")
    
    # Comparison
    print("\n3️⃣ Full Comparison:")
    comparison = embedder.compare_embeddings("ışığı aç")
    for method, (vec, dim) in comparison.items():
        print(f"   {method:25} -> Dim: {dim}")
    
    print("\n" + "="*60)
    print("✅ TEST TAMAMLANDI!")
    print("="*60)