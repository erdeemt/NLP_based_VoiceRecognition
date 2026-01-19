# -*- coding: utf-8 -*-
"""
PDF REQUIREMENT: "You should fine tune the best performing model"

This script:
1. Selects the best SBERT model from Phase 1
2. Fine-tunes for Command classification
3. Creates new embeddings with fine-tuned model
4. Compares performance
"""
import os
from xml.parsers.expat import model
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
from datasets import Dataset
import torch

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_DIR = "results_phase1"
OUTPUT_DIR = "finetuned_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COMMAND_NAMES_TR = {1: "ışığı aç", 2: "ışığı kapa", 3: "ışığı kıs", 4: "parlaklığı arttır", 5: "parlaklığı azalt", 6: "aydınlatmayı arttır", 7: "aydınlatmayı azalt", 8: "kırmızı ışığı aç", 9: "kırmızı ışığı kapa", 10: "kırmızı ışığı arttır", 11: "kırmızı ışığı azalt", 12: "kırmızı ışığı kıs", 13: "mavi ışığı aç", 14: "mavi ışığı kapa", 15: "mavi ışığı arttır", 16: "mavi ışığı azalt", 17: "mavi ışığı kıs", 18: "yeşil ışığı aç", 19: "yeşil ışığı kapa", 20: "yeşil ışığı arttır", 21: "yeşil ışığı azalt", 22: "yeşil ışığı kıs", 23: "klimayı aç", 24: "klimayı kapa", 25: "iklimlendirmeyi aç", 26: "iklimlendirmeyi kapa", 27: "ısıtmayı aç", 28: "ısıtmayı kapa", 29: "ısıt", 30: "soğut", 31: "sıcaklığı arttır", 32: "sıcaklığı düşür", 33: "evi ısıt", 34: "evi soğut", 35: "odayı ısıt", 36: "odayı soğut", 37: "kombiyi aç", 38: "kombiyi kapa", 39: "fanı aç", 40: "fanı kapa", 41: "fanı arttır", 42: "fanı düşür", 43: "TV aç", 44: "TV kapa", 45: "televizyonu aç", 46: "televizyonu kapa", 47: "multimedyayı aç", 48: "multimedyayı kapa", 49: "müzik aç", 50: "müzik kapa", 51: "panjuru aç", 52: "panjuru kapa", 53: "perdeyi aç", 54: "perdeyi kapa", 55: "alarmı aç", 56: "alarmı kapa", 57: "evet", 58: "hayır", 59: "parti zamanı", 60: "dinlenme zamanı", 61: "uyku zamanı", 62: "Eve Geliyorum", 63: "Evden Çıkıyorum", 64: "Film Zamanı", 65: "Çalışma Zamanı", 66: "Spor Zamanı"}

COMMAND_NAMES_EN = {1: "turn on the light", 2: "turn off the light", 3: "dim the light", 4: "increase brightness", 5: "decrease brightness", 6: "increase lighting", 7: "decrease lighting", 8: "turn on red light", 9: "turn off red light", 10: "increase red light", 11: "decrease red light", 12: "dim red light", 13: "turn on blue light", 14: "turn off blue light", 15: "increase blue light", 16: "decrease blue light", 17: "dim blue light", 18: "turn on green light", 19: "turn off green light", 20: "increase green light", 21: "decrease green light", 22: "dim green light", 23: "turn on the AC", 24: "turn off the AC", 25: "turn on climate control", 26: "turn off climate control", 27: "turn on heating", 28: "turn off heating", 29: "heat", 30: "cool", 31: "increase temperature", 32: "decrease temperature", 33: "heat the house", 34: "cool the house", 35: "heat the room", 36: "cool the room", 37: "turn on the boiler", 38: "turn off the boiler", 39: "turn on the fan", 40: "turn off the fan", 41: "increase fan", 42: "decrease fan", 43: "turn on the TV", 44: "turn off the TV", 45: "turn on the television", 46: "turn off the television", 47: "turn on multimedia", 48: "turn off multimedia", 49: "turn on music", 50: "turn off music", 51: "open the shutter", 52: "close the shutter", 53: "open the curtain", 54: "close the curtain", 55: "turn on the alarm", 56: "turn off the alarm", 57: "yes", 58: "no", 59: "Party Mode", 60: "Relax Mode", 61: "Sleep Mode", 62: "Arriving Home", 63: "I am arriving", 64: "Leaving Home", 65: "I am leaving", 66: "Movie Time", 67: "Work Time", 68: "Workout Time", 69: "Sport Time"}

# ==========================================
# 1. EN İYİ MODELI SEÇ
# ==========================================
def select_best_model():
    """Select the best SBERT model from Phase 1 results"""
    print("="*70)
    print("🔍 SELECTING BEST SBERT MODEL FROM PHASE 1")
    print("="*70)
    
    results_path = os.path.join(RESULTS_DIR, "comprehensive_results.csv")
    if not os.path.exists(results_path):
        print("❌ comprehensive_results.csv not found!")
        print("   Complete Phase 1 first (02_train_models.py)")
        return None, None
    
    df = pd.read_csv(results_path)
    
    # Find the best SBERT model for each language
    best_models = {}
    
    for lang in ["TR", "EN"]:
        lang_df = df[df['Language'] == lang]
        
        # Average F1-score according to SBERT model
        sbert_performance = lang_df.groupby('SBERT_Model')['F1_Macro'].mean().sort_values(ascending=False)
        
        best_sbert = sbert_performance.index[0]
        best_f1 = sbert_performance.iloc[0]
        
        best_models[lang] = {
            'sbert_model': best_sbert,
            'avg_f1': best_f1
        }
        
        print(f"\n🌍 {lang} Language:")
        print(f"   🏆 Best SBERT: {best_sbert}")
        print(f"   📊 Avg F1-Score: {best_f1:.4f}")
        print(f"\n   SBERT Model Ranking:")
        for idx, (model, score) in enumerate(sbert_performance.items(), 1):
            print(f"      {idx}. {model}: {score:.4f}")
    
    return best_models

# ==========================================
# 2. FINE-TUNING DATA HAZIRLA
# ==========================================
def prepare_training_data(lang, command_dict):
    """
   Create training examples for fine-tuning
    
    Approach: Contrastive Learning
    -Positive examples: Different variations within the same class
    -Negative examples: Different classes
    """
    print(f"\n📋 Preparing Fine-tuning Data for {lang}...")
    
    train_examples = []
    
    # Create positive pairs (similar commands)
    similar_pairs = {
        "TR": [
            ("ışığı aç", "aydınlatmayı arttır", 0.8),
            ("ışığı kapa", "aydınlatmayı azalt", 0.7),
            ("klimayı aç", "iklimlendirmeyi aç", 0.9),
            ("TV aç", "televizyonu aç", 1.0),
            ("ısıt", "ısıtmayı aç", 0.9),
            ("soğut", "klimayı aç", 0.7),
            ("evet", "evet", 1.0),
            ("hayır", "hayır", 1.0),
        ],
        "EN": [
            ("turn on the light", "increase lighting", 0.8),
            ("turn off the light", "decrease lighting", 0.7),
            ("turn on the AC", "turn on climate control", 0.9),
            ("turn on the TV", "turn on the television", 1.0),
            ("heat", "turn on heating", 0.9),
            ("cool", "turn on the AC", 0.7),
            ("yes", "yes", 1.0),
            ("no", "no", 1.0),
        ]
    }
    
    # Positive examples
    for text1, text2, score in similar_pairs.get(lang, []):
        train_examples.append(InputExample(texts=[text1, text2], label=float(score)))
    
    # Positive examples from Commands (same command = similarity 1.0)
    commands = list(command_dict.values())
    for cmd in commands:
        train_examples.append(InputExample(texts=[cmd, cmd], label=1.0))
    
    # Negative examples (different commands = similarity 0.0)
    import random
    random.seed(42)
    
    for _ in range(len(commands)):
        cmd1, cmd2 = random.sample(commands, 2)
        train_examples.append(InputExample(texts=[cmd1, cmd2], label=0.0))
    
    print(f"   ✅ Created {len(train_examples)} training examples")
    return train_examples

# ==========================================
# 3. FINE-TUNE MODEL
# ==========================================
def finetune_sbert(base_model_name, train_examples, lang, epochs=3):
    """Fine-tune the SBERT model"""
    print(f"\n🧠 Fine-tuning {base_model_name} for {lang}...")
    
    # Model mapping
    model_map = {
        "en_orig_1": "all-MiniLM-L6-v2",
        "en_orig_2": "all-mpnet-base-v2",
        "multi_1": "paraphrase-multilingual-MiniLM-L12-v2",
        "multi_2": "distiluse-base-multilingual-cased-v1"
    }
    
    model_name = model_map.get(base_model_name, base_model_name)
    
   # Load base model
    model = SentenceTransformer(model_name)
    print(f"   📥 Base model loaded: {model_name}")
    
 # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Loss function: CosineSimilarityLoss
    train_loss = losses.CosineSimilarityLoss(model)
    
    # Warmup steps
    warmup_steps = int(len(train_dataloader) * epochs * 0.1)
    
    print(f"   🔥 Training Configuration:")
    print(f"      Epochs: {epochs}")
    print(f"      Batch size: 16")
    print(f"      Warmup steps: {warmup_steps}")
    print(f"      Training examples: {len(train_examples)}")
    
    # Fine-tune
    output_path = os.path.join(OUTPUT_DIR, f"finetuned_{base_model_name}_{lang}")
    
# inside the finetune_sbert function
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        # In some versions this argument can avoid conflict:
        callback=None 
    )
    
    print(f"   ✅ Model saved to: {output_path}")
    return output_path

# ==========================================
# 4. COMPARE PERFORMANCE
# ==========================================
def compare_embeddings(base_model_name, finetuned_path, lang, command_dict):
    """Compare Base vs Fine-tuned embedding quality"""
    print(f"\n📊 Comparing Base vs Fine-tuned Embeddings ({lang})...")
    
    model_map = {
        "en_orig_1": "all-MiniLM-L6-v2",
        "en_orig_2": "all-mpnet-base-v2",
        "multi_1": "paraphrase-multilingual-MiniLM-L12-v2",
        "multi_2": "distiluse-base-multilingual-cased-v1"
    }
    
    # Load models
    base_model = SentenceTransformer(model_map[base_model_name])
    finetuned_model = SentenceTransformer(finetuned_path)
    
    commands = list(command_dict.values())
    
    # Embeddings
    base_embeddings = base_model.encode(commands)
    finetuned_embeddings = finetuned_model.encode(commands)
    
    # Calculate cosine similarity matrices
    from sklearn.metrics.pairwise import cosine_similarity
    
    base_sim = cosine_similarity(base_embeddings)
    finetuned_sim = cosine_similarity(finetuned_embeddings)
    
    # Statistics
    results = {
        "Model": ["Base", "Fine-tuned"],
        "Mean_Similarity": [base_sim.mean(), finetuned_sim.mean()],
        "Std_Similarity": [base_sim.std(), finetuned_sim.std()],
        "Max_Similarity": [base_sim.max(), finetuned_sim.max()],
        "Min_Similarity": [base_sim.min(), finetuned_sim.min()]
    }
    
    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(
        os.path.join(OUTPUT_DIR, f"embedding_comparison_{lang}_{base_model_name}.csv"),
        index=False
    )
    
    print(f"\n   Base Model Stats:")
    print(f"      Mean Similarity: {base_sim.mean():.4f}")
    print(f"      Std Similarity: {base_sim.std():.4f}")
    
    print(f"\n   Fine-tuned Model Stats:")
    print(f"      Mean Similarity: {finetuned_sim.mean():.4f}")
    print(f"      Std Similarity: {finetuned_sim.std():.4f}")
    
    return comparison_df

# ==========================================
# MAIN PIPELINE
# ==========================================
def main():
    print("="*70)
    print("🚀 PHASE 2: SBERT FINE-TUNING")
    print("="*70)
    
    # 1.Choose the best model
    best_models = select_best_model()
    
    if best_models is None:
        return
    
    results_summary = []
    
    # 2. Fine-tune for every language
    for lang in ["TR", "EN"]:
        print(f"\n{'='*70}")
        print(f"🌍 PROCESSING {lang} LANGUAGE")
        print(f"{'='*70}")
        
        best_sbert = best_models[lang]['sbert_model']
        command_dict = COMMAND_NAMES_TR if lang == "TR" else COMMAND_NAMES_EN
        
        # Prepare training data
        train_examples = prepare_training_data(lang, command_dict)
        
        # Fine-tune
        finetuned_path = finetune_sbert(
            best_sbert, 
            train_examples, 
            lang, 
            epochs=3
        )
        
        # Compare
        comparison = compare_embeddings(
            best_sbert,
            finetuned_path,
            lang,
            command_dict
        )
        
        results_summary.append({
            "Language": lang,
            "Base_Model": best_sbert,
            "Finetuned_Path": finetuned_path,
            "Base_F1": best_models[lang]['avg_f1']
        })
    
    # Final summary
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "finetuning_summary.csv"), index=False)
    
    print(f"\n{'='*70}")
    print("✅ SBERT FINE-TUNING COMPLETED!")
    print(f"{'='*70}")
    print(f"\n📁 Output:")
    print(f"   Fine-tuned models: {OUTPUT_DIR}/")
    print(f"   - finetuned_{{model}}_{{lang}}/")
    print(f"   - embedding_comparison_{{lang}}_{{model}}.csv")
    print(f"   - finetuning_summary.csv")
    
    print(f"\n⏭️  Next Steps:")
    print(f"   1. Run '05_retrain_with_finetuned.py' to retrain models")
    print(f"   2. Run '06_meta_learning.py' for ensemble methods")

if __name__ == "__main__":
    main()