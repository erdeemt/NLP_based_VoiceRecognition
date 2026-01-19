# -*- coding: utf-8 -*-
"""
Recreate dataset and train models with fine-tuned SBERT models
"""
import os
import numpy as np
import pandas as pd
import librosa
import joblib
import tensorflow as tf
from scipy.signal import butter, filtfilt
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DATA_FOLDER = "Noisy_16kHz_Padded_Segments"
FINETUNED_DIR = "finetuned_models"
OUTPUT_DATASET = "datasets_phase2_finetuned"
OUTPUT_MODELS = "models_phase2_finetuned"
OUTPUT_RESULTS = "results_phase2_finetuned"

os.makedirs(OUTPUT_DATASET, exist_ok=True)
os.makedirs(OUTPUT_MODELS, exist_ok=True)
os.makedirs(OUTPUT_RESULTS, exist_ok=True)

MODELS = {
    "TR": {
        "asr": "reports_deep_learning_only/run_20251210_015930/TR/MEL/model_cnn1d_TR_MEL.h5",
        "scaler": "reports_deep_learning_only/run_20251210_015930/TR/MEL/scaler_TR_MEL.joblib",
        "le": "reports_deep_learning_only/run_20251210_015930/TR/MEL/label_encoder_TR_MEL.joblib"
    },
    "EN": {
        "asr": "reports_deep_learning_only/run_20251210_015930/EN/MEL/model_cnn1d_EN_MEL.h5",
        "scaler": "reports_deep_learning_only/run_20251210_015930/EN/MEL/scaler_EN_MEL.joblib",
        "le": "reports_deep_learning_only/run_20251210_015930/EN/MEL/label_encoder_EN_MEL.joblib"
    }
}

COMMAND_NAMES_TR = {1: "ışığı aç", 2: "ışığı kapa", 3: "ışığı kıs", 4: "parlaklığı arttır", 5: "parlaklığı azalt", 6: "aydınlatmayı arttır", 7: "aydınlatmayı azalt", 8: "kırmızı ışığı aç", 9: "kırmızı ışığı kapa", 10: "kırmızı ışığı arttır", 11: "kırmızı ışığı azalt", 12: "kırmızı ışığı kıs", 13: "mavi ışığı aç", 14: "mavi ışığı kapa", 15: "mavi ışığı arttır", 16: "mavi ışığı azalt", 17: "mavi ışığı kıs", 18: "yeşil ışığı aç", 19: "yeşil ışığı kapa", 20: "yeşil ışığı arttır", 21: "yeşil ışığı azalt", 22: "yeşil ışığı kıs", 23: "klimayı aç", 24: "klimayı kapa", 25: "iklimlendirmeyi aç", 26: "iklimlendirmeyi kapa", 27: "ısıtmayı aç", 28: "ısıtmayı kapa", 29: "ısıt", 30: "soğut", 31: "sıcaklığı arttır", 32: "sıcaklığı düşür", 33: "evi ısıt", 34: "evi soğut", 35: "odayı ısıt", 36: "odayı soğut", 37: "kombiyi aç", 38: "kombiyi kapa", 39: "fanı aç", 40: "fanı kapa", 41: "fanı arttır", 42: "fanı düşür", 43: "TV aç", 44: "TV kapa", 45: "televizyonu aç", 46: "televizyonu kapa", 47: "multimedyayı aç", 48: "multimedyayı kapa", 49: "müzik aç", 50: "müzik kapa", 51: "panjuru aç", 52: "panjuru kapa", 53: "perdeyi aç", 54: "perdeyi kapa", 55: "alarmı aç", 56: "alarmı kapa", 57: "evet", 58: "hayır", 59: "parti zamanı", 60: "dinlenme zamanı", 61: "uyku zamanı", 62: "Eve Geliyorum", 63: "Evden Çıkıyorum", 64: "Film Zamanı", 65: "Çalışma Zamanı", 66: "Spor Zamanı"}

COMMAND_NAMES_EN = {1: "turn on the light", 2: "turn off the light", 3: "dim the light", 4: "increase brightness", 5: "decrease brightness", 6: "increase lighting", 7: "decrease lighting", 8: "turn on red light", 9: "turn off red light", 10: "increase red light", 11: "decrease red light", 12: "dim red light", 13: "turn on blue light", 14: "turn off blue light", 15: "increase blue light", 16: "decrease blue light", 17: "dim blue light", 18: "turn on green light", 19: "turn off green light", 20: "increase green light", 21: "decrease green light", 22: "dim green light", 23: "turn on the AC", 24: "turn off the AC", 25: "turn on climate control", 26: "turn off climate control", 27: "turn on heating", 28: "turn off heating", 29: "heat", 30: "cool", 31: "increase temperature", 32: "decrease temperature", 33: "heat the house", 34: "cool the house", 35: "heat the room", 36: "cool the room", 37: "turn on the boiler", 38: "turn off the boiler", 39: "turn on the fan", 40: "turn off the fan", 41: "increase fan", 42: "decrease fan", 43: "turn on the TV", 44: "turn off the TV", 45: "turn on the television", 46: "turn off the television", 47: "turn on multimedia", 48: "turn off multimedia", 49: "turn on music", 50: "turn off music", 51: "open the shutter", 52: "close the shutter", 53: "open the curtain", 54: "close the curtain", 55: "turn on the alarm", 56: "turn off the alarm", 57: "yes", 58: "no", 59: "Party Mode", 60: "Relax Mode", 61: "Sleep Mode", 62: "Arriving Home", 63: "I am arriving", 64: "Leaving Home", 65: "I am leaving", 66: "Movie Time", 67: "Work Time", 68: "Workout Time", 69: "Sport Time"}

TARGET_SR = 16000
WINDOW_SEC = 1.0
HOP_SEC = 0.25
N_MELS = 20
CONF_THRESHOLD = 0.50
MAX_WINDOWS = 20

# Utility functions
def butter_bandpass_filter(y, sr, fmin=100, fmax=6000, order=6):
    nyq = 0.5 * sr
    b, a = butter(order, [fmin / nyq, fmax / nyq], btype="band")
    return filtfilt(b, a, y).astype(np.float32)

def extract_mel_features(w, sr):
    S = librosa.feature.melspectrogram(y=w, sr=sr, n_fft=1024, hop_length=512, n_mels=N_MELS)
    S_db = librosa.power_to_db(S, ref=np.max)
    if S_db.shape[1] > 30: S_db = S_db[:, :30]
    else: S_db = np.pad(S_db, ((0, 0), (0, 30 - S_db.shape[1])), mode='constant')
    return S_db.flatten()

def pad_or_truncate(vec, target_len):
    if len(vec) > target_len:
        return vec[:target_len]
    return np.pad(vec, (0, target_len - len(vec)), mode='constant')

# Quick training function (no hyperparameter search for speed)
def quick_train_models(X_train_scaled, X_test_scaled, y_train, y_test, lang, model_label):
    """Quick model training (without hyperparameter tuning)"""
    
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, use_label_encoder=False),
        "LightGBM": LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1),
        "MLP": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=300, random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"    Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        
        print(f"      {name}: F1={f1:.4f}, Acc={acc:.4f}")
        
        # Save model
        joblib.dump(model, os.path.join(OUTPUT_MODELS, f"{name.lower()}_{model_label}.joblib"))
        
        results.append({
            "Language": lang,
            "Model_Type": "Fine-tuned_SBERT",
            "ML_Model": name,
            "Accuracy": acc,
            "F1_Macro": f1,
            "Precision_Macro": prec,
            "Recall_Macro": rec
        })
    
    return results

def main():
    print("="*70)
    print("🚀 PHASE 2: RETRAIN WITH FINE-TUNED SBERT")
    print("="*70)
    
    # Find fine-tuned models
    finetuned_models = {}
    
    for lang in ["TR", "EN"]:
        lang_models = [d for d in os.listdir(FINETUNED_DIR) if lang in d and os.path.isdir(os.path.join(FINETUNED_DIR, d))]
        
        if lang_models:
            finetuned_models[lang] = os.path.join(FINETUNED_DIR, lang_models[0])
            print(f"✅ {lang}: {lang_models[0]}")
        else:
            print(f"⚠️  {lang}: Fine-tuned model bulunamadı! 04_finetune_sbert.py çalıştırın.")
            return
    
    # Load ASR models
    print("\n📦 Loading ASR models...")
    asr_tr = tf.keras.models.load_model(MODELS["TR"]["asr"])
    scaler_tr = joblib.load(MODELS["TR"]["scaler"])
    le_tr = joblib.load(MODELS["TR"]["le"])
    
    asr_en = tf.keras.models.load_model(MODELS["EN"]["asr"])
    scaler_en = joblib.load(MODELS["EN"]["scaler"])
    le_en = joblib.load(MODELS["EN"]["le"])
    
    all_results = []
    
    for lang in ["TR", "EN"]:
        print(f"\n{'='*70}")
        print(f"🌍 {lang} - DATASET CREATION WITH FINE-TUNED SBERT")
        print(f"{'='*70}")
        
        # Install fine-tuned SBERT
        finetuned_sbert = SentenceTransformer(finetuned_models[lang])
        print(f"  ✅ Fine-tuned SBERT loaded")
        
        # Model selection
        if lang == "TR":
            asr_model, scaler, le, cmd_dict = asr_tr, scaler_tr, le_tr, COMMAND_NAMES_TR
        else:
            asr_model, scaler, le, cmd_dict = asr_en, scaler_en, le_en, COMMAND_NAMES_EN
        
        data_X, data_Y = [], []
        
        class_folders = sorted([d for d in os.listdir(BASE_DATA_FOLDER) if os.path.isdir(os.path.join(BASE_DATA_FOLDER, d))], key=lambda x: int(x))
        
        for label_str in class_folders:
            label_id = int(label_str)
            if lang == "EN" and label_id == 70:
                continue
            
            folder_path = os.path.join(BASE_DATA_FOLDER, label_str)
            lang_files = [f for f in os.listdir(folder_path) if f.startswith(f"{lang}_") and f.endswith('.wav')]
            
            if not lang_files:
                continue
            
            for wav_name in tqdm(lang_files, leave=False, desc=f"  Class {label_id}"):
                try:
                    y, sr = librosa.load(os.path.join(folder_path, wav_name), sr=TARGET_SR)
                    y = librosa.util.normalize(butter_bandpass_filter(y, sr))
                    
                    win_samples = int(WINDOW_SEC * sr)
                    hop_samples = int(HOP_SEC * sr)
                    window_confidences = []
                    
                    for start in range(0, len(y) - win_samples, hop_samples):
                        window = y[start : start + win_samples]
                        feat = extract_mel_features(window, sr)
                        X_win = scaler.transform(feat.reshape(1, -1))
                        X_win = X_win.reshape(1, X_win.shape[1] // N_MELS, N_MELS)
                        
                        preds = asr_model.predict(X_win, verbose=0)[0]
                        idx = np.argmax(preds)
                        conf = preds[idx]
                        cid = le.inverse_transform([idx])[0]
                        
                        if cid != 0 and conf >= CONF_THRESHOLD:
                            window_confidences.append(conf)
                    
                    if len(window_confidences) == 0:
                        continue
                    
                    # FINE-TUNED SBERT EMBEDDING
                    txt = cmd_dict.get(label_id, "")
                    text_embedding = finetuned_sbert.encode([txt])[0]
                    
                    # CONFIDENCE VECTOR
                    conf_vector = pad_or_truncate(np.array(window_confidences), MAX_WINDOWS)
                    
                    # CONCATENATE
                    final_X = np.concatenate([text_embedding, conf_vector])
                    
                    data_X.append(final_X)
                    data_Y.append(label_id)
                    
                except Exception as e:
                    continue
        
        # Save dataset
        if len(data_X) > 0:
            np.save(os.path.join(OUTPUT_DATASET, f"X_finetuned_{lang}.npy"), np.array(data_X))
            np.save(os.path.join(OUTPUT_DATASET, f"y_finetuned_{lang}.npy"), np.array(data_Y))
            print(f"  ✅ Dataset: {len(data_X)} samples")
            
            # Train models
            print(f"\n  🤖 Training ML Models...")
            X = np.array(data_X)
            y = np.array(data_Y)
            
            le_new = LabelEncoder()
            y_encoded = le_new.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
            
            scaler_new = StandardScaler()
            X_train_scaled = scaler_new.fit_transform(X_train)
            X_test_scaled = scaler_new.transform(X_test)
            
            # Save scaler
            joblib.dump(scaler_new, os.path.join(OUTPUT_MODELS, f"scaler_finetuned_{lang}.joblib"))
            joblib.dump(le_new, os.path.join(OUTPUT_MODELS, f"label_encoder_finetuned_{lang}.joblib"))
            
            # Quick train
            results = quick_train_models(X_train_scaled, X_test_scaled, y_train, y_test, lang, f"finetuned_{lang}")
            all_results.extend(results)
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(os.path.join(OUTPUT_RESULTS, "finetuned_results.csv"), index=False)
        
        print(f"\n{'='*70}")
        print("📊 FINE-TUNED MODEL RESULTS")
        print(f"{'='*70}")
        print(results_df.to_string(index=False))
    
    print(f"\n✅ PHASE 2 RETRAIN COMPLETED!")

if __name__ == "__main__":
    main()