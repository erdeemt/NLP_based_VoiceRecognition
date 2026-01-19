# -*- coding: utf-8 -*-
"""
PDF Requirement: "Word embedding approach implemented in the first term project"
This script uses Word2Vec + TF-IDF weighting for sentence vectorization.
"""
import os
import numpy as np
import librosa
import joblib
import tensorflow as tf
from scipy.signal import butter, filtfilt
from embedder_tool import ProjectEmbedder 
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_DATA_FOLDER = "Noisy_16kHz_Padded_Segments"

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
    elif len(vec) < target_len:
        return np.pad(vec, (0, target_len - len(vec)), mode='constant')
    return vec

def main():
    print("🚀 WORD2VEC + TF-IDF DATASET PREPARATION")
    print("="*60)
    
    # Upload ASR Models
    asr_tr = tf.keras.models.load_model(MODELS["TR"]["asr"])
    scaler_tr = joblib.load(MODELS["TR"]["scaler"])
    le_tr = joblib.load(MODELS["TR"]["le"])
    
    asr_en = tf.keras.models.load_model(MODELS["EN"]["asr"])
    scaler_en = joblib.load(MODELS["EN"]["scaler"])
    le_en = joblib.load(MODELS["EN"]["le"])
    
    # Embedder
    embedder = ProjectEmbedder()
    
    for lang in ["TR", "EN"]:
        print(f"\n{'='*60}")
        print(f"🌍 {lang} LANGUAGE IS PROCESSED")
        print(f"{'='*60}")
        
        # Model selection
        if lang == "TR":
            asr_model, scaler, le, cmd_dict = asr_tr, scaler_tr, le_tr, COMMAND_NAMES_TR
        else:
            asr_model, scaler, le, cmd_dict = asr_en, scaler_en, le_en, COMMAND_NAMES_EN
        
        # Word2Vec Training
        print(f"  🧠 Word2Vec being trained ({lang})...")
        sentences = [cmd.lower().split() for cmd in cmd_dict.values()]
        embedder.train_w2v(sentences, vector_size=100, window=5)
        
        # TF-IDF Corpus
        corpus = list(cmd_dict.values())
        
        # Separate dataset for each approach
        for approach in ["mean", "tfidf"]:
            print(f"\n  🔹 Approach: Word2Vec-{approach.upper()}")
            
            data_X = []
            data_Y = []
            
            class_folders = sorted([d for d in os.listdir(BASE_DATA_FOLDER) 
                                   if os.path.isdir(os.path.join(BASE_DATA_FOLDER, d))], 
                                  key=lambda x: int(x))
            
            for label_str in class_folders:
                label_id = int(label_str)
                
                if lang == "EN" and label_id == 70:
                    continue
                
                folder_path = os.path.join(BASE_DATA_FOLDER, label_str)
                all_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
                lang_files = [f for f in all_files if f.startswith(f"{lang}_")]
                
                if not lang_files:
                    continue
                
                for wav_name in tqdm(lang_files, leave=False, desc=f"  Class {label_id}"):
                    try:
                        wav_path = os.path.join(folder_path, wav_name)
                        y, sr = librosa.load(wav_path, sr=TARGET_SR)
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
                        
                        # TEXT EMBEDDING
                        txt = cmd_dict.get(label_id, "")
                        
                        if approach == "mean":
                            text_embedding = embedder.get_word_mean_vector(txt)
                        else:  # tfidf
                            # Train TF-IDF first time
                            if embedder.tfidf_vectorizer is None:
                                text_embedding = embedder.get_tfidf_weighted_vector(txt, corpus=corpus)
                            else:
                                text_embedding = embedder.get_tfidf_weighted_vector(txt)
                        
                        # CONFIDENCE VECTOR
                        conf_vector = pad_or_truncate(np.array(window_confidences), MAX_WINDOWS)
                        
                        # CONCATENATION
                        final_X = np.concatenate([text_embedding, conf_vector])
                        
                        data_X.append(final_X)
                        data_Y.append(label_id)
                        
                    except Exception as e:
                        continue
            
            # Save
            if len(data_X) > 0:
                output_dir = f"datasets_phase1/{lang}/word2vec_{approach}"
                os.makedirs(output_dir, exist_ok=True)
                
                np.save(f"{output_dir}/X_features.npy", np.array(data_X))
                np.save(f"{output_dir}/y_labels.npy", np.array(data_Y))
                
                # Save the embedder too (required for inference)
                joblib.dump(embedder, f"{output_dir}/embedder.joblib")
                
                print(f"    ✅ {len(data_X)} sample saved")
                print(f"       Shape: {np.array(data_X).shape}")
    
    print(f"\n{'='*60}")
    print("✅ WORD2VEC DATASET READY!")
    print(f"{'='*60}")
    print("\n📁 Created Files:")
    print("   datasets_phase1/")
    print("   ├── TR/")
    print("   │   ├── word2vec_mean/")
    print("   │   └── word2vec_tfidf/")
    print("   └── EN/")
    print("       ├── word2vec_mean/")
    print("       └── word2vec_tfidf/")

if __name__ == "__main__":
    main()