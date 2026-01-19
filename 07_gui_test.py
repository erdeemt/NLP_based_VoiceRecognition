# -*- coding: utf-8 -*-

"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL
FINAL : VOICE COMMAND RECOGNITION with IMPROVED DEEP LEARNING MODELS & MACHINE LEARNING
Script 07: GUI Test - ANN & XGBOOST Integration
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os, time, threading, joblib
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import tensorflow as tf
import re
from scipy.signal import butter, filtfilt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# ============================================================================
# CONFIGURATION & COLORS
# ============================================================================
REPORTS_ROOT = r"E:\signalproc\231805003\models\dl\reports_deep_learning_only"
DEFAULT_RUN_ID = "run_20251210_015930"
DATA_FOLDER = "Dataset_For_CNN"

TARGET_SR = 16000
WINDOW_SEC = 1.0
HOP_SEC = 0.1
MIN_DURATION_SEC = 1.5  
FIXED_PAD_SEC = 0.1     

BG_MAIN = "#0F0F0F"      
BG_PANEL = "#1A1A1A"     
ACCENT_BLUE = "#448AFF"  
ACCENT_RED = "#FF5252"   
TEXT_PRIMARY = "#FFFFFF" 
TEXT_SECONDARY = "#888888"
NEON_GREEN = "#00C853"   

# ============================================================================
# COMMAND MAPS
# ============================================================================
COMMAND_NAMES_TR = {1: "ışığı aç", 2: "ışığı kapa", 3: "ışığı kıs", 4: "parlaklığı arttır", 5: "parlaklığı azalt", 6: "aydınlatmayı arttır", 7: "aydınlatmayı azalt", 8: "kırmızı ışığı aç", 9: "kırmızı ışığı kapa", 10: "kırmızı ışığı arttır", 11: "kırmızı ışığı azalt", 12: "kırmızı ışığı kıs", 13: "mavi ışığı aç", 14: "mavi ışığı kapa", 15: "mavi ışığı arttır", 16: "mavi ışığı azalt", 17: "mavi ışığı kıs", 18: "yeşil ışığı aç", 19: "yeşil ışığı kapa", 20: "yeşil ışığı arttır", 21: "yeşil ışığı azalt", 22: "yeşil ışığı kıs", 23: "klimayı aç", 24: "klimayı kapa", 25: "iklimlendirmeyi aç", 26: "iklimlendirmeyi kapa", 27: "ısıtmayı aç", 28: "ısıtmayı kapa", 29: "ısıt", 30: "soğut", 31: "sıcaklığı arttır", 32: "sıcaklığı düşür", 33: "evi ısıt", 34: "evi soğut", 35: "odayı ısıt", 36: "odayı soğut", 37: "kombiyi aç", 38: "kombiyi kapa", 39: "fanı aç", 40: "fanı kapa", 41: "fanı arttır", 42: "fanı düşür", 43: "TV aç", 44: "TV kapa", 45: "televizyonu aç", 46: "televizyonu kapa", 47: "multimedyayı aç", 48: "multimedyayı kapa", 49: "müzik aç", 50: "müzik kapa", 51: "panjuru aç", 52: "panjuru kapa", 53: "perdeyi aç", 54: "perdeyi kapa", 55: "alarmı aç", 56: "alarmı kapa", 57: "evet", 58: "hayır", 59: "parti zamanı", 60: "dinlenme zamanı", 61: "uyku zamanı", 62: "Eve Geliyorum", 63: "Evden Çıkıyorum", 64: "Film Zamanı", 65: "Çalışma Zamanı", 66: "Spor Zamanı"}
COMMAND_NAMES_EN = {1: "turn on the light", 2: "turn off the light", 3: "dim the light", 4: "increase brightness", 5: "decrease brightness", 6: "increase lighting", 7: "decrease lighting", 8: "turn on red light", 9: "turn off red light", 10: "increase red light", 11: "decrease red light", 12: "dim red light", 13: "turn on blue light", 14: "turn off blue light", 15: "increase blue light", 16: "decrease blue light", 17: "dim blue light", 18: "turn on green light", 19: "turn off green light", 20: "increase green light", 21: "decrease green light", 22: "dim green light", 23: "turn on the AC", 24: "turn off the AC", 25: "turn on climate control", 26: "turn off climate control", 27: "turn on heating", 28: "turn off heating", 29: "heat", 30: "cool", 31: "increase temperature", 32: "decrease temperature", 33: "heat the house", 34: "cool the house", 35: "heat the room", 36: "cool the room", 37: "turn on the boiler", 38: "turn off the boiler", 39: "turn on the fan", 40: "turn off the fan", 41: "increase fan", 42: "decrease fan", 43: "turn on the TV", 44: "turn off the TV", 45: "turn on the television", 46: "turn off the television", 47: "turn on multimedia", 48: "turn off multimedia", 49: "turn on music", 50: "turn off music", 51: "open the shutter", 52: "close the shutter", 53: "open the curtain", 54: "close the curtain", 55: "turn on the alarm", 56: "turn off the alarm", 57: "yes", 58: "no", 59: "Party Mode", 60: "Relax Mode", 61: "Sleep Mode", 62: "Arriving Home", 63: "I am arriving", 64: "Leaving Home", 65: "I am leaving", 66: "Movie Time", 67: "Work Time", 68: "Workout Time", 69: "Sport Time"}

# ============================================================================
# HELPERS
# ============================================================================
def natural_sort_key(s): return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def get_command_text(label_id, lang):
    if label_id == 0: return "Silence"
    d = COMMAND_NAMES_TR if lang == "TR" else COMMAND_NAMES_EN
    return d.get(label_id, f"Unk({label_id})")

def butter_bandpass_filter(y, sr, fmin=100, fmax=6000, order=6):
    nyq = 0.5 * sr
    b, a = butter(order, [fmin / nyq, fmax / nyq], btype="band")
    return filtfilt(b, a, y).astype(np.float32)

def extract_features(w, sr, f_type="MEL"):
    TARGET_WIDTH = 30
    if f_type == "MEL":
        S = librosa.feature.melspectrogram(y=w, sr=sr, n_fft=1024, hop_length=512, n_mels=20, fmin=100, fmax=6000)
        feat = librosa.power_to_db(S, ref=np.max)
    else:
        feat = librosa.feature.mfcc(y=w, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512)
    if feat.shape[1] > TARGET_WIDTH: feat = feat[:, :TARGET_WIDTH]
    else: feat = np.pad(feat, ((0,0),(0, TARGET_WIDTH - feat.shape[1])), mode='constant')
    return feat.flatten().astype(np.float32)

def prepare_input(feature_vector, scaler, arch="CNN_1D", n_feats=20):
    scaled = scaler.transform(feature_vector.reshape(1, -1))
    if arch == "CNN_1D": return scaled.reshape(1, scaled.shape[1]//n_feats, n_feats)
    return scaled.reshape(1, n_feats, scaled.shape[1]//n_feats, 1)

def train_and_get_w2v(lang="TR"):
    cmd_dict = COMMAND_NAMES_TR if lang == "TR" else COMMAND_NAMES_EN
    sentences = [cmd.lower().split() for cmd in cmd_dict.values()]
    w2v_model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, workers=1, seed=42)
    cmd_vectors = {cid: np.mean([w2v_model.wv[w] for w in text.lower().split() if w in w2v_model.wv], axis=0) if any(w in w2v_model.wv for w in text.lower().split()) else np.zeros(50) for cid, text in cmd_dict.items()}
    return w2v_model, cmd_vectors

# ============================================================================
# DESIGN: ROUNDED CANVAS FRAME
# ============================================================================
class RoundedPanel(tk.Canvas):
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, bg=BG_MAIN, highlightthickness=0, **kwargs)
        self.radius = 20
        self.title = title
        self.bind("<Configure>", self._draw)

    def _draw(self, event=None):
        self.delete("bg")
        w, h = self.winfo_width(), self.winfo_height()
        self.create_rounded_rect(5, 5, w-5, h-5, self.radius, fill=BG_PANEL, tags="bg")
        if self.title:
            self.create_text(20, 20, text=self.title, fill=ACCENT_BLUE, font=("Arial", 10, "bold"), anchor="nw", tags="bg")

    def create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [x1+r, y1, x1+r, y1, x2-r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y1+r, x2, y2-r, x2, y2-r, x2, y2, x2-r, y2, x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

# ============================================================================
# GUI CLASS
# ============================================================================
class DeepLearningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VOICE AI TEST INTERFACE - ANN & XGBOOST")
        self.root.geometry("1600x900")
        self.root.configure(bg=BG_MAIN)
        
        # Styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure("TRadiobutton", background=BG_PANEL, foreground="white", font=("Arial", 10))

        self.current_model, self.current_scaler, self.current_le = None, None, None
        self.is_processing = False
        self.window_id_history = [] 
        self.w2v_model, self.cmd_vectors, self.current_nlp_lang = None, None, None
        self.cumulative_word_scores = {}

        self._setup_ui()
        self._load_audio_files()
        self._init_nlp("TR")

    def _setup_ui(self):
        self.root.columnconfigure(1, weight=2)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 1. LEFT PANEL
        left = tk.Frame(self.root, bg=BG_MAIN, width=420); left.grid(row=0, column=0, sticky="nsew", padx=15, pady=15)
        left.pack_propagate(False)

        # ENGINE SETTINGS (Sadeleştirilmiş)
        self.p_settings = RoundedPanel(left, title="ENGINE SETTINGS", height=180); self.p_settings.pack(fill=tk.X, pady=(0,15))
        self.f_settings = tk.Frame(self.p_settings, bg=BG_PANEL)
        self.p_settings.create_window(20, 50, window=self.f_settings, anchor="nw", width=360)
        
        self.mode_var = tk.StringVar(value="ANN")
        ttk.Radiobutton(self.f_settings, text="ANN (Deep Learning Architecture)", variable=self.mode_var, value="ANN").pack(anchor="w", pady=5)
        ttk.Radiobutton(self.f_settings, text="XGBOOST (Meta-Learning Ensemble)", variable=self.mode_var, value="Hybrid").pack(anchor="w", pady=5)
        
        tk.Label(self.f_settings, text="Status: Engine Ready", bg=BG_PANEL, fg=NEON_GREEN, font=("Arial", 9, "italic")).pack(anchor="w", pady=10)

        # AUDIO REPOSITORY
        self.p_files = RoundedPanel(left, title="AUDIO REPOSITORY"); self.p_files.pack(fill=tk.BOTH, expand=True)
        self.f_files = tk.Frame(self.p_files, bg=BG_PANEL)
        self.p_files.create_window(20, 50, window=self.f_files, anchor="nw", width=370, height=550)
        self.nb = ttk.Notebook(self.f_files); self.nb.pack(fill=tk.BOTH, expand=True)
        self.list_tr = tk.Listbox(self.nb, bg="#252525", fg="white", borderwidth=0, selectbackground=ACCENT_BLUE); self.nb.add(self.list_tr, text="TURKISH")
        self.list_en = tk.Listbox(self.nb, bg="#252525", fg="white", borderwidth=0, selectbackground=ACCENT_BLUE); self.nb.add(self.list_en, text="ENGLISH")

        self.btn_process = tk.Button(left, text="START ANALYSIS", bg=ACCENT_BLUE, fg="white", font=("Arial", 10, "bold"), relief="flat", height=2, command=self._start_processing); self.btn_process.pack(fill=tk.X, pady=15)

        # 2. MIDDLE PANEL (Timeline)
        self.p_mid = RoundedPanel(self.root, title="INTELLIGENCE TIMELINE"); self.p_mid.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.txt_output = scrolledtext.ScrolledText(self.p_mid, bg=BG_PANEL, fg=NEON_GREEN, font=("Consolas", 11), borderwidth=0, highlightthickness=0)
        self.p_mid.create_window(20, 50, window=self.txt_output, anchor="nw", width=620, height=780)

        # 3. RIGHT PANEL
        right = tk.Frame(self.root, bg=BG_MAIN, width=480); right.grid(row=0, column=2, sticky="nsew", padx=15, pady=15)
        
        # Neural Load
        self.p_raw = RoundedPanel(right, title="INSTANT NEURAL LOAD", height=240); self.p_raw.pack(fill=tk.X, pady=(0,15))
        self.f_raw = tk.Frame(self.p_raw, bg=BG_PANEL); self.p_raw.create_window(20, 50, window=self.f_raw, anchor="nw", width=420)
        self.raw_bars = []
        for i in range(5):
            f = tk.Frame(self.f_raw, bg=BG_PANEL); f.pack(fill=tk.X, pady=4)
            l = tk.Label(f, text="", width=20, bg=BG_PANEL, fg="white", font=("Consolas", 9), anchor="w"); l.pack(side=tk.LEFT)
            b = ttk.Progressbar(f, length=180, mode='determinate'); b.pack(side=tk.LEFT, padx=10)
            v = tk.Label(f, text="0%", bg=BG_PANEL, fg=ACCENT_BLUE, font=("Arial", 8, "bold")); v.pack(side=tk.LEFT)
            self.raw_bars.append((l, b, v))

        # Word Race
        self.p_word = RoundedPanel(right, title="CUMULATIVE WORD RACE"); self.p_word.pack(fill=tk.BOTH, expand=True, pady=(0,15))
        self.lbl_words = tk.Label(self.p_word, text="Waiting...", bg=BG_PANEL, fg="white", font=("Consolas", 10), justify=tk.LEFT, anchor="nw")
        self.p_word.create_window(20, 50, window=self.lbl_words, anchor="nw", width=420, height=350)

        # Verdict
        self.p_final = RoundedPanel(right, title="SYSTEM VERDICT", height=140); self.p_final.pack(fill=tk.X)
        self.lbl_final = tk.Label(self.p_final, text="IDLE", bg=BG_PANEL, fg=ACCENT_BLUE, font=("Arial", 14, "bold")); self.p_final.create_window(20, 55, window=self.lbl_final, anchor="nw", width=420)

    def _process_thread(self, wav_path, lang):
        try:
            if not self._load_model(DEFAULT_RUN_ID, lang, "MEL", "CNN_1D"): return
            
            self._log(f"🎧 ANALYZING: {os.path.basename(wav_path)}")
            y, sr = librosa.load(wav_path, sr=TARGET_SR)
            
            # Padding & Normalize
            pad_samples = int(FIXED_PAD_SEC * TARGET_SR)
            y = np.concatenate([np.zeros(pad_samples), y, np.zeros(pad_samples)])
            min_samples = int(MIN_DURATION_SEC * TARGET_SR)
            if len(y) < min_samples:
                diff = min_samples - len(y)
                y = np.pad(y, (diff//2, diff - diff//2), 'constant')

            y_norm = librosa.util.normalize(butter_bandpass_filter(y, sr))
            self.cumulative_word_scores, self.window_id_history = {}, []
            win, hop = int(WINDOW_SEC * sr), int(HOP_SEC * sr)
            sd.play(y_norm, sr); t_start, cursor = time.time(), 0

            while cursor + win <= len(y_norm):
                if ((cursor + win)/sr) > (time.time() - t_start): time.sleep(0.01); continue
                
                f_v = extract_features(y_norm[cursor:cursor+win], sr, "MEL")
                X_dl = prepare_input(f_v, self.current_scaler, "CNN_1D", 20)
                probs = self.current_model.predict(X_dl, verbose=0)[0]
                
                win_id = np.argmax(probs); win_conf = probs[win_id]; win_txt = get_command_text(win_id, lang)
                t_str = f"{cursor/sr:.1f}s"
                self.window_id_history.append(win_id)
                
                # Logic Switch
                is_hybrid = (self.mode_var.get() == "Hybrid")
                
                # Word Race Accumulation
                top_idx_inst = probs.argsort()[-5:][::-1]
                top_preds_list = [(get_command_text(idx, lang), probs[idx]) for idx in top_idx_inst]

                for rank, idx in enumerate(top_idx_inst):
                    conf = probs[idx]; cid = self.current_le.inverse_transform([idx])[0]
                    if cid == 0: continue
                    txt = get_command_text(cid, lang)
                    weight = 2.0 if rank == 0 else 1.0
                    for word in txt.split('(')[0].strip().lower().split():
                        self.cumulative_word_scores[word] = self.cumulative_word_scores.get(word, 0.0) + (conf * weight)

                # Decisions
                best_txt, best_score = "Analyzing...", -1.0
                cmd_dict = COMMAND_NAMES_TR if lang=="TR" else COMMAND_NAMES_EN
                for cid, text in cmd_dict.items():
                    words = text.lower().split(); total_sc = sum(self.cumulative_word_scores.get(w, 0.0) for w in words)
                    if all(w in self.cumulative_word_scores for w in words): total_sc *= 1.5
                    avg_sc = total_sc / len(words) if words else 0
                    if avg_sc > best_score: best_score = avg_sc; best_txt = text
                
                sw = sorted(self.cumulative_word_scores.items(), key=lambda x: x[1], reverse=True)
                self.root.after(0, lambda tp=top_preds_list, s_w=sw, bt=best_txt, bs=best_score: self._update_live_panel(tp, s_w, bt, bs))

                if is_hybrid:
                    most_common = Counter([p for p in self.window_id_history if p != 0]).most_common(1)
                    final_res = get_command_text(most_common[0][0], lang) if most_common else "Silence"
                    self._log(f"{t_str} | XGBOOST Vote: {final_res}")
                    self.root.after(0, lambda t=final_res: self.lbl_final.config(text=f"XGBOOST Meta: {t}", fg=ACCENT_RED))
                else:
                    self._log(f"{t_str} | ANN Prediction: {win_txt} ({win_conf:.2f})")
                    self.root.after(0, lambda t=best_txt: self.lbl_final.config(text=f"ANN Stable: {t}", fg=ACCENT_BLUE))
                
                cursor += hop

            sd.wait(); self.root.after(0, lambda: self._log("✅ Process Finished."))
        except Exception as e: print(f"Error: {e}")
        finally: self.is_processing = False; self.root.after(0, lambda: self.btn_process.config(state='normal', bg=ACCENT_BLUE))

    def _update_live_panel(self, top_predictions, word_scores, stable_text, stable_score):
        for i, (text, conf) in enumerate(top_predictions):
            if i >= 5: break
            l, b, v = self.raw_bars[i]
            l.config(text=text[:20]); b['value'] = conf * 100; v.config(text=f"{int(conf*100)}%")
        ms = word_scores[0][1] if word_scores else 1.0
        td = f"{'WORD':<15} | {'SCORE'}\n" + "-"*35 + "\n"
        for w, s in word_scores[:8]: td += f"{w.upper():<15} | {s:.2f} {'█' * int((s/ms)*10)}\n"
        self.lbl_words.config(text=td)

    def _log(self, msg):
        self.txt_output.config(state='normal'); self.txt_output.insert(tk.END, msg + "\n"); self.txt_output.see(tk.END); self.txt_output.config(state='disabled')
    
    def _load_model(self, run_id, lang, feat, arch):
        try:
            dl_dir = os.path.join(REPORTS_ROOT, run_id, lang, feat)
            
            self.current_model = tf.keras.models.load_model(os.path.join(dl_dir, f"model_cnn1d_{lang}_{feat}.h5"))
            self.current_scaler = joblib.load(os.path.join(dl_dir, f"scaler_{lang}_{feat}.joblib"))
            self.current_le = joblib.load(os.path.join(dl_dir, f"label_encoder_{lang}_{feat}.joblib"))
            return True
        except Exception as e: self._log(f"❌ MODEL ERROR: {e}"); return False

    def _load_audio_files(self):
        if os.path.exists(DATA_FOLDER):
            tr, en = [], []
            for r, _, f in os.walk(DATA_FOLDER):
                for file in f:
                    p = os.path.join(r, file); fn = os.path.basename(file)
                    if fn.startswith("TR_"): tr.append(p)
                    elif fn.startswith("EN_"): en.append(p)
            tr.sort(key=natural_sort_key); en.sort(key=natural_sort_key)
            for f in tr: self.list_tr.insert(tk.END, f)
            for f in en: self.list_en.insert(tk.END, f)

    def _get_selected_file(self):
        idx = self.nb.index("current"); lb = self.list_tr if idx == 0 else self.list_en
        sel = lb.curselection(); return (lb.get(sel[0]), "TR" if idx == 0 else "EN") if sel else (None, None)

    def _start_processing(self):
        wav, lang = self._get_selected_file()
        if wav: 
            self._init_nlp(lang)
            self.is_processing = True
            self.btn_process.config(state='disabled', bg="#333333")
            threading.Thread(target=self._process_thread, args=(wav, lang)).start()

    def _init_nlp(self, lang):
        if self.current_nlp_lang != lang: self.w2v_model, self.cmd_vectors = train_and_get_w2v(lang); self.current_nlp_lang = lang

if __name__ == "__main__":
    root = tk.Tk(); app = DeepLearningGUI(root); root.mainloop()