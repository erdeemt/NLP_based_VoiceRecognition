# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, auc
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==========================================
# PHASE 1: CONFIGURATION
# ==========================================
LANGUAGES = ["EN","TR"]
SBERT_MODELS = {
    "EN": ["en_orig_1", "en_orig_2","multi_1", "multi_2","word2vec_mean","word2vec_tfidf"],
    "TR": ["multi_1", "multi_2","word2vec_mean","word2vec_tfidf"]
}
INPUT_DIR = "datasets_phase1"
OUTPUT_DIR = "models_phase1"
RESULTS_DIR = "results_phase1"
PLOTS_DIR = "plots_phase1"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ==========================================
# VISUALIZATION UTILITIES
# ==========================================
class ReportGenerator:
    def __init__(self, plots_dir):
        self.plots_dir = plots_dir
        
    def plot_confusion_matrix(self, y_true, y_pred, classes, model_name, save_name):
        """Confusion Matrix Heatmap"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'cm_{save_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_training_history(self, history, model_name, save_name):
        """ANN Training History"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_title(f'{model_name} - Accuracy', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Accuracy', fontsize=11)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_title(f'{model_name} - Loss', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'history_{save_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_metrics_comparison(self, results_df, save_name):
        """Model Comparison Bar Chart"""
        metrics = ['Accuracy', 'F1_Macro', 'Precision_Macro', 'Recall_Macro']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = results_df.sort_values(metric, ascending=False).head(10)
            
            bars = ax.barh(range(len(data)), data[metric], color=sns.color_palette("viridis", len(data)))
            ax.set_yticks(range(len(data)))
            ax.set_yticklabels([f"{row['ML_Model']}\n({row['SBERT_Model']})" for _, row in data.iterrows()], fontsize=9)
            ax.set_xlabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'Top 10 Models by {metric}', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, data[metric])):
                ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, f'comparison_{save_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_model_family_comparison(self, results_df):
        """Model Family Performance"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        
        metrics = ['Accuracy', 'F1_Macro', 'Precision_Macro', 'Recall_Macro']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            data = results_df.groupby('ML_Model')[metric].agg(['mean', 'std']).sort_values('mean', ascending=False)
            
            x = range(len(data))
            ax.bar(x, data['mean'], yerr=data['std'], capsize=5, alpha=0.7, color=sns.color_palette("Set2", len(data)))
            ax.set_xticks(x)
            ax.set_xticklabels(data.index, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_title(f'Average {metric} by Model Family', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, (mean, std) in enumerate(zip(data['mean'], data['std'])):
                ax.text(i, mean + std + 0.01, f'{mean:.3f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'model_family_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_language_comparison(self, results_df):
        """Language Performance Comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Boxplot
        ax1 = axes[0]
        results_df.boxplot(column='F1_Macro', by='Language', ax=ax1)
        ax1.set_title('F1-Score Distribution by Language', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Language', fontsize=11)
        ax1.set_ylabel('F1-Score (Macro)', fontsize=11)
        plt.sca(ax1)
        plt.xticks(rotation=0)
        
        # Violin plot
        ax2 = axes[1]
        sns.violinplot(data=results_df, x='Language', y='Accuracy', ax=ax2, palette='Set3')
        ax2.set_title('Accuracy Distribution by Language', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Language', fontsize=11)
        ax2.set_ylabel('Accuracy', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'language_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_sbert_comparison(self, results_df):
        """SBERT Model Comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, lang in enumerate(LANGUAGES):
            ax = axes[idx]
            lang_data = results_df[results_df['Language'] == lang]
            
            sbert_perf = lang_data.groupby('SBERT_Model')[['Accuracy', 'F1_Macro']].mean()
            
            x = np.arange(len(sbert_perf))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, sbert_perf['Accuracy'], width, label='Accuracy', alpha=0.8)
            bars2 = ax.bar(x + width/2, sbert_perf['F1_Macro'], width, label='F1-Score', alpha=0.8)
            
            ax.set_xlabel('SBERT Model', fontsize=11, fontweight='bold')
            ax.set_ylabel('Score', fontsize=11)
            ax.set_title(f'{lang} - SBERT Model Comparison', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(sbert_perf.index, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'sbert_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_heatmap_all_results(self, results_df):
        """Heatmap of All Results"""
        pivot = results_df.pivot_table(
            values='F1_Macro', 
            index='ML_Model', 
            columns=['Language', 'SBERT_Model']
        )
        
        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', 
                    cbar_kws={'label': 'F1-Score (Macro)'}, linewidths=0.5)
        plt.title('F1-Score Heatmap: All Models × Languages × SBERT', fontsize=14, fontweight='bold')
        plt.xlabel('Language & SBERT Model', fontsize=11)
        plt.ylabel('ML Model', fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'heatmap_all_results.png'), dpi=300, bbox_inches='tight')
        plt.close()

# ==========================================
# DEEP LEARNING MODEL BUILDER
# ==========================================
def build_ann_model(input_dim, num_classes, architecture="medium"):
    if architecture == "small":
        hidden_layers = [128, 64]
    elif architecture == "large":
        hidden_layers = [512, 256, 128, 64]
    else:
        hidden_layers = [256, 128, 64]
    
    model = keras.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    for i, units in enumerate(hidden_layers):
        model.add(layers.Dense(
            units, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001),
            name=f'hidden_{i+1}'
        ))
        model.add(layers.Dropout(0.3, name=f'dropout_{i+1}'))
    
    model.add(layers.Dense(num_classes, activation='softmax', name='output'))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_ann_with_cv(X_train, y_train, input_dim, num_classes):
    print("    🧠 ANN/MLP Training with Architecture Search...")
    
    best_model = None
    best_score = 0
    best_arch = None
    best_history = None
    
    architectures = ["small", "medium", "large"]
    learning_rates = [0.001, 0.0005]
    
    for arch in architectures:
        for lr in learning_rates:
            print(f"      Testing: arch={arch}, lr={lr}")
            
            model = build_ann_model(input_dim, num_classes, architecture=arch)
            model.optimizer.learning_rate.assign(lr)
            
            early_stop = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=100,
                batch_size=32,
                callbacks=[early_stop],
                verbose=0
            )
            
            val_acc = max(history.history['val_accuracy'])
            
            if val_acc > best_score:
                best_score = val_acc
                best_model = model
                best_arch = f"{arch}_lr{lr}"
                best_history = history
            
            print(f"        Val Accuracy: {val_acc:.4f}")
    
    print(f"    ✅ Best ANN Config: {best_arch} (Val Acc: {best_score:.4f})")
    return best_model, best_history

# ==========================================
# TRADITIONAL ML MODELS
# ==========================================
def get_ml_models():
    return {
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [100, 200, 300],
                "max_depth": [10, 20, None],
                "min_samples_split": [2, 5]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(eval_metric='mlogloss', random_state=42, use_label_encoder=False),
            "params": {
                "n_estimators": [100, 200],
                "learning_rate": [0.01, 0.05, 0.1],
                "max_depth": [3, 5, 7]
            }
        },
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=500, random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "solver": ['lbfgs'] # En hızlı solver
            }
        }
    }

# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================
def main():
    print("="*70)
    print("🚀 PHASE 1: MODEL TRAINING WITH COMPREHENSIVE EVALUATION")
    print("="*70)
    
    report_gen = ReportGenerator(PLOTS_DIR)
    all_results = []
    all_detailed_results = []
    
    for lang in LANGUAGES:
        print(f"\n{'='*70}")
        print(f"🌍 LANGUAGE: {lang}")
        print(f"{'='*70}")
        
        for sbert_model in SBERT_MODELS[lang]:
            print(f"\n{'─'*70}")
            print(f"📊 SBERT Model: {sbert_model}")
            print(f"{'─'*70}")
            
            # Load Data
            data_path = os.path.join(INPUT_DIR, lang, sbert_model)
            X = np.load(os.path.join(data_path, "X_features.npy"))
            y = np.load(os.path.join(data_path, "y_labels.npy"))
            
            print(f"  📥 Data Loaded: X={X.shape}, y={y.shape}")
            
            # Label Encoding
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            
            # Train/Test Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=0.2, 
                random_state=42, 
                stratify=y_encoded
            )
            
            # Scaling
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Save scaler and label encoder
            model_name = f"{lang}_{sbert_model}"
            joblib.dump(scaler, os.path.join(OUTPUT_DIR, f"scaler_{model_name}.joblib"))
            joblib.dump(le, os.path.join(OUTPUT_DIR, f"label_encoder_{model_name}.joblib"))
            
            # ==========================================
            # 1. DEEP LEARNING (ANN/MLP)
            # ==========================================
            print(f"\n  🤖 Training Deep Learning Model (ANN)...")
            
            ann_model, ann_history = train_ann_with_cv(
                X_train_scaled, y_train,
                input_dim=X_train_scaled.shape[1],
                num_classes=len(np.unique(y_train))
            )
            
            # ANN Prediction
            y_pred_ann_proba = ann_model.predict(X_test_scaled, verbose=0)
            y_pred_ann = np.argmax(y_pred_ann_proba, axis=1)
            
            # ANN Metrics
            acc_ann = accuracy_score(y_test, y_pred_ann)
            f1_ann = f1_score(y_test, y_pred_ann, average='macro')
            prec_ann = precision_score(y_test, y_pred_ann, average='macro', zero_division=0)
            rec_ann = recall_score(y_test, y_pred_ann, average='macro', zero_division=0)
            
            print(f"    ✅ ANN Results: Acc={acc_ann:.4f}, F1={f1_ann:.4f}, Prec={prec_ann:.4f}, Rec={rec_ann:.4f}")
            
            # Save ANN Model
            ann_model.save(os.path.join(OUTPUT_DIR, f"ann_{model_name}.keras"))
            
            # Visualizations for ANN
            report_gen.plot_training_history(ann_history, f"ANN ({lang}/{sbert_model})", f"ann_{model_name}")
            report_gen.plot_confusion_matrix(y_test, y_pred_ann, le.classes_, f"ANN ({lang}/{sbert_model})", f"ann_{model_name}")
            
            # Classification Report
            report_ann = classification_report(y_test, y_pred_ann, target_names=le.classes_, output_dict=True, zero_division=0)
            
            all_results.append({
                "Language": lang,
                "SBERT_Model": sbert_model,
                "ML_Model": "ANN_DeepLearning",
                "Accuracy": acc_ann,
                "F1_Macro": f1_ann,
                "Precision_Macro": prec_ann,
                "Recall_Macro": rec_ann,
                "Num_Params": ann_model.count_params()
            })
            
            # Detailed per-class results
            for class_name in le.classes_:
                if class_name in report_ann:
                    all_detailed_results.append({
                        "Language": lang,
                        "SBERT_Model": sbert_model,
                        "ML_Model": "ANN_DeepLearning",
                        "Class": class_name,
                        "Precision": report_ann[class_name]['precision'],
                        "Recall": report_ann[class_name]['recall'],
                        "F1-Score": report_ann[class_name]['f1-score'],
                        "Support": report_ann[class_name]['support']
                    })
            
            # Save detailed report
            report_df = pd.DataFrame(report_ann).transpose()
            report_df.to_csv(os.path.join(RESULTS_DIR, f"report_ANN_{model_name}.csv"))
            
            # ==========================================
            # 2. TRADITIONAL ML MODELS
            # ==========================================
            ml_models = get_ml_models()
            
            for ml_name, config in ml_models.items():
                print(f"\n  🔧 Training {ml_name}...")
                
                # Grid Search with Cross-Validation
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                grid = GridSearchCV(
                    config["model"],
                    config["params"],
                    cv=cv,
                    n_jobs=-1,
                    scoring='f1_macro',
                    verbose=0
                )
                
                grid.fit(X_train_scaled, y_train)
                best_model = grid.best_estimator_
                
                # Prediction
                y_pred = best_model.predict(X_test_scaled)
                
                # Metrics
                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
                prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
                rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
                
                print(f"    ✅ {ml_name}: Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Rec={rec:.4f}")
                print(f"       Best Params: {grid.best_params_}")
                
                # Save Model
                joblib.dump(best_model, os.path.join(OUTPUT_DIR, f"{ml_name.lower()}_{model_name}.joblib"))
                
                # Visualizations
                report_gen.plot_confusion_matrix(y_test, y_pred, le.classes_, f"{ml_name} ({lang}/{sbert_model})", f"{ml_name.lower()}_{model_name}")
                
                # Classification Report
                report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                report_df.to_csv(os.path.join(RESULTS_DIR, f"report_{ml_name}_{model_name}.csv"))
                
                all_results.append({
                    "Language": lang,
                    "SBERT_Model": sbert_model,
                    "ML_Model": ml_name,
                    "Accuracy": acc,
                    "F1_Macro": f1,
                    "Precision_Macro": prec,
                    "Recall_Macro": rec,
                    "Best_Params": str(grid.best_params_),
                    "CV_Best_Score": grid.best_score_
                })
                
                # Detailed per-class results
                for class_name in le.classes_:
                    if class_name in report:
                        all_detailed_results.append({
                            "Language": lang,
                            "SBERT_Model": sbert_model,
                            "ML_Model": ml_name,
                            "Class": class_name,
                            "Precision": report[class_name]['precision'],
                            "Recall": report[class_name]['recall'],
                            "F1-Score": report[class_name]['f1-score'],
                            "Support": report[class_name]['support']
                        })
    
    # ==========================================
    # FINAL COMPREHENSIVE REPORT & VISUALIZATIONS
    # ==========================================
    print(f"\n{'='*70}")
    print("📊 GENERATING COMPREHENSIVE REPORT & VISUALIZATIONS")
    print(f"{'='*70}")
    
    results_df = pd.DataFrame(all_results)
    detailed_df = pd.DataFrame(all_detailed_results)
    
    # Save full results
    results_df.to_csv(os.path.join(RESULTS_DIR, "comprehensive_results.csv"), index=False)
    detailed_df.to_csv(os.path.join(RESULTS_DIR, "detailed_per_class_results.csv"), index=False)
    
    # Statistical Summary
    stats_summary = results_df.groupby('ML_Model')[['Accuracy', 'F1_Macro', 'Precision_Macro', 'Recall_Macro']].agg(['mean', 'std', 'min', 'max'])
    stats_summary.to_csv(os.path.join(RESULTS_DIR, "statistical_summary.csv"))
    
    # Generate all visualizations
    print("\n📈 Generating Visualizations...")
    report_gen.plot_metrics_comparison(results_df, "all_models")
    report_gen.plot_model_family_comparison(results_df)
    report_gen.plot_language_comparison(results_df)
    report_gen.plot_sbert_comparison(results_df)
    report_gen.plot_heatmap_all_results(results_df)
    
    # Print summary
    print("\n📈 TOP 5 MODELS BY F1-SCORE:")
    top_models = results_df.nlargest(5, 'F1_Macro')[['Language', 'SBERT_Model', 'ML_Model', 'F1_Macro', 'Accuracy']]
    print(top_models.to_string(index=False))
    
    # Language-specific summaries
    for lang in LANGUAGES:
        lang_df = results_df[results_df['Language'] == lang]
        lang_df.to_csv(os.path.join(RESULTS_DIR, f"results_{lang}.csv"), index=False)
        print(f"\n🌍 {lang} - Best Model: {lang_df.loc[lang_df['F1_Macro'].idxmax(), 'ML_Model']} (F1={lang_df['F1_Macro'].max():.4f})")
    
    # Create Executive Summary
    with open(os.path.join(RESULTS_DIR, "executive_summary.txt"), "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("EXECUTIVE SUMMARY - PHASE 1 MODEL TRAINING\n")
        f.write("="*70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Total Models Trained: {len(results_df)}\n")
        f.write(f"Languages: {', '.join(LANGUAGES)}\n")
        f.write(f"SBERT Models: {sum(len(v) for v in SBERT_MODELS.values())}\n")
        f.write(f"ML Algorithms: {results_df['ML_Model'].nunique()}\n\n")
        
        f.write("="*70 + "\n")
        f.write("BEST PERFORMING MODELS\n")
        f.write("="*70 + "\n\n")
        
        for metric in ['F1_Macro', 'Accuracy']:
            best_row = results_df.loc[results_df[metric].idxmax()]
            f.write(f"Best {metric}:\n")
            f.write(f"  Model: {best_row['ML_Model']}\n")
            f.write(f"  Language: {best_row['Language']}\n")
            f.write(f"  SBERT: {best_row['SBERT_Model']}\n")
            f.write(f"  Score: {best_row[metric]:.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("STATISTICAL SUMMARY BY MODEL TYPE\n")
        f.write("="*70 + "\n\n")
        f.write(stats_summary.to_string())
    
    print(f"\n{'='*70}")
    print("✅ PHASE 1 TRAINING & REPORTING COMPLETED!")
    print(f"{'='*70}")
    print(f"\n📁 Output Files:")
    print(f"   Models: {OUTPUT_DIR}/")
    print(f"   Results & Reports: {RESULTS_DIR}/")
    print(f"     - comprehensive_results.csv")
    print(f"     - detailed_per_class_results.csv")
    print(f"     - statistical_summary.csv")
    print(f"     - executive_summary.txt")
    print(f"     - results_TR.csv, results_EN.csv")
    print(f"     - report_[MODEL]_[LANG]_[SBERT].csv")
    print(f"   Visualizations: {PLOTS_DIR}/")
    print("✅ PHASE 1 TRAINING & REPORTING ENDED!")
if __name__ == "__main__":
    main()