# -*- coding: utf-8 -*-
"""
COMPREHENSIVE REPORTING & VISUALIZATION SYSTEM
PDF Requirement: "A comprehensive report detailing the solutions and results"

Bu script:
1. Collects all model results
2. Creates detailed comparison tables
3. Confusion produces matrices
4. Draws performance graphs
5. Highlights the best models
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Turkish character support for Matplotlib
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# CONFIGURATION
# ==========================================
RESULTS_DIR = "results_phase1"
MODELS_DIR = "models_phase1"
DATASETS_DIR = "datasets_phase1"
REPORT_OUTPUT = "final_reports"
FIGURES_OUTPUT = os.path.join(REPORT_OUTPUT, "figures")

os.makedirs(REPORT_OUTPUT, exist_ok=True)
os.makedirs(FIGURES_OUTPUT, exist_ok=True)

LANGUAGES = ["TR", "EN"]
SBERT_MODELS = {
    "TR": ["multi_1", "multi_2"],
    "EN": ["en_orig_1", "en_orig_2", "multi_1", "multi_2"]
}

# ==========================================
# 1. COMPREHENSIVE CSV REPORTS
# ==========================================
def generate_summary_tables():
    """Collect all results and create summary tables"""
    print("📊 Generating Summary Tables...")
    
   # Comprehensive results'u yükle
    if not os.path.exists(os.path.join(RESULTS_DIR, "comprehensive_results.csv")):
        print("❌ comprehensive_results.csv not found!")
        return None
    
    df = pd.read_csv(os.path.join(RESULTS_DIR, "comprehensive_results.csv"))
    
    #1. GENERAL SUMMARY
    summary = df.groupby(['Language', 'ML_Model']).agg({
        'Accuracy': ['mean', 'std', 'max'],
        'F1_Macro': ['mean', 'std', 'max'],
        'Precision_Macro': ['mean', 'std', 'max'],
        'Recall_Macro': ['mean', 'std', 'max']
    }).round(4)
    
    summary.to_csv(os.path.join(REPORT_OUTPUT, "01_overall_summary.csv"))
    print("  ✅ 01_overall_summary.csv")
    
    #2. BEST MODELS (For every language)
    best_models = []
    for lang in LANGUAGES:
        lang_df = df[df['Language'] == lang]
        if len(lang_df) > 0:
            best_idx = lang_df['F1_Macro'].idxmax()
            best_models.append(lang_df.loc[best_idx])
    
    best_df = pd.DataFrame(best_models)
    best_df.to_csv(os.path.join(REPORT_OUTPUT, "02_best_models.csv"), index=False)
    print("  ✅ 02_best_models.csv")
    
    #3. SBERT MODEL COMPARISON
    sbert_comparison = df.groupby(['Language', 'SBERT_Model']).agg({
        'F1_Macro': ['mean', 'max'],
        'Accuracy': ['mean', 'max']
    }).round(4)
    
    sbert_comparison.to_csv(os.path.join(REPORT_OUTPUT, "03_sbert_comparison.csv"))
    print("  ✅ 03_sbert_comparison.csv")
    
    # 4. ML ALGORITHM COMPARISON
    ml_comparison = df.groupby(['Language', 'ML_Model']).agg({
        'F1_Macro': ['mean', 'std', 'min', 'max'],
        'Accuracy': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    ml_comparison.to_csv(os.path.join(REPORT_OUTPUT, "04_ml_algorithm_comparison.csv"))
    print("  ✅ 04_ml_algorithm_comparison.csv")
    
    # 5. HYPERPARAMETER ANALYSIS
    if 'Best_Params' in df.columns:
        hyperparams = df[['Language', 'ML_Model', 'SBERT_Model', 'F1_Macro', 'Best_Params']].copy()
        hyperparams = hyperparams.sort_values('F1_Macro', ascending=False)
        hyperparams.to_csv(os.path.join(REPORT_OUTPUT, "05_hyperparameter_analysis.csv"), index=False)
        print("  ✅ 05_hyperparameter_analysis.csv")
    
    # 6. DETAILED METRICS BY CLASS(For each model)
    print("\n  📋 Generating per-class reports...")
    for lang in LANGUAGES:
        for sbert in SBERT_MODELS[lang]:
            for model in ['ANN', 'RandomForest', 'XGBoost', 'LightGBM', 'MLP_Sklearn']:
                report_path = os.path.join(RESULTS_DIR, f"report_{model}_{lang}_{sbert}.csv")
                if os.path.exists(report_path):
                    report_df = pd.read_csv(report_path, index_col=0)
                    
                    # Get only numeric classes (not accuracy, macro avg, weighted avg)
                    numeric_classes = [idx for idx in report_df.index if str(idx).isdigit()]
                    class_report = report_df.loc[numeric_classes]
                    
                    output_name = f"class_report_{model}_{lang}_{sbert}.csv"
                    class_report.to_csv(os.path.join(REPORT_OUTPUT, output_name))
    
    print("  ✅ Per-class reports generated")
    
    return df

# ==========================================
# 2. VISUALIZATIONS
# ==========================================
def plot_model_comparison(df):
    """Model performance comparison graphs"""
    print("\n📈 Generating Comparison Plots...")
    
    # Plot 1: F1-Score Comparison (Bar Chart)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, lang in enumerate(LANGUAGES):
        lang_df = df[df['Language'] == lang]
        
        # Average F1 for each SBERT model
        pivot = lang_df.pivot_table(
            index='ML_Model', 
            columns='SBERT_Model', 
            values='F1_Macro', 
            aggfunc='mean'
        )
        
        pivot.plot(kind='bar', ax=axes[idx], rot=45)
        axes[idx].set_title(f'{lang} - F1-Score by Model & SBERT', fontsize=14, fontweight='bold')
        axes[idx].set_ylabel('F1-Score (Macro)', fontsize=12)
        axes[idx].set_xlabel('ML Model', fontsize=12)
        axes[idx].legend(title='SBERT Model', bbox_to_anchor=(1.05, 1))
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT, "01_f1_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 01_f1_comparison.png")
    
    # Plot 2: Accuracy vs F1-Score Scatter
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, lang in enumerate(LANGUAGES):
        lang_df = df[df['Language'] == lang]
        
        for model in lang_df['ML_Model'].unique():
            model_df = lang_df[lang_df['ML_Model'] == model]
            axes[idx].scatter(
                model_df['Accuracy'], 
                model_df['F1_Macro'], 
                label=model, 
                s=100, 
                alpha=0.6
            )
        
        axes[idx].set_title(f'{lang} - Accuracy vs F1-Score', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Accuracy', fontsize=12)
        axes[idx].set_ylabel('F1-Score (Macro)', fontsize=12)
        axes[idx].legend(bbox_to_anchor=(1.05, 1))
        axes[idx].grid(alpha=0.3)
        axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.3)  # Diagonal line
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT, "02_accuracy_vs_f1.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 02_accuracy_vs_f1.png")
    
    # Plot 3: Heatmap - F1 Scores
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, lang in enumerate(LANGUAGES):
        lang_df = df[df['Language'] == lang]
        pivot = lang_df.pivot_table(
            index='ML_Model',
            columns='SBERT_Model',
            values='F1_Macro',
            aggfunc='mean'
        )
        
        sns.heatmap(
            pivot, 
            annot=True, 
            fmt='.4f', 
            cmap='RdYlGn', 
            ax=axes[idx],
            vmin=0, 
            vmax=1,
            cbar_kws={'label': 'F1-Score'}
        )
        axes[idx].set_title(f'{lang} - F1-Score Heatmap', fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('SBERT Model', fontsize=12)
        axes[idx].set_ylabel('ML Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT, "03_f1_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 03_f1_heatmap.png")

def plot_metric_distributions(df):
    """Metric distributions (box plots)"""
    print("\n📦 Generating Distribution Plots...")
    
    metrics = ['Accuracy', 'F1_Macro', 'Precision_Macro', 'Recall_Macro']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        for lang in LANGUAGES:
            lang_df = df[df['Language'] == lang]
            
            # Box plot
            lang_df.boxplot(column=metric, by='ML_Model', ax=axes[idx])
            axes[idx].set_title(f'{metric} Distribution by Model', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('ML Model', fontsize=10)
            axes[idx].set_ylabel(metric, fontsize=10)
            axes[idx].grid(alpha=0.3)
            plt.sca(axes[idx])
            plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_OUTPUT, "04_metric_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✅ 04_metric_distributions.png")

# ==========================================
# MAIN REPORT GENERATION
# ==========================================
def main():
    print("="*70)
    print("📊 COMPREHENSIVE REPORT GENERATION")
    print("="*70)
    
    # 1. Generate CSV Reports
    df = generate_summary_tables()
    
    if df is not None:
        # 2. Generate Visualizations
        plot_model_comparison(df)
        plot_metric_distributions(df)
        
        # 3. Generate Final Summary
        print("\n📝 Generating Final Summary...")
        
        summary_text = f"""
# COMPREHENSIVE PROJECT REPORT

## Overall Statistics
- Total Models Trained: {len(df)}
- Languages: {', '.join(LANGUAGES)}
- Best Overall F1-Score: {df['F1_Macro'].max():.4f}
- Average F1-Score: {df['F1_Macro'].mean():.4f}

## Top 5 Models (by F1-Score)
{df.nlargest(5, 'F1_Macro')[['Language', 'SBERT_Model', 'ML_Model', 'F1_Macro', 'Accuracy']].to_string(index=False)}

## Language-wise Best Models
"""
        
        for lang in LANGUAGES:
            lang_df = df[df['Language'] == lang]
            if len(lang_df) > 0:
                best_idx = lang_df['F1_Macro'].idxmax()
                best = lang_df.loc[best_idx]
                summary_text += f"\n### {lang}\n"
                summary_text += f"- Best Model: {best['ML_Model']}\n"
                summary_text += f"- SBERT: {best['SBERT_Model']}\n"
                summary_text += f"- F1-Score: {best['F1_Macro']:.4f}\n"
                summary_text += f"- Accuracy: {best['Accuracy']:.4f}\n"
        
        # Save summary
        with open(os.path.join(REPORT_OUTPUT, "00_SUMMARY.txt"), 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print("  ✅ 00_SUMMARY.txt")
        
        print(f"\n{'='*70}")
        print("✅ COMPREHENSIVE REPORT COMPLETED!")
        print(f"{'='*70}")
        print(f"\n📁 Output Directory: {REPORT_OUTPUT}/")
        print("   CSV Reports:")
        print("   - 01_overall_summary.csv")
        print("   - 02_best_models.csv")
        print("   - 03_sbert_comparison.csv")
        print("   - 04_ml_algorithm_comparison.csv")
        print("   - 05_hyperparameter_analysis.csv")
        print("\n   Visualizations:")
        print("   - figures/01_f1_comparison.png")
        print("   - figures/02_accuracy_vs_f1.png")
        print("   - figures/03_f1_heatmap.png")
        print("   - figures/04_metric_distributions.png")
    else:
        print("❌ The report could not be created! Train the models first.")

if __name__ == "__main__":
    main()