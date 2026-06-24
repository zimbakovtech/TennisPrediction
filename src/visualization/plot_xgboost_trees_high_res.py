import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import pandas as pd
from src.functions.utils import read_file
import seaborn as sns

def plot_xgboost_trees_high_res():
    """
    Load the saved XGBoost model and plot its trees with high resolution and readable text
    """
    print("Loading XGBoost model...")
    
    # Load the saved model
    model = joblib.load("src/models/joblib/xgboost_model.pkl")
    
    # Get feature names
    _, feature_names = read_file()
    
    print(f"Model loaded successfully!")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Feature names: {list(feature_names)}")
    
    # Set up high-resolution plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # High resolution settings
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['figure.titlesize'] = 22
    
    # Plot individual trees with high resolution
    num_trees_to_plot = min(3, model.n_estimators)  # Plot first 3 trees
    
    for i in range(num_trees_to_plot):
        # Create a very large figure for better readability
        plt.figure(figsize=(30, 20))
        
        # Plot the tree using tree_idx parameter
        xgb.plot_tree(model, tree_idx=i, feature_names=list(feature_names))
        plt.title(f'XGBoost Tree {i+1}', fontsize=24, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'xgboost_tree_{i+1}_high_res.png', dpi=600, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print(f"Tree {i+1} saved as xgboost_tree_{i+1}_high_res.png")
    
    # Enhanced feature importance analysis with high resolution
    plt.figure(figsize=(20, 14))
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Create horizontal bar plot with better styling
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df)))
    bars = plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'], 
                   color=colors, height=0.8)
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'], fontsize=16)
    plt.xlabel('Feature Importance', fontsize=18, fontweight='bold')
    plt.title('XGBoost Feature Importance Analysis', fontsize=24, fontweight='bold', pad=20)
    
    # Add value labels on bars with better positioning
    for i, (bar, importance) in enumerate(zip(bars, feature_importance_df['importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}', ha='left', va='center', fontsize=14, fontweight='bold')
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance_high_res.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("Feature importance plot saved as xgboost_feature_importance_high_res.png")
    
    # Print detailed feature importance summary
    print("\n" + "="*70)
    print("DETAILED FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    total_importance = feature_importance_df['importance'].sum()
    for feature, imp in zip(feature_importance_df['feature'], feature_importance_df['importance']):
        percentage = (imp / total_importance) * 100
        print(f"{feature:30s}: {imp:.4f} ({percentage:5.1f}%)")
    
    # Create a summary plot of multiple trees in a grid with high resolution
    fig, axes = plt.subplots(2, 2, figsize=(30, 24))
    axes = axes.flatten()
    
    for i in range(min(4, model.n_estimators)):
        xgb.plot_tree(model, tree_idx=i, ax=axes[i], feature_names=list(feature_names))
        axes[i].set_title(f'Tree {i+1}', fontsize=18, fontweight='bold')
    
    # Hide empty subplots if we have fewer than 4 trees
    for i in range(4, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('XGBoost Ensemble Trees Overview', fontsize=26, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('xgboost_ensemble_trees_high_res.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("Ensemble trees plot saved as xgboost_ensemble_trees_high_res.png")
    
    # Additional analysis: Tree depth distribution
    print("\n" + "="*70)
    print("MODEL ANALYSIS")
    print("="*70)
    print(f"Total number of trees: {model.n_estimators}")
    print(f"Maximum tree depth: {model.max_depth}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Subsample ratio: {model.subsample}")
    print(f"Number of features: {len(feature_names)}")
    
    # Top features analysis
    top_features = feature_importance_df.tail(5)  # Top 5 features
    print(f"\nTop 5 Most Important Features:")
    print("-" * 50)
    for feature, imp in zip(top_features['feature'], top_features['importance']):
        percentage = (imp / total_importance) * 100
        print(f"• {feature}: {percentage:.1f}% of total importance")
    
    # Create a pie chart for top features with high resolution
    plt.figure(figsize=(16, 12))
    top_5_features = feature_importance_df.tail(5)
    other_importance = feature_importance_df.head(len(feature_importance_df)-5)['importance'].sum()
    
    pie_data = list(top_5_features['importance']) + [other_importance]
    pie_labels = list(top_5_features['feature']) + ['Other Features']
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
    wedges, texts, autotexts = plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                                      startangle=90, colors=colors, textprops={'fontsize': 14})
    
    # Make the percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)
    
    plt.title('Feature Importance Distribution (Top 5 Features)', fontsize=22, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance_pie_high_res.png', dpi=600, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.show()
    
    print("Feature importance pie chart saved as xgboost_feature_importance_pie_high_res.png")
    
    # Create a detailed tree structure analysis
    print("\n" + "="*70)
    print("TREE STRUCTURE ANALYSIS")
    print("="*70)
    
    # Analyze the first few trees in detail
    for i in range(min(3, model.n_estimators)):
        print(f"\nTree {i+1} Analysis:")
        print("-" * 30)
        
        # Get tree structure
        tree = model.get_booster().get_dump(dump_format='text')[i]
        lines = tree.strip().split('\n')
        
        # Count nodes and depth
        node_count = len([line for line in lines if 'leaf' in line or '[' in line])
        max_depth = max([line.count('\t') for line in lines if line.strip()])
        
        print(f"  Number of nodes: {node_count}")
        print(f"  Maximum depth: {max_depth}")
        
        # Show first few decision rules
        print(f"  First few decision rules:")
        for j, line in enumerate(lines[:6]):  # Show first 6 lines
            if line.strip():
                print(f"    {line.strip()}")
        if len(lines) > 6:
            print(f"    ... and {len(lines) - 6} more lines")

if __name__ == "__main__":
    plot_xgboost_trees_high_res() 