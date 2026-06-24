import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
import pandas as pd
from src.functions.utils import read_file
import seaborn as sns

def plot_xgboost_trees_improved():
    """
    Load the saved XGBoost model and plot its trees with improved analysis
    """
    print("Loading XGBoost model...")
    
    # Load the saved model
    model = joblib.load("src/models/joblib/xgboost_model.pkl")
    
    # Get feature names
    _, feature_names = read_file()
    
    print(f"Model loaded successfully!")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Max depth: {model.max_depth}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Feature names: {list(feature_names)}")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    
    # Plot individual trees (using tree_idx instead of num_trees)
    num_trees_to_plot = min(3, model.n_estimators)  # Plot first 3 trees
    
    for i in range(num_trees_to_plot):
        plt.figure(figsize=(20, 12))
        
        # Plot the tree using tree_idx parameter
        xgb.plot_tree(model, tree_idx=i, feature_names=list(feature_names))
        plt.title(f'XGBoost Tree {i+1}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'xgboost_tree_{i+1}_improved.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Tree {i+1} saved as xgboost_tree_{i+1}_improved.png")
    
    # Enhanced feature importance analysis
    plt.figure(figsize=(14, 10))
    
    # Get feature importance
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=True)
    
    # Create horizontal bar plot with better styling
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df)))
    bars = plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'], color=colors)
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('XGBoost Feature Importance Analysis', fontsize=16, fontweight='bold')
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, feature_importance_df['importance'])):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{importance:.4f}', ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Feature importance plot saved as xgboost_feature_importance_improved.png")
    
    # Print detailed feature importance summary
    print("\n" + "="*60)
    print("DETAILED FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    total_importance = feature_importance_df['importance'].sum()
    for feature, imp in zip(feature_importance_df['feature'], feature_importance_df['importance']):
        percentage = (imp / total_importance) * 100
        print(f"{feature:25s}: {imp:.4f} ({percentage:5.1f}%)")
    
    # Create a summary plot of multiple trees in a grid
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    for i in range(min(4, model.n_estimators)):
        xgb.plot_tree(model, tree_idx=i, ax=axes[i], feature_names=list(feature_names))
        axes[i].set_title(f'Tree {i+1}', fontsize=12, fontweight='bold')
    
    # Hide empty subplots if we have fewer than 4 trees
    for i in range(4, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('XGBoost Ensemble Trees Overview', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig('xgboost_ensemble_trees_improved.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Ensemble trees plot saved as xgboost_ensemble_trees_improved.png")
    
    # Additional analysis: Tree depth distribution
    print("\n" + "="*60)
    print("MODEL ANALYSIS")
    print("="*60)
    print(f"Total number of trees: {model.n_estimators}")
    print(f"Maximum tree depth: {model.max_depth}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Subsample ratio: {model.subsample}")
    print(f"Number of features: {len(feature_names)}")
    
    # Top features analysis
    top_features = feature_importance_df.tail(5)  # Top 5 features
    print(f"\nTop 5 Most Important Features:")
    print("-" * 40)
    for feature, imp in zip(top_features['feature'], top_features['importance']):
        percentage = (imp / total_importance) * 100
        print(f"• {feature}: {percentage:.1f}% of total importance")
    
    # Create a pie chart for top features
    plt.figure(figsize=(10, 8))
    top_5_features = feature_importance_df.tail(5)
    other_importance = feature_importance_df.head(len(feature_importance_df)-5)['importance'].sum()
    
    pie_data = list(top_5_features['importance']) + [other_importance]
    pie_labels = list(top_5_features['feature']) + ['Other Features']
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(pie_data)))
    plt.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=colors)
    plt.title('Feature Importance Distribution (Top 5 Features)', fontsize=14, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance_pie.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Feature importance pie chart saved as xgboost_feature_importance_pie.png")

if __name__ == "__main__":
    plot_xgboost_trees_improved() 