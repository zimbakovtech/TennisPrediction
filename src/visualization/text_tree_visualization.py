import joblib
import pandas as pd
from src.functions.utils import read_file

def visualize_trees_as_text():
    """
    Create a text-based visualization of XGBoost trees that's easy to read
    """
    print("Loading XGBoost model...")
    
    # Load the saved model
    model = joblib.load("src/models/joblib/xgboost_model.pkl")
    
    # Get feature names
    _, feature_names = read_file()
    
    print(f"Model loaded successfully!")
    print(f"Number of trees: {model.n_estimators}")
    print(f"Feature names: {list(feature_names)}")
    
    # Create a mapping from feature indices to names
    feature_map = {f'f{i}': name for i, name in enumerate(feature_names)}
    
    print("\n" + "="*80)
    print("XGBOOST TREE VISUALIZATION (TEXT FORMAT)")
    print("="*80)
    
    # Analyze the first few trees in detail
    for tree_idx in range(min(3, model.n_estimators)):
        print(f"\n{'='*60}")
        print(f"TREE {tree_idx + 1} DETAILED STRUCTURE")
        print(f"{'='*60}")
        
        # Get tree structure
        tree_dump = model.get_booster().get_dump(dump_format='text')[tree_idx]
        lines = tree_dump.strip().split('\n')
        
        # Count nodes and depth
        node_count = len([line for line in lines if 'leaf' in line or '[' in line])
        max_depth = max([line.count('\t') for line in lines if line.strip()])
        
        print(f"Tree Statistics:")
        print(f"  • Number of nodes: {node_count}")
        print(f"  • Maximum depth: {max_depth}")
        print(f"  • Tree index: {tree_idx}")
        print()
        
        print("Tree Structure (with feature names):")
        print("-" * 50)
        
        for line in lines:
            if line.strip():
                # Replace feature indices with actual names
                for f_idx, f_name in feature_map.items():
                    line = line.replace(f_idx, f_name)
                
                # Add proper indentation for better readability
                indent_level = line.count('\t')
                indent = "  " * indent_level
                clean_line = line.strip()
                
                # Format the line for better readability
                if 'leaf' in clean_line:
                    # This is a leaf node
                    print(f"{indent}📄 LEAF: {clean_line}")
                elif '[' in clean_line:
                    # This is a decision node
                    print(f"{indent}🔀 DECISION: {clean_line}")
                else:
                    print(f"{indent}{clean_line}")
        
        print()
    
    # Create a summary of all trees
    print(f"\n{'='*80}")
    print("ENSEMBLE SUMMARY")
    print(f"{'='*80}")
    
    print(f"Total number of trees: {model.n_estimators}")
    print(f"Maximum tree depth: {model.max_depth}")
    print(f"Learning rate: {model.learning_rate}")
    print(f"Subsample ratio: {model.subsample}")
    
    # Feature importance summary
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    print(f"\nFeature Importance Ranking:")
    print("-" * 40)
    total_importance = feature_importance_df['importance'].sum()
    
    for i, (feature, imp) in enumerate(zip(feature_importance_df['feature'], feature_importance_df['importance'])):
        percentage = (imp / total_importance) * 100
        print(f"{i+1:2d}. {feature:25s}: {imp:.4f} ({percentage:5.1f}%)")
    
    # Create a simple decision path example
    print(f"\n{'='*80}")
    print("EXAMPLE DECISION PATH")
    print(f"{'='*80}")
    
    # Get the first tree and show a sample decision path
    first_tree = model.get_booster().get_dump(dump_format='text')[0]
    lines = first_tree.strip().split('\n')
    
    print("Sample decision path from Tree 1:")
    print("-" * 40)
    
    path_count = 0
    for line in lines[:10]:  # Show first 10 lines as example
        if line.strip():
            # Replace feature indices with actual names
            for f_idx, f_name in feature_map.items():
                line = line.replace(f_idx, f_name)
            
            indent_level = line.count('\t')
            indent = "  " * indent_level
            clean_line = line.strip()
            
            if 'leaf' in clean_line:
                print(f"{indent}🎯 FINAL PREDICTION: {clean_line}")
                path_count += 1
                if path_count >= 3:  # Show 3 example paths
                    break
            elif '[' in clean_line:
                print(f"{indent}❓ IF {clean_line}")
    
    print(f"\nNote: This shows just a few example paths from the first tree.")
    print(f"Each tree has {len([l for l in lines if 'leaf' in l])} leaf nodes (final predictions).")
    print(f"The ensemble combines predictions from all {model.n_estimators} trees.")

if __name__ == "__main__":
    visualize_trees_as_text() 