import pickle
import os

def compare_localized_vs_full_model_scores():
    """Compare brain alignment scores between full model and localized units."""
    
    # Paths to the score files
    full_model_path = "dumps/scores_model=gpt2_benchmark=Pereira2018.243sentences-linear_seed=42.pkl"
    localized_model_path = "dumps/scores_model=gpt2_benchmark=Pereira2018.243sentences-linear_localized=True_nunits=128_seed=42.pkl"
    
    print("=== Brain Alignment Score Comparison ===\n")
    
    # Load and display full model scores
    if os.path.exists(full_model_path):
        print("📊 FULL MODEL (All Units)")
        print("─" * 40)
        with open(full_model_path, 'rb') as f:
            full_scores = pickle.load(f)
        
        print(f"Aggregated Score: {full_scores.item():.4f}")
        
        # Show layer-wise scores if available
        if 'layer' in full_scores.coords:
            print("\nLayer-wise Scores:")
            for layer in full_scores['layer'].values:
                layer_score = full_scores.sel(layer=layer)
                print(f"  {layer}: {layer_score.item():.4f}")
        
        # Show total number of units
        if 'neuroid' in full_scores.coords:
            total_units = len(full_scores['neuroid'])
            print(f"\nTotal Units Used: {total_units}")
    else:
        print(f"❌ Full model scores not found: {full_model_path}")
    
    print("\n" + "="*60 + "\n")
    
    # Load and display localized model scores
    if os.path.exists(localized_model_path):
        print("🎯 LOCALIZED MODEL (Language-Selective Units Only)")
        print("─" * 50)
        with open(localized_model_path, 'rb') as f:
            localized_scores = pickle.load(f)
        
        print(f"Aggregated Score: {localized_scores.item():.4f}")
        
        # Show layer-wise scores if available
        if 'layer' in localized_scores.coords:
            print("\nLayer-wise Scores:")
            for layer in localized_scores['layer'].values:
                layer_score = localized_scores.sel(layer=layer)
                print(f"  {layer}: {layer_score.item():.4f}")
        
        # Show total number of units
        if 'neuroid' in localized_scores.coords:
            total_units = len(localized_scores['neuroid'])
            print(f"\nTotal Language Units Used: {total_units}")
    else:
        print(f"❌ Localized model scores not found: {localized_model_path}")
        print("   Run the localized brain alignment first!")
    
    print("\n" + "="*60 + "\n")
    
    # Compare scores if both are available
    if os.path.exists(full_model_path) and os.path.exists(localized_model_path):
        print("📈 COMPARISON")
        print("─" * 20)
        
        full_score = full_scores.item()
        localized_score = localized_scores.item()
        
        improvement = localized_score - full_score
        improvement_pct = (improvement / full_score) * 100 if full_score != 0 else 0
        
        print(f"Full Model Score:       {full_score:.4f}")
        print(f"Localized Model Score:  {localized_score:.4f}")
        print(f"Improvement:            {improvement:+.4f} ({improvement_pct:+.2f}%)")
        
        if improvement > 0:
            print("\n✅ Language localization IMPROVED brain alignment!")
        elif improvement < 0:
            print("\n⚠️  Language localization decreased brain alignment.")
        else:
            print("\n➖ No change in brain alignment score.")
        
        # Show efficiency gain
        if 'neuroid' in full_scores.coords and 'neuroid' in localized_scores.coords:
            full_units = len(full_scores['neuroid'])
            localized_units = len(localized_scores['neuroid'])
            efficiency = (localized_units / full_units) * 100
            
            print(f"\n🎯 Efficiency: Using only {efficiency:.1f}% of units ({localized_units}/{full_units})")
            
            if improvement >= 0:
                print(f"   → Better or equal performance with {100-efficiency:.1f}% fewer units!")

if __name__ == "__main__":
    compare_localized_vs_full_model_scores()
