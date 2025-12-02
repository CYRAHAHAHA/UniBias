"""
GPT-2 UniBias Verification Script
Run this script locally to verify all changes are correct before uploading to Colab
"""

import sys
import os

def check_file_exists(filepath):
    """Check if file exists"""
    if os.path.exists(filepath):
        print(f"✅ {filepath}")
        return True
    else:
        print(f"❌ {filepath} - NOT FOUND")
        return False

def check_file_content(filepath, search_strings, label):
    """Check if file contains expected GPT-2 specific code"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        found_all = True
        for search_str in search_strings:
            if search_str in content:
                print(f"  ✅ Contains: {search_str[:50]}...")
            else:
                print(f"  ❌ Missing: {search_str[:50]}...")
                found_all = False
        
        return found_all
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")
        return False

def main():
    print("="*80)
    print("GPT-2 UniBias Verification Script")
    print("="*80)
    
    all_passed = True
    
    # Check files exist
    print("\n1. Checking files exist...")
    print("-"*80)
    files = [
        "model_modifications.py",
        "attention_manipulate.py",
        "FFN_manipulate.py",
        "evaluation.py",
        "main.py",
        "UniBias_Colab.ipynb",
        "GPT2_MIGRATION_GUIDE.md",
        "READY_FOR_COLAB.md"
    ]
    
    for f in files:
        if not check_file_exists(f):
            all_passed = False
    
    # Check model_modifications.py
    print("\n2. Checking model_modifications.py...")
    print("-"*80)
    if not check_file_content("model_modifications.py", [
        "model.transformer.h",
        "layer.attn",
        "model.config.n_layer",
        "model.config.n_head",
        "GPT-2"
    ], "model_modifications.py"):
        all_passed = False
    
    # Check attention_manipulate.py
    print("\n3. Checking attention_manipulate.py...")
    print("-"*80)
    if not check_file_content("attention_manipulate.py", [
        "model.transformer.h",
        "layer.attn",
        "model.config.n_layer"
    ], "attention_manipulate.py"):
        all_passed = False
    
    # Check FFN_manipulate.py
    print("\n4. Checking FFN_manipulate.py...")
    print("-"*80)
    if not check_file_content("FFN_manipulate.py", [
        "model.transformer.h",
        "model.transformer.ln_f",
        "mlp.c_proj",
        "mlp.c_fc"
    ], "FFN_manipulate.py"):
        all_passed = False
    
    # Check main.py
    print("\n5. Checking main.py...")
    print("-"*80)
    if not check_file_content("main.py", [
        "gpt2",
        "model.transformer.ln_f",
        "torch.float32",
        "pad_token"
    ], "main.py"):
        all_passed = False
    
    # Check notebook
    print("\n6. Checking UniBias_Colab.ipynb...")
    print("-"*80)
    if not check_file_content("UniBias_Colab.ipynb", [
        "GPT-2",
        "gpt2-medium",
        "float32",
        "transformer.ln_f"
    ], "UniBias_Colab.ipynb"):
        all_passed = False
    
    # Check for Llama-2 references (should NOT exist)
    print("\n7. Checking for leftover Llama-2 references...")
    print("-"*80)
    check_files = ["model_modifications.py", "attention_manipulate.py", "FFN_manipulate.py", "main.py"]
    
    llama_refs = [
        "meta-llama",
        "Llama-2-7b",
        "model.model.layers",
        "self_attn",
        "o_proj",
        "down_proj",
        "gate_proj",
        "up_proj",
        "model.model.norm",
        "num_hidden_layers",
        "num_attention_heads"
    ]
    
    for filepath in check_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            found_llama_ref = False
            for ref in llama_refs:
                if ref in content and ref not in ["# meta-llama", "# model.model.layers"]:  # Allow in comments
                    print(f"  ⚠️ {filepath}: Found Llama-2 reference: {ref}")
                    found_llama_ref = True
                    all_passed = False
            
            if not found_llama_ref:
                print(f"  ✅ {filepath}: No Llama-2 references")
        except Exception as e:
            print(f"  ❌ Error checking {filepath}: {e}")
            all_passed = False
    
    # Final summary
    print("\n" + "="*80)
    if all_passed:
        print("✅ ALL CHECKS PASSED!")
        print("="*80)
        print("\nYour UniBias code is ready for GPT-2!")
        print("Next step: Upload UniBias_Colab.ipynb to Google Colab")
        print("\nRecommended first run:")
        print("  Model: gpt2-medium")
        print("  Dataset: sst2")
        print("  Expected runtime: 15-30 minutes")
        return 0
    else:
        print("❌ SOME CHECKS FAILED")
        print("="*80)
        print("\nPlease review the errors above and fix them before uploading to Colab.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
