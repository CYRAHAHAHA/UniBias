# GPT-2 Migration Verification Checklist

## ✅ Pre-Upload Verification (Local)

### File Completeness

- [x] model_modifications.py - Updated for GPT-2
- [x] attention_manipulate.py - Updated for GPT-2
- [x] FFN_manipulate.py - Updated for GPT-2
- [x] evaluation.py - Verified model-agnostic
- [x] main.py - Updated for GPT-2
- [x] utils.py - Should be model-agnostic
- [x] UniBias_Colab.ipynb - Updated for GPT-2
- [x] requirements.txt - Present
- [x] README.md - Present (may need updating)

### Code Consistency Check

Run these checks before uploading:

```bash
# Check for remaining Llama-specific references
grep -r "model\.model\." *.py
grep -r "self_attn" *.py
grep -r "num_hidden_layers" *.py
grep -r "down_proj\|gate_proj\|up_proj" *.py
grep -r "Llama" *.py

# Expected: Only comments/docstrings should mention these
```

## 🧪 Testing on Colab (Step-by-Step)

### Phase 1: Environment Setup

- [ ] Create new Colab notebook or upload existing
- [ ] Set runtime to GPU (Runtime → Change runtime type → GPU)
- [ ] Upload all files to `/content/UniBias/`
- [ ] Run Cell 1 (package installation) - Should complete without errors
- [ ] Run Cell 2 (clone/navigate) - Should navigate to UniBias directory

### Phase 2: Model Loading

- [ ] Run Cell 4 (imports) - All imports should succeed
- [ ] Run Cell 5 (configuration) - Set `dataset_name = 'sst2'`
- [ ] Run Cell 6 (model loading) - Should download gpt2-medium (~1.5 GB)
- [ ] Verify output shows:
  ```
  🔧 Adding custom attributes to GPT-2 model...
     Layers: 24, Heads: 16, Head dim: 64
  ✅ Added custom attributes to 24 layers
  ```

### Phase 3: Dataset Preparation

- [ ] Run Cell 7 (dataset prep) - Should load SST-2 successfully
- [ ] Verify outputs:
  - Test samples: 872
  - Validation samples: ~100
  - Labels: ['negative', 'positive']
  - Example prompt displayed

### Phase 4: UniBias Debiasing

- [ ] Run Cell 8 (UniBias) - This will take 5-10 minutes
- [ ] Watch for progress indicators:
  - "Step 1/2: Identifying biased FFN neurons..."
  - Grid search progress bars
  - "Step 2/2: Identifying biased Attention heads..."
- [ ] Verify outputs:
  - Found 5-15 biased FFN neurons
  - Found 2-5 biased attention heads
  - Debias alpha values printed

### Phase 5: Evaluation

- [ ] Run Cell 9 (evaluation) - Will take 3-5 minutes
- [ ] Verify accuracy is reasonable (75-90%)
- [ ] Check confusion matrix is 2x2
- [ ] Run Cell 9a (baseline comparison) - Optional but recommended
- [ ] Run Cell 9b (debiasing verification) - Should show masks = 0.0
- [ ] Run Cell 10 (calibration) - Optional

### Phase 6: Results Verification

- [ ] Check results file created: `results/sst2.json`
- [ ] Run Cell 11 (view results)
- [ ] Download results file (Cell 12)

## ⚠️ Common Issues & Solutions

### Issue 1: Import Errors

**Symptom:** `ModuleNotFoundError` or `ImportError`
**Solution:**

- Re-run Cell 1 (package installation)
- Restart runtime and try again

### Issue 2: GPU Not Available

**Symptom:** `CUDA not available` or using CPU
**Solution:**

- Runtime → Change runtime type → Hardware accelerator → GPU → Save
- Restart runtime

### Issue 3: Model Download Fails

**Symptom:** Network error or timeout
**Solution:**

- Check internet connection
- Try smaller model first: change `model_name = "gpt2"` in Cell 6
- Re-run cell

### Issue 4: Out of Memory

**Symptom:** `CUDA out of memory` error
**Solution:**

- Use smaller model: `model_name = "gpt2"` or `"gpt2-medium"`
- Runtime → Factory reset runtime
- Reduce batch size if applicable

### Issue 5: Attribute Errors

**Symptom:** `'GPT2Model' object has no attribute 'model'`
**Solution:**

- This means old Llama code is still present
- Verify you uploaded the correct updated files
- Check file contents match the GPT-2 version

### Issue 6: No Biased Components Found

**Symptom:** `biased_FFN_neurons = {}` or empty attention heads
**Solution:**

- This is actually OK! It means the model is already well-calibrated
- Try different random seed
- Try different dataset

### Issue 7: Accuracy Too Low (<60%)

**Symptom:** Final accuracy below 60%
**Solution:**

- Check tokenizer setup (pad_token)
- Verify dataset loaded correctly
- Try without UniBias first (set `Unibias = False`)

## 📊 Expected Results (gpt2-medium on SST-2)

### Biased Components

- **FFN Neurons:** 5-15 neurons across layers 15-23
- **Attention Heads:** 2-5 heads, typically in later layers
- **Debias Alpha:** Usually 0.0 (full suppression)

### Accuracy Ranges

- **Baseline (no UniBias):** 75-85%
- **UniBias:** 80-90%
- **CC/DC Calibration:** 80-90%

### Performance

- **UniBias identification:** 5-10 minutes
- **Evaluation:** 3-5 minutes
- **Total runtime:** 10-20 minutes
- **GPU Memory:** ~4-6 GB

## 📝 Success Criteria

Mark as successful if:

- [ ] No Python errors or crashes
- [ ] Model loads successfully
- [ ] UniBias identifies at least some biased components (or correctly identifies none)
- [ ] Final accuracy is reasonable (>70%)
- [ ] Results file is created and downloadable
- [ ] Baseline comparison shows UniBias effect (if components found)

## 🎯 Next Steps After Verification

If all checks pass:

1. Document actual results in AG News experiments
2. Compare SST-2 results with Llama-2 version
3. Test on other datasets (ag_news, trec)
4. Experiment with different GPT-2 model sizes
5. Write up findings for report

## 📚 Quick Reference

### Model Sizes

```python
model_name = "gpt2"         # 124M params,  ~500MB
model_name = "gpt2-medium"  # 355M params, ~1.5GB (recommended)
model_name = "gpt2-large"   # 774M params, ~3GB
model_name = "gpt2-xl"      # 1.5B params, ~6GB
```

### Dataset Options

```python
dataset_name = 'sst2'     # Binary sentiment (easy)
dataset_name = 'ag_news'  # 4-class news topics (medium)
dataset_name = 'trec'     # 6-class questions (medium)
```

### Debug Mode

To see more detailed output, add print statements:

```python
print(f"Model dtype: {model.dtype}")
print(f"Model device: {next(model.parameters()).device}")
print(f"Config: {model.config}")
```

---

**Remember:** GPT-2 is significantly smaller than Llama-2-7b, so expect slightly lower performance but faster runtime!
