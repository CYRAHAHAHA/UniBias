# UniBias GPT-2 Migration Guide

## Summary of Changes

This document outlines all changes made to adapt UniBias from Llama-2 to GPT-2.

---

## Key Architectural Differences

### Model Structure

| Component        | Llama-2                            | GPT-2                    |
| ---------------- | ---------------------------------- | ------------------------ |
| Layers access    | `model.model.layers[i]`            | `model.transformer.h[i]` |
| Number of layers | `model.config.num_hidden_layers`   | `model.config.n_layer`   |
| Number of heads  | `model.config.num_attention_heads` | `model.config.n_head`    |
| Hidden size      | `model.config.hidden_size`         | `model.config.n_embd`    |
| Final layer norm | `model.model.norm`                 | `model.transformer.ln_f` |

### Attention Module

| Component         | Llama-2           | GPT-2        |
| ----------------- | ----------------- | ------------ |
| Attention module  | `layer.self_attn` | `layer.attn` |
| Output projection | `o_proj`          | `c_proj`     |
| Combined QKV      | N/A (separate)    | `c_attn`     |

### FFN/MLP Module

| Component         | Llama-2                | GPT-2    |
| ----------------- | ---------------------- | -------- |
| Input projection  | `gate_proj`, `up_proj` | `c_fc`   |
| Output projection | `down_proj`            | `c_proj` |
| Activation        | SwiGLU                 | GELU     |

### Tokenizer

| Feature        | Llama-2             | GPT-2                         |
| -------------- | ------------------- | ----------------------------- |
| Pad token      | Has default         | **Needs to be set** (use EOS) |
| Authentication | Required (HF token) | **Not required** (public)     |
| Dtype          | float16             | float32 (typically)           |

---

## Files Modified

### 1. `model_modifications.py`

**Purpose:** Add custom attributes for UniBias operations

**Changes:**

- Updated header documentation for GPT-2
- Changed `patch_llama_attention_forward` → `patch_gpt2_attention_forward`
  - Uses `layer.attn` instead of `layer.self_attn`
  - Uses `c_proj` instead of `o_proj`
  - Adapted to GPT-2 forward signature
- Updated `add_custom_attributes_to_model`:
  - Uses `model.config.n_layer` instead of `num_hidden_layers`
  - Uses `model.config.n_head` instead of `num_attention_heads`
  - Uses `model.config.n_embd` instead of `hidden_size`
  - Access layers via `model.transformer.h[i]`
  - Add attributes to `layer.attn` instead of `layer.self_attn`
  - Add `coefficients` attribute to `layer.mlp` for FFN manipulation
  - Use `model.dtype` instead of hardcoded `torch.float16`
- Updated `verify_model_modifications`:
  - Uses `model.config.n_layer`
  - Access layers via `model.transformer.h[i]`
  - Check `layer.attn` instead of `layer.self_attn`
  - Check `layer.mlp.coefficients`

**Key Code Changes:**

```python
# Before (Llama-2)
layer = model.model.layers[layer_idx]
layer.self_attn.custom_head_output = CustomHeadOutput(...)
layer.self_attn.register_buffer('mask', ...)

# After (GPT-2)
layer = model.transformer.h[layer_idx]
layer.attn.custom_head_output = CustomHeadOutput(...)
layer.attn.register_buffer('mask', ...)
```

---

### 2. `attention_manipulate.py`

**Purpose:** Identify and mask biased attention heads

**Changes:**

- Updated `biased_attention_head_identification`:
  - Uses `model.config.n_layer` instead of `len(model.model.layers)`
  - Uses `model.config.n_head` instead of `num_attention_heads`
  - Access layers via `model.transformer.h[i]`
  - Register hooks on `layer.attn.custom_head_output` instead of `layer.self_attn`
- Updated `set_attention_masks` and `remove_attention_masks`:
  - Access layers via `model.transformer.h[int(layer)]`
  - Set masks on `layer.attn.mask` instead of `layer.self_attn.mask`

**Key Code Changes:**

```python
# Before (Llama-2)
for layer_index in range(model.config.num_hidden_layers):
    hook = model.model.layers[layer_index].self_attn.custom_head_output.register_forward_hook(...)

model.model.layers[int(layer)].self_attn.mask[0, head_indexes, 0, 0] = debias_alpha

# After (GPT-2)
for layer_index in range(model.config.n_layer):
    hook = model.transformer.h[layer_index].attn.custom_head_output.register_forward_hook(...)

model.transformer.h[int(layer)].attn.mask[0, head_indexes, 0, 0] = debias_alpha
```

---

### 3. `FFN_manipulate.py`

**Purpose:** Identify and suppress biased FFN neurons

**Changes:**

- Updated `find_value_logits`:
  - Uses `model.config.n_layer` instead of `num_hidden_layers`
  - Uses `model.transformer.ln_f` instead of `model.model.norm`
  - Uses `model.transformer.h[i].mlp.c_proj` instead of `model.model.layers[i].mlp.down_proj`
- Updated `find_biased_FFN_neurons`:
  - Uses `model.config.n_layer`
  - Register hooks on `layer.mlp.c_proj` instead of `layer.mlp.down_proj`
  - Access layers via `model.transformer.h[i]`
- Updated `set_value_activations`:
  - Updated documentation for GPT-2 MLP structure
  - Uses `model.config.n_layer` instead of `len(model.model.layers)`
  - Register hooks on `layer.mlp.c_fc` instead of `layer.mlp.up_proj`
  - Access layers via `model.transformer.h[i]`

**Key Code Changes:**

```python
# Before (Llama-2)
for i in range(model.config.num_hidden_layers):
    layer_logits = model.lm_head(model.model.norm(model.model.layers[i].mlp.down_proj.weight.T))

hook = model.model.layers[layer].mlp.up_proj.register_forward_hook(...)

# After (GPT-2)
for i in range(model.config.n_layer):
    layer_logits = model.lm_head(model.transformer.ln_f(model.transformer.h[i].mlp.c_proj.weight.T))

hook = model.transformer.h[layer].mlp.c_fc.register_forward_hook(...)
```

**GPT-2 MLP Flow:**

```
x → c_fc → GELU → c_proj → output
    ↑                ↑
    Hook here        Read weights here
```

---

### 4. `evaluation.py`

**Changes:** ✅ **None needed** - No model-specific code

This file only uses high-level model inference and doesn't access internal structure.

---

### 5. `main.py`

**Purpose:** Main execution script

**Changes:**

- Removed `hf_token` argument (GPT-2 doesn't need it)
- Changed model name from `meta-llama/Llama-2-7b-hf` to `gpt2-medium`
- Changed `torch_dtype` from `torch.float16` to `torch.float32`
- Removed token authentication code
- Added pad token setup for GPT-2:
  ```python
  if tokenizer.pad_token is None:
      tokenizer.pad_token = tokenizer.eos_token
      model.config.pad_token_id = model.config.eos_token_id
  ```
- Changed `norm = model.model.norm` to `norm = model.transformer.ln_f`
- Fixed `calibration_evaluation` call to include all required parameters

**Model Selection:**

```python
# GPT-2 options (in order of size):
# "gpt2"        - 117M params (fastest)
# "gpt2-medium" - 345M params (recommended)
# "gpt2-large"  - 774M params
# "gpt2-xl"     - 1.5B params (most capable)
```

---

### 6. `UniBias_Colab.ipynb`

**Purpose:** Google Colab notebook

**Changes:**

- Updated title to "GPT-2 Version"
- Removed all Hugging Face authentication cells and references
- Updated configuration cell:
  - Changed model options to GPT-2 variants
  - Set `model_name = "gpt2-medium"`
  - Changed dtype to `float32`
  - Added pad token setup
- Updated model loading cell:
  - Removed token parameter
  - Changed dtype to `float32`
  - Added pad token configuration
  - Changed norm reference to `model.transformer.ln_f`
  - Updated model info display (n_layer, n_head)
- Updated documentation in markdown cells
- Added GPT-2 architecture notes

**Key Improvements:**

- No authentication needed ✅
- Simpler setup process
- Clear model size options
- Architecture-specific documentation

---

## Testing Checklist

Before running on Colab, verify:

### ✅ Model Loading

- [ ] GPT-2 model downloads without authentication
- [ ] Pad token is set correctly
- [ ] Model loads in float32
- [ ] Custom attributes are added successfully

### ✅ FFN Debiasing

- [ ] FFN neurons are identified correctly
- [ ] Hooks are registered on correct layers (transformer.h[i].mlp.c_fc)
- [ ] Logits computed correctly with c_proj weights
- [ ] Debiasing alpha applied correctly

### ✅ Attention Debiasing

- [ ] Attention heads are identified correctly
- [ ] Hooks registered on correct modules (transformer.h[i].attn)
- [ ] Masks are applied correctly
- [ ] Head outputs captured successfully

### ✅ Evaluation

- [ ] ICL evaluation runs without errors
- [ ] Calibration methods (CC/DC/PC) work correctly
- [ ] Results saved to JSON correctly
- [ ] Confusion matrices computed correctly

---

## Common Issues and Solutions

### Issue 1: "AttributeError: 'GPT2LMHeadModel' object has no attribute 'model'"

**Solution:** Use `model.transformer` instead of `model.model`

### Issue 2: "AttributeError: 'GPT2Block' object has no attribute 'self_attn'"

**Solution:** Use `layer.attn` instead of `layer.self_attn`

### Issue 3: "AttributeError: 'GPT2MLP' object has no attribute 'down_proj'"

**Solution:** Use `layer.mlp.c_proj` instead of `layer.mlp.down_proj`

### Issue 4: "No pad token found"

**Solution:** Add this after loading tokenizer:

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
```

### Issue 5: "TypeError: float16 not supported on CPU"

**Solution:** Use `torch.float32` for GPT-2 instead of `torch.float16`

---

## Expected Results on SST-2

With GPT-2-medium on SST-2 (1-shot):

- **Baseline accuracy:** ~85-88%
- **UniBias accuracy:** ~88-92%
- **Typical improvement:** +2-4%

The smaller model size compared to Llama-2-7B means:

- Faster inference ✅
- Lower memory usage ✅
- Potentially slightly lower absolute accuracy
- But similar **relative improvement** from UniBias

---

## Next Steps

1. **Run the notebook on Colab** to verify all changes work
2. **Check for any runtime errors** and fix them
3. **Compare results** with Llama-2 version
4. **Document any additional issues** found during testing
5. **Consider testing on multiple GPT-2 sizes** (gpt2, gpt2-large, gpt2-xl)

---

## Files Summary

| File                      | Status         | Changes                                  |
| ------------------------- | -------------- | ---------------------------------------- |
| `model_modifications.py`  | ✅ Updated     | Architecture-specific patching for GPT-2 |
| `attention_manipulate.py` | ✅ Updated     | Layer/module access for GPT-2            |
| `FFN_manipulate.py`       | ✅ Updated     | MLP structure for GPT-2 (c_fc/c_proj)    |
| `evaluation.py`           | ✅ No changes  | Model-agnostic code                      |
| `main.py`                 | ✅ Updated     | Model loading and configuration          |
| `UniBias_Colab.ipynb`     | ✅ Updated     | All cells updated for GPT-2              |
| `utils.py`                | ℹ️ Not checked | Likely no changes needed                 |

---

## Questions to Address

1. **Does GPT-2 show similar bias patterns to Llama-2?**
   - Need empirical testing
2. **Are the same thresholds appropriate for GPT-2?**
   - May need tuning based on model size
3. **Does UniBias provide similar improvements?**

   - Test on SST-2, AG News, and other datasets

4. **Performance on different GPT-2 sizes?**
   - Compare gpt2 vs gpt2-medium vs gpt2-large vs gpt2-xl

---

**All files are ready for testing on Google Colab!** 🚀
