# Implementation Comparison: main.py vs UniBias_Colab.ipynb

## 🔧 CRITICAL FIXES APPLIED

### **Fix 1: Missing `num_heads` Attribute**

`AttributeError: 'LlamaAttention' object has no attribute 'num_heads'`

Updated `attention_manipulate.py` line 60:

```python
# ❌ Old - breaks with newer transformers
NB_HEADS = model.model.layers[0].self_attn.num_heads

# ✅ New - compatible across versions
NB_HEADS = model.config.num_attention_heads
```

### **Fix 2: Missing `custom_head_output` Attribute**

`AttributeError: 'LlamaAttention' object has no attribute 'custom_head_output'`

**Root Cause**: The UniBias algorithm requires custom modifications to the Llama model:

- `custom_head_output`: A module for capturing attention head outputs
- `mask`: A buffer for masking biased attention heads

**Solution**: Created `model_modifications.py` with `add_custom_attributes_to_model()` function that:

1. Adds `CustomHeadOutput` module to each attention layer
2. Registers mask buffers for attention head debiasing
3. Must be called **immediately after** loading the model

**Updated Code** (both main.py and Colab):

```python
# After model loading
add_custom_attributes_to_model(model)
```

---

## ✅ CORE ALGORITHM - IDENTICAL

The **core UniBias algorithm logic is 100% identical** between both implementations. All critical computations, model manipulations, and evaluation steps are the same.

---

## 📊 Detailed Comparison

### 1. **Configuration & Parameters**

| Aspect           | main.py (Local)            | UniBias_Colab.ipynb (Colab) | Status  |
| ---------------- | -------------------------- | --------------------------- | ------- |
| **seed_value**   | 10 (via argparse)          | 10 (hardcoded)              | ✅ SAME |
| **dataset_name** | 'sst2' (via argparse)      | 'sst2' (hardcoded)          | ✅ SAME |
| **format_index** | None (via argparse)        | None (hardcoded)            | ✅ SAME |
| **order_index**  | None (via argparse)        | None (hardcoded)            | ✅ SAME |
| **num_shot**     | 1 (via argparse)           | 1 (hardcoded)               | ✅ SAME |
| **Unibias**      | True (via argparse)        | True (hardcoded)            | ✅ SAME |
| **Calibration**  | True (via argparse)        | True (hardcoded)            | ✅ SAME |
| **model_name**   | "meta-llama/Llama-2-7b-hf" | "meta-llama/Llama-2-7b-hf"  | ✅ SAME |

**Difference:** Local uses command-line arguments for flexibility; Colab uses cell variables for easier editing in notebooks. Both use the same default values.

---

### 2. **Random Seed Setting**

| Implementation | Code                                                                                                     |
| -------------- | -------------------------------------------------------------------------------------------------------- |
| **main.py**    | `random.seed(seed_value)`                                                                                |
| **Colab**      | `random.seed(seed_value)`<br>`torch.manual_seed(seed_value)`<br>`torch.cuda.manual_seed_all(seed_value)` |

**⚠️ MINOR DIFFERENCE:**

- Colab has **additional** `torch.manual_seed()` and `torch.cuda.manual_seed_all()` for more complete reproducibility
- This is actually **BETTER** in Colab - ensures PyTorch operations are deterministic
- **Impact:** Potentially slightly different random initializations in PyTorch operations, but shouldn't affect final results significantly since the main randomness comes from the dataset preparation

---

### 3. **CUDA Device Selection**

| Implementation | Code                                                                                     | Notes                                            |
| -------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------ |
| **main.py**    | `os.environ["CUDA_VISIBLE_DEVICES"]=cuda_device_id`<br>`device = torch.device("cuda:0")` | Sets specific GPU, always assumes GPU available  |
| **Colab**      | `device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`                | Auto-detects GPU availability, falls back to CPU |

**⚠️ MINOR DIFFERENCE:**

- main.py assumes GPU is always available
- Colab checks first with `torch.cuda.is_available()`
- **Impact:** Colab is more robust, but both assume GPU in practice

---

### 4. **Model Authentication**

| Implementation | Authentication Method                                                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **main.py**    | `hf_token = args.hf_token or os.getenv("HF_TOKEN")`                                                                                                    |
| **Colab**      | Uses `huggingface_hub.login()` with multiple fallbacks:<br>1. Colab secrets (`userdata.get('HF_TOKEN')`)<br>2. Environment variable<br>3. Manual input |

**⚠️ MINOR DIFFERENCE:**

- Colab uses the official `huggingface_hub.login()` method (modern approach)
- main.py passes token directly to model loading (older approach)
- **Impact:** Both work correctly, Colab method is slightly more secure

---

### 5. **Model Caching Directory**

| Implementation | Cache Directory                                 |
| -------------- | ----------------------------------------------- |
| **main.py**    | `./models` (relative to script location)        |
| **Colab**      | `./models` (local) or `/content/models` (Colab) |

**✅ IDENTICAL** - Both use `./models` when running locally

---

### 6. **Model Loading Logic**

Both implementations use **IDENTICAL** model loading logic:

```python
# Check cache first
if os.path.exists(model_path) and os.listdir(model_path):
    # Load from cache
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
else:
    # Download and cache
    tokenizer = AutoTokenizer.from_pretrained(model_name, ...)
    model = AutoModelForCausalLM.from_pretrained(model_name, ...)
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
```

**✅ IDENTICAL**

---

### 7. **Dataset Preparation**

Both implementations use **IDENTICAL** logic:

```python
if not order_index and not format_index:
    prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_test(...)
    validate_data = prepare_dataset_validate(...)
if format_index:
    prompt_list, test_labels, demonstration, test_sentences, ans_label_list = gen_test_data_format(...)
    validate_data = gen_validate_data_format(...)
if order_index:
    prompt_list, test_labels, demonstration, test_sentences, rand_example_sample_index_order = gen_test_data_order(...)
    validate_data = gen_validate_data_order(...)

ans_label_list = task_labels(dataset_name)
gt_ans_ids_list = find_possible_ids_for_labels(ans_label_list, tokenizer)
```

**✅ IDENTICAL**

---

### 8. **UniBias Algorithm - CRITICAL SECTION**

Both implementations use **IDENTICAL** code:

```python
if Unibias:
    # Identify and eliminate biased FFN neurons
    biased_FFN_neurons, min_bias_label_logit, debias_alpha_value = biased_FFN_identify_and_eliminate(
        model, tokenizer, validate_data, ans_label_list, dataset_name
    )
    write_json(record_file_path, "biased FFN neurons:" + str(biased_FFN_neurons) + str(debias_alpha_value))
    write_json(record_file_path, debias_alpha_value)

    # Identify and eliminate biased Attention heads
    biased_AHs, min_bias_label_logit, debias_alpha_value = attention_manipulate(
        model, tokenizer, validate_data, ans_label_list, dataset_name
    )
    write_json(record_file_path, "biased attention heads:" + str(biased_AHs) + str(debias_alpha_value))
    write_json(record_file_path, debias_alpha_value)
```

**✅ 100% IDENTICAL** - This is the core algorithm, completely unchanged

---

### 9. **Evaluation**

Both implementations use **IDENTICAL** code:

```python
final_acc, all_label_probs, cf = ICL_evaluation(
    model, prompt_list, test_labels, gt_ans_ids_list, dataset_name
)

if Unibias:
    write_json(record_file_path, 'Unibias: ' + final_acc + str(cf))
else:
    write_json(record_file_path, final_acc + str(cf))

if Calibration:
    calibration_evaluation(
        model, all_label_probs, gt_ans_ids_list,
        test_sentences, test_labels, demonstration
    )
```

**✅ 100% IDENTICAL** - Evaluation logic is completely the same

---

### 10. **Results Storage**

| Implementation | Results Path                    |
| -------------- | ------------------------------- |
| **main.py**    | `./results/{dataset_name}.json` |
| **Colab**      | `./results/{dataset_name}.json` |

**✅ IDENTICAL**

---

## 🔍 Summary of Differences

### Non-Critical Differences (Won't Affect Results):

1. **Input Method**: Command-line args (main.py) vs. cell variables (Colab)
2. **Random Seeds**: Colab sets additional PyTorch seeds (actually better)
3. **Authentication**: Colab uses modern `login()` method vs. direct token passing
4. **GPU Detection**: Colab checks availability, main.py assumes it exists
5. **Progress Display**: Colab has emoji-rich output for better UX
6. **Structure**: main.py is single script, Colab is organized in cells

### Critical Similarities (Ensure Identical Results):

✅ **Same model**: Llama-2-7b-hf  
✅ **Same dataset**: sst2  
✅ **Same random seed**: 10  
✅ **Same hyperparameters**: num_shot=1, etc.  
✅ **Same UniBias algorithm**: Identical function calls  
✅ **Same evaluation**: Identical metrics  
✅ **Same model precision**: torch.float16  
✅ **Same device placement**: device_map="auto"

---

## ⚠️ Potential Sources of Different Results

If you're seeing different outputs, it could be due to:

### 1. **PyTorch Random Seeds** (Most Likely)

- **Issue**: main.py doesn't set `torch.manual_seed()` or `torch.cuda.manual_seed_all()`
- **Impact**: Different random operations in PyTorch (dropout, layer initialization if re-running)
- **Solution**: Add these lines to main.py after line 28:
  ```python
  random.seed(seed_value)
  torch.manual_seed(seed_value)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed_value)
  ```

### 2. **Different GPU Hardware**

- **Issue**: Different GPU types (local vs Colab) may have slight numerical differences
- **Impact**: Floating-point precision differences in fp16 operations
- **Normal**: Differences in 4th-5th decimal place are expected

### 3. **Library Versions**

- **Issue**: If local environment has different package versions
- **Check**: Compare your local `pip list` with requirements.txt
- **Solution**: Recreate virtual environment with exact versions

### 4. **Model Checkpoint**

- **Issue**: If models were downloaded at different times
- **Unlikely**: Hugging Face checksums ensure identical weights
- **Verify**: Compare model hashes

### 5. **Dataset Order**

- **Issue**: If dataset loading has any non-deterministic ordering
- **Check**: The `prepare_dataset_test()` function should use the random seed
- **Verify**: Print first few `test_sentences` in both environments

---

## ✅ Conclusion

**The implementations are algorithmically identical.**

The Colab notebook is essentially the same code from main.py, just:

- Reorganized into cells for better notebook experience
- With additional PyTorch random seeding (improvement)
- With better user interface elements
- With environment detection for flexibility

**Any differences in results would be due to:**

1. Missing PyTorch random seeds in main.py (fix recommended above)
2. Hardware/numerical precision differences (expected, minor)
3. Different package versions (check requirements.txt)

**The core UniBias algorithm is 100% preserved.**
