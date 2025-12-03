# Custom Dataset Guide for UniBias

This guide shows you how to run UniBias experiments on your own datasets, such as Singaporean government sentiment analysis.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Format](#dataset-format)
3. [Loading Your Dataset](#loading-your-dataset)
4. [Running Experiments](#running-experiments)
5. [Complete Example](#complete-example)
6. [Troubleshooting](#troubleshooting)
7. [Tips for Small/Domain-Specific Datasets](#tips-for-smalldomain-specific-datasets)

---

## Prerequisites

- Your dataset in CSV or JSONL format
- Python environment with UniBias dependencies installed
- Llama-2 model access (or GPT-2 for faster testing)

---

## Dataset Format

### Required Columns

Your CSV or JSONL file must have at least two columns:
- **Text column**: Contains the sentences/reviews/documents
- **Label column**: Contains the classification labels (numeric or string)

### Supported Formats

#### Option 1: CSV Format (Recommended)

**Example: `singapore_sentiment.csv`**
```csv
sentence,label
"The government's COVID-19 response was effective and timely.",positive
"I am disappointed with the recent housing policy changes.",negative
"The new infrastructure projects will benefit many citizens.",positive
"Tax increases are hurting middle-class families.",negative
"Education reforms show promising results.",positive
```

#### Option 2: JSONL Format

**Example: `singapore_sentiment.jsonl`**
```jsonl
{"sentence": "The government's COVID-19 response was effective and timely.", "label": "positive"}
{"sentence": "I am disappointed with the recent housing policy changes.", "label": "negative"}
{"sentence": "The new infrastructure projects will benefit many citizens.", "label": "positive"}
```

#### Option 3: Separate Train/Test Files

If you already have separate train and test sets:
- `singapore_sentiment_train.csv`
- `singapore_sentiment_test.csv`

Both should have the same column structure.

---

## Loading Your Dataset

### Step 1: Import the Loader Function

```python
from utils import prepare_dataset_from_csv, find_possible_ids_for_labels
```

### Step 2: Load Your Dataset

#### Single File (80/20 Auto-Split)

```python
prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_from_csv(
    file_path='/path/to/singapore_sentiment.csv',
    text_col='sentence',        # Column name with text
    label_col='label',           # Column name with labels
    num_shot=1,                  # Number of examples per class (0 for zero-shot)
    test_file=None,              # None = auto 80/20 split
    label_map={'positive': 'positive', 'negative': 'negative'},  # Optional: map labels to text
    seed=10                      # Random seed for reproducibility
)
```

#### Separate Train/Test Files

```python
prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_from_csv(
    file_path='/path/to/singapore_sentiment_train.csv',  # Training data for demonstrations
    text_col='sentence',
    label_col='label',
    num_shot=1,
    test_file='/path/to/singapore_sentiment_test.csv',   # Explicit test set
    label_map={'positive': 'positive', 'negative': 'negative'},
    seed=10
)
```

#### Numeric Labels with Custom Text Mapping

If your labels are numeric (0, 1) but you want text labels:

```python
prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_from_csv(
    file_path='/path/to/singapore_sentiment.csv',
    text_col='sentence',
    label_col='label',           # Contains 0, 1
    num_shot=1,
    label_map={0: 'negative', 1: 'positive'},  # Map 0→'negative', 1→'positive'
    seed=10
)
```

### Step 3: Prepare Label Token IDs

After loading, you need to find the token IDs for your label words:

```python
# Define your label words (must match label_map values if provided)
ans_label_list = ['negative', 'positive']

# Convert to format expected by find_possible_ids_for_labels
ans_label_list_formatted = [[label] for label in ans_label_list]

# Find token IDs for each label
gt_ans_ids_list = find_possible_ids_for_labels(ans_label_list_formatted, tokenizer)

print(f"Label token IDs: {gt_ans_ids_list}")
# Example output: [[3480, 6892], [6374, 13907]]  (negative, positive)
```

---

## Running Experiments

### Option A: Standard ICL (Without UniBias)

Run baseline in-context learning without any debiasing:

```python
from evaluation import ICL_evaluation

# Run evaluation
final_acc, all_label_probs, cf = ICL_evaluation(
    model, 
    prompt_list, 
    test_labels, 
    gt_ans_ids_list, 
    dataset_name='singapore_sentiment',  # Just a name for logging
    tokenizer=tokenizer,
    device=device
)

print(f"Baseline Accuracy: {final_acc}")
print(f"Confusion Matrix:\n{cf}")
```

### Option B: With UniBias

Run with UniBias debiasing to remove biased components:

```python
from FFN_manipulate import biased_FFN_identify_and_eliminate
from attention_manipulate import attention_manipulate
from evaluation import ICL_evaluation

# Prepare validation data (same as test in this example, or use a held-out set)
validate_data = prepare_dataset_from_csv(
    file_path='/path/to/singapore_sentiment_validation.csv',  # Or reuse train
    text_col='sentence',
    label_col='label',
    num_shot=0,  # No demonstrations for validation
    seed=42
)[0]  # Just need prompt_list

# Step 1: Identify and suppress biased FFN neurons
biased_FFN_neurons, min_bias_logit_ffn, debias_alpha_ffn = biased_FFN_identify_and_eliminate(
    model, 
    tokenizer, 
    validate_data, 
    ans_label_list, 
    'singapore_sentiment'
)
print(f"Biased FFN neurons: {biased_FFN_neurons}")

# Step 2: Identify and mask biased attention heads
biased_AHs, min_bias_logit_ah, debias_alpha_ah = attention_manipulate(
    model, 
    tokenizer, 
    validate_data, 
    ans_label_list, 
    'singapore_sentiment'
)
print(f"Biased attention heads: {biased_AHs}")

# Step 3: Evaluate with UniBias
final_acc, all_label_probs, cf = ICL_evaluation(
    model, 
    prompt_list, 
    test_labels, 
    gt_ans_ids_list, 
    'singapore_sentiment',
    tokenizer=tokenizer,
    device=device
)

print(f"UniBias Accuracy: {final_acc}")
print(f"Confusion Matrix:\n{cf}")
```

### Option C: Calibration Methods Only

Run calibration methods (CC/DC/PC) without UniBias:

```python
from evaluation import calibration_evaluation

# Run all calibration methods
calibration_evaluation(
    model, 
    all_label_probs,  # From ICL_evaluation output
    gt_ans_ids_list, 
    test_sentences, 
    test_labels, 
    demonstration,
    record_file_path='./results/singapore_sentiment.json',
    dataset_name='singapore_sentiment',
    seed_value=10,
    tokenizer=tokenizer,
    device=device
)
```

---

## Complete Example

### Full Workflow for Singapore Government Sentiment

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import prepare_dataset_from_csv, find_possible_ids_for_labels, write_json
from FFN_manipulate import biased_FFN_identify_and_eliminate
from attention_manipulate import attention_manipulate
from evaluation import ICL_evaluation, calibration_evaluation
import random

# Set seed
seed_value = 10
random.seed(seed_value)
torch.manual_seed(seed_value)

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load your dataset
prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_from_csv(
    file_path='/path/to/singapore_sentiment.csv',
    text_col='sentence',
    label_col='label',
    num_shot=1,
    label_map={'positive': 'positive', 'negative': 'negative'},
    seed=seed_value
)

# Prepare label tokens
ans_label_list = ['negative', 'positive']
gt_ans_ids_list = find_possible_ids_for_labels([[l] for l in ans_label_list], tokenizer)

print(f"Loaded {len(prompt_list)} test samples")
print(f"Example prompt:\n{prompt_list[0][:300]}...\n")

# Setup results
record_file_path = './results/singapore_sentiment.json'
write_json(record_file_path, f'singapore_sentiment seed_value: {seed_value}')

# === BASELINE EVALUATION ===
print("=" * 80)
print("Running BASELINE (no debiasing)...")
print("=" * 80)

baseline_acc, baseline_probs, baseline_cf = ICL_evaluation(
    model, prompt_list, test_labels, gt_ans_ids_list, 
    'singapore_sentiment', tokenizer=tokenizer, device=device
)
write_json(record_file_path, f'Baseline: {baseline_acc} {baseline_cf}')
print(f"\nBaseline Results: {baseline_acc}")
print(f"Confusion Matrix:\n{baseline_cf}\n")

# === UNIBIAS DEBIASING ===
print("=" * 80)
print("Running UniBias debiasing...")
print("=" * 80)

# Prepare validation data (use portion of training for demonstration)
validate_data = prepare_dataset_from_csv(
    file_path='/path/to/singapore_sentiment.csv',
    text_col='sentence',
    label_col='label',
    num_shot=0,
    seed=seed_value + 100  # Different seed for validation
)[0][:100]  # Use first 100 as validation

# Identify biased FFN neurons
biased_FFN_neurons, _, debias_alpha_ffn = biased_FFN_identify_and_eliminate(
    model, tokenizer, validate_data, ans_label_list, 'singapore_sentiment'
)
write_json(record_file_path, f'Biased FFN neurons: {biased_FFN_neurons} alpha={debias_alpha_ffn}')
print(f"Found biased FFN neurons: {biased_FFN_neurons}")

# Identify biased attention heads
biased_AHs, _, debias_alpha_ah = attention_manipulate(
    model, tokenizer, validate_data, ans_label_list, 'singapore_sentiment'
)
write_json(record_file_path, f'Biased attention heads: {biased_AHs} alpha={debias_alpha_ah}')
print(f"Found biased attention heads: {biased_AHs}\n")

# Evaluate with UniBias
unibias_acc, unibias_probs, unibias_cf = ICL_evaluation(
    model, prompt_list, test_labels, gt_ans_ids_list, 
    'singapore_sentiment', tokenizer=tokenizer, device=device
)
write_json(record_file_path, f'UniBias: {unibias_acc} {unibias_cf}')
print(f"\nUniBias Results: {unibias_acc}")
print(f"Confusion Matrix:\n{unibias_cf}\n")

# === CALIBRATION METHODS ===
print("=" * 80)
print("Running calibration methods...")
print("=" * 80)

calibration_evaluation(
    model, unibias_probs, gt_ans_ids_list, test_sentences, test_labels,
    demonstration, record_file_path, 'singapore_sentiment', seed_value,
    tokenizer=tokenizer, device=device
)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Baseline:  {baseline_acc}")
print(f"UniBias:   {unibias_acc}")
print(f"\nResults saved to: {record_file_path}")
```

---

## Troubleshooting

### Issue: "KeyError: 'sentence'"

**Solution**: Check your column names. Update `text_col` parameter:
```python
prepare_dataset_from_csv(
    file_path='your_file.csv',
    text_col='text',  # Use 'text' instead of 'sentence'
    label_col='sentiment'  # Use 'sentiment' instead of 'label'
)
```

### Issue: "No token IDs found for label"

**Solution**: Your label words might not tokenize well. Try different label words:
```python
# Instead of 'positive'/'negative', try:
label_map={'positive': 'good', 'negative': 'bad'}
# or
label_map={'positive': 'yes', 'negative': 'no'}
```

Check tokenization:
```python
print(tokenizer.tokenize('positive'))  # Check how it's tokenized
print(tokenizer.tokenize('good'))
```

### Issue: UniBias finds no biased neurons

**Possible causes**:
1. **Small validation set**: Use at least 50-100 validation samples
2. **Model already calibrated**: Your model might not have strong bias on this dataset
3. **Wrong label mapping**: Verify `ans_label_list` matches your `label_map`

**Solution**: Try calibration methods (CC/DC/PC) instead, which don't require neuron identification.

### Issue: Poor accuracy (< 60%)

**Debugging steps**:
1. Check a few prompts manually:
   ```python
   print(prompt_list[0])  # Inspect format
   ```
2. Test model generation:
   ```python
   inputs = tokenizer(prompt_list[0], return_tensors="pt").to(device)
   outputs = model.generate(**inputs, max_new_tokens=5)
   print(tokenizer.decode(outputs[0]))
   ```
3. Verify label token IDs are correct:
   ```python
   for i, label in enumerate(ans_label_list):
       tokens = gt_ans_ids_list[i]
       decoded = [tokenizer.decode(t) for t in tokens[:3]]
       print(f"{label}: {tokens[:3]} = {decoded}")
   ```

---

## Tips for Small/Domain-Specific Datasets

### 1. Start Simple

- **Use `num_shot=0` or `1`** first to avoid overfitting to demonstrations
- **Test on a subset** (first 50-100 samples) before full evaluation

### 2. Label Word Selection

For domain-specific tasks, choose label words that:
- Appear naturally in the domain
- Tokenize into single or few tokens
- Are unambiguous

**Example for Singaporean sentiment**:
```python
# Generic labels (may not work well)
label_map={'positive': 'positive', 'negative': 'negative'}

# Domain-specific alternatives
label_map={'positive': 'support', 'negative': 'oppose'}
label_map={'positive': 'favorable', 'negative': 'critical'}
```

### 3. UniBias vs Calibration

**Use UniBias when**:
- You have a large model (Llama-2-7B or larger)
- You have at least 100+ validation samples
- You see strong bias in baseline (e.g., predicts one class > 70%)

**Use Calibration (DC/PC) when**:
- Using smaller models (GPT-2)
- Limited validation data
- UniBias finds no/few biased components
- Results from `CUSTOM_DATASET_GUIDE.md` show DC/PC works better

### 4. Validation Set Size

**Recommended sizes**:
- Minimum: 50 samples (25 per class for binary)
- Good: 100-200 samples
- Ideal: 200+ samples

### 5. Monitoring Results

Always compare multiple methods:
```python
print(f"Baseline:  {baseline_acc}")
print(f"UniBias:   {unibias_acc}")
# From calibration_evaluation output:
print(f"CC:        {cc_acc}")
print(f"DC:        {dc_acc}")
print(f"PC:        {pc_acc}")
```

Pick the best-performing method for your dataset.

---

## Example Dataset Formats

### Example 1: Binary Sentiment (Singaporean Government)

**File: `singapore_sentiment.csv`**
```csv
sentence,label
"The government's response to economic challenges has been proactive.",1
"Recent policy changes do not address the concerns of working families.",0
"Infrastructure investments will improve quality of life.",1
"Housing affordability remains a critical issue.",0
"Education initiatives show commitment to future generations.",1
"Healthcare costs continue to rise despite reforms.",0
```

**Usage**:
```python
prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_from_csv(
    file_path='singapore_sentiment.csv',
    text_col='sentence',
    label_col='label',
    num_shot=1,
    label_map={0: 'negative', 1: 'positive'},
    seed=10
)

ans_label_list = ['negative', 'positive']
```

### Example 2: Multi-Class Policy Topics

**File: `singapore_policy.csv`**
```csv
text,category
"New housing grants announced for first-time buyers",housing
"MOE introduces coding curriculum in primary schools",education
"Budget 2024 allocates S$2B for healthcare",healthcare
"MRT expansion to complete by 2030",transport
```

**Usage**:
```python
prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_from_csv(
    file_path='singapore_policy.csv',
    text_col='text',
    label_col='category',
    num_shot=1,
    label_map=None,  # Will use raw category names
    seed=10
)

ans_label_list = ['education', 'healthcare', 'housing', 'transport']
```

### Example 3: JSONL Format

**File: `singapore_sentiment.jsonl`**
```jsonl
{"sentence": "The government handled COVID-19 effectively.", "label": "positive"}
{"sentence": "Income inequality is worsening.", "label": "negative"}
{"sentence": "Smart Nation initiatives drive innovation.", "label": "positive"}
```

**Usage**: Same as CSV, just change file extension.

---

## Quick Start Checklist

- [ ] Prepare your CSV/JSONL with `text_col` and `label_col`
- [ ] Import `prepare_dataset_from_csv` and `find_possible_ids_for_labels`
- [ ] Load dataset and check `prompt_list[0]` format
- [ ] Define `ans_label_list` matching your label words
- [ ] Get `gt_ans_ids_list` using `find_possible_ids_for_labels`
- [ ] Run baseline evaluation first
- [ ] (Optional) Run UniBias if you have 100+ validation samples
- [ ] Compare all methods and pick the best

---

## Additional Resources

- **Main notebook**: `main.py` - Example with SST-2 dataset
- **Colab notebook**: `UniBias_Colab.ipynb` - Ready-to-run in Google Colab
- **Supported datasets**: See `task_labels()` in `utils.py` for examples
- **Paper**: [UniBias: Unifying Bias Mitigation in Language Models](https://arxiv.org/abs/2310.07516)

---

## Questions?

Common questions:

**Q: Can I use this for non-English languages?**  
A: Yes, but ensure your tokenizer supports the language and label words tokenize properly.

**Q: What if I have more than 2 classes?**  
A: The code supports multi-class! Just provide all labels in `ans_label_list` and ensure `label_map` covers all classes.

**Q: Should I use Llama-2 or GPT-2?**  
A: Llama-2-7B works better with UniBias. GPT-2 is faster for testing but may need calibration methods instead.

**Q: How much data do I need?**  
A: Minimum 100 samples total (50/50 split for binary). More is better (500+ recommended).

---

**Happy experimenting!** 🚀
