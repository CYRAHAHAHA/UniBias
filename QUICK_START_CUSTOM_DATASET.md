# Quick Start: Using Custom Datasets with UniBias

This is a minimal working example for using your Singapore sentiment dataset with UniBias.

## Step 1: Prepare Your CSV File

Create a file `singapore_sentiment.csv`:

```csv
sentence,label
"The government's COVID-19 response was effective and timely.",1
"I am disappointed with the recent housing policy changes.",0
"The new infrastructure projects will benefit many citizens.",1
"Housing affordability remains a critical issue.",0
"Education initiatives show commitment to future generations.",1
"Healthcare costs continue to rise despite reforms.",0
```

## Step 2: Run with Custom Dataset

### Option A: Simple Baseline (No Debiasing)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import prepare_dataset_from_csv, find_possible_ids_for_labels
from evaluation import ICL_evaluation

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
    file_path='singapore_sentiment.csv',
    text_col='sentence',
    label_col='label',
    num_shot=1,
    label_map={0: 'negative', 1: 'positive'},
    seed=10
)

# Prepare label tokens
ans_label_list = ['negative', 'positive']
gt_ans_ids_list = find_possible_ids_for_labels([[l] for l in ans_label_list], tokenizer)

# Run evaluation
accuracy, all_label_probs, confusion_mat = ICL_evaluation(
    model,
    prompt_list,
    test_labels,
    gt_ans_ids_list,
    'singapore_sentiment',
    tokenizer=tokenizer,
    device=device
)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_mat}")
```

### Option B: With UniBias (Full Pipeline)

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import prepare_dataset_from_csv, find_possible_ids_for_labels
from FFN_manipulate import biased_FFN_identify_and_eliminate
from attention_manipulate import attention_manipulate
from evaluation import ICL_evaluation, calibration_evaluation
from model_modifications import add_custom_attributes_to_model

# Load model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# IMPORTANT: Add custom attributes for UniBias
add_custom_attributes_to_model(model)

# Load your dataset
prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_from_csv(
    file_path='singapore_sentiment.csv',
    text_col='sentence',
    label_col='label',
    num_shot=1,
    label_map={0: 'negative', 1: 'positive'},
    seed=10
)

# Prepare label tokens
ans_label_list = ['negative', 'positive']
gt_ans_ids_list = find_possible_ids_for_labels([[l] for l in ans_label_list], tokenizer)

print(f"Loaded {len(prompt_list)} test samples")

# Prepare validation data for bias identification
validate_data = prepare_dataset_from_csv(
    file_path='singapore_sentiment.csv',
    text_col='sentence',
    label_col='label',
    num_shot=0,  # No demonstrations for validation
    seed=42
)[0][:100]  # Use first 100 prompts

# Step 1: Identify and suppress biased FFN neurons
print("Identifying biased FFN neurons...")
biased_FFN_neurons, _, debias_alpha_ffn = biased_FFN_identify_and_eliminate(
    model, tokenizer, validate_data, ans_label_list, 'singapore_sentiment'
)
print(f"Biased FFN neurons: {biased_FFN_neurons}")

# Step 2: Identify and mask biased attention heads
print("Identifying biased attention heads...")
biased_AHs, _, debias_alpha_ah = attention_manipulate(
    model, tokenizer, validate_data, ans_label_list, 'singapore_sentiment'
)
print(f"Biased attention heads: {biased_AHs}")

# Step 3: Evaluate with UniBias
print("Evaluating with UniBias...")
accuracy, all_label_probs, confusion_mat = ICL_evaluation(
    model,
    prompt_list,
    test_labels,
    gt_ans_ids_list,
    'singapore_sentiment',
    tokenizer=tokenizer,
    device=device
)

print(f"\nUniBias Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{confusion_mat}")

# Step 4: Run calibration methods
print("\nRunning calibration methods...")
calibration_evaluation(
    model,
    all_label_probs,
    gt_ans_ids_list,
    test_sentences,
    test_labels,
    demonstration,
    record_file_path='./singapore_sentiment_results.json',
    dataset_name='singapore_sentiment',
    seed_value=10,
    tokenizer=tokenizer,
    device=device
)
```

## Step 3: Run via Command Line (After Modifying main.py)

If you want to use the command-line interface, add this to `main.py` after line 100:

```python
def main():
    # ... existing code ...

    # Add custom dataset support
    if dataset_name not in ['sst2', 'sst5', 'trec', 'mnli', 'ag_news', 'cr', 'mr', 'copa', 'rte', 'wic', 'arc', 'mmlu']:
        # Assume it's a custom CSV file path
        prompt_list, test_labels, demonstration, test_sentences = prepare_dataset_from_csv(
            file_path=dataset_name,  # Treat dataset_name as file path
            text_col='sentence',
            label_col='label',
            num_shot=num_shot
        )
        validate_data = prepare_dataset_from_csv(
            file_path=dataset_name,
            text_col='sentence',
            label_col='label',
            num_shot=0,
            seed=seed_value + 100
        )[0][:100]

        # You'll need to specify ans_label_list manually or derive from data
        ans_label_list = ['negative', 'positive']  # Adjust for your labels
    elif not order_index and not format_index:
        # ... existing built-in dataset code ...
```

Then run:

```bash
python main.py --dataset_name singapore_sentiment.csv --num_shot 1 --seed 10
```

## Common Issues

### Issue 1: "Could not find answer location"

**Solution**: Make sure your prompt format uses "Sentiment:" (this is automatic with `prepare_dataset_from_csv`)

### Issue 2: No biased neurons found

**Solution**: This is normal for some datasets. Use calibration methods (CC/DC/PC) instead.

### Issue 3: Low accuracy

**Solution**:

1. Try different label words in `label_map`
2. Increase `num_shot` (try 2 or 4)
3. Check if label tokens are valid:
   ```python
   for i, label in enumerate(ans_label_list):
       print(f"{label}: {tokenizer.decode(gt_ans_ids_list[i][0])}")
   ```

## Performance Expectations

Based on SST-2 results:

- **Llama-2-7B + UniBias**: ~95% accuracy ✅
- **GPT-2 + UniBias**: ~51% accuracy (fails) ❌
- **GPT-2 + DC calibration**: ~79% accuracy ✅

**Recommendation**: Use Llama-2 for best results with UniBias.
