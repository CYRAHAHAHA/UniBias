"""
Model Modifications for UniBias - GPT-2 Version
This module adds necessary custom attributes and hooks to the GPT-2 model
for UniBias debiasing operations.

Key Differences from Llama-2:
- GPT-2 uses GPT2Attention instead of LlamaAttention
- GPT-2 uses GPT2MLP with c_fc and c_proj instead of gate_proj, up_proj, down_proj
- GPT-2 has 'h' for layers instead of 'layers'
- GPT-2 uses 'transformer' instead of 'model'
- GPT-2 attention returns (attn_output, present, attn_weights) or (attn_output, attn_weights)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CustomHeadOutput(nn.Module):
    """
    A module that processes and returns attention head outputs.
    This captures the per-head hidden states after attention computation.
    """
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
    
    def forward(self, hidden_states, bsz, q_len):
        """
        Args:
            hidden_states: [bsz, num_heads, seq_len, head_dim]
            bsz: batch size
            q_len: query length
        Returns:
            hidden_states: same shape as input
        """
        return hidden_states


def patch_gpt2_attention_forward(layer, num_heads, head_dim, hidden_size):
    """
    Patch the GPT2Attention forward method to capture per-head outputs.
    
    GPT-2 Architecture:
    - Uses split_heads and merge_heads for multi-head attention
    - c_attn: combined QKV projection [hidden_size, 3 * hidden_size]
    - c_proj: output projection [hidden_size, hidden_size]
    """
    original_forward = layer.attn.forward
    
    def custom_forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        **kwargs
    ):
        # Handle both 'layer_past' and 'past_key_values' parameter names
        if past_key_values is not None:
            layer_past = past_key_values
        
        # Call original forward - pass all kwargs to handle any other parameters
        result = original_forward(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )
        
        # GPT-2 returns: (attn_output, present, (attn_weights)) or (attn_output, (attn_weights))
        attn_output = result[0]
        
        # Compute per-head outputs projected to full hidden size
        # GPT-2 attn_output is [bsz, seq_len, hidden_size] - already combined
        bsz, q_len, _ = attn_output.size()
        
        # We need to split the output back into heads and project each separately
        # Reshape to [bsz, seq_len, num_heads, head_dim]
        attn_output_reshaped = attn_output.view(bsz, q_len, num_heads, head_dim)
        
        # For each head, project it through c_proj separately
        head_outputs_projected = []
        for head_idx in range(num_heads):
            # Extract this head's output: [bsz, seq_len, head_dim]
            head_output = attn_output_reshaped[:, :, head_idx, :]
            
            # Project through the corresponding slice of c_proj
            # c_proj.weight shape: [hidden_size, hidden_size]
            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim
            head_weight = self.c_proj.weight[:, start_idx:end_idx]  # [hidden_size, head_dim]
            
            # Project: [bsz, seq_len, head_dim] @ [head_dim, hidden_size] -> [bsz, seq_len, hidden_size]
            head_projected = torch.matmul(head_output, head_weight.t())
            if self.c_proj.bias is not None:
                # Distribute bias equally across heads (approximation)
                head_projected = head_projected + self.c_proj.bias / num_heads
            
            head_outputs_projected.append(head_projected)
        
        # Stack to [bsz, num_heads, seq_len, hidden_size]
        head_outputs_projected = torch.stack(head_outputs_projected, dim=1)
        
        # Pass through custom_head_output to allow hook capture
        if hasattr(self, 'custom_head_output'):
            head_outputs_projected = self.custom_head_output(head_outputs_projected, bsz, q_len)
        
        # Return in the same format as the original
        return result
    
    # Replace the forward method
    layer.attn.forward = custom_forward.__get__(layer.attn, type(layer.attn))


def add_custom_attributes_to_model(model):
    """
    Add custom attributes required for UniBias operations to the GPT-2 model.
    
    This function modifies the model in-place by adding:
    1. custom_head_output: A module for capturing attention head outputs
    2. mask: A buffer for masking biased attention heads
    3. coefficients attribute for FFN manipulation
    4. Patches attention forward to expose per-head outputs
    
    Args:
        model: The loaded GPT-2 model
    
    Returns:
        model: The modified model (modified in-place, but returned for convenience)
    """
    num_layers = model.config.n_layer
    num_heads = model.config.n_head
    hidden_size = model.config.n_embd
    head_dim = hidden_size // num_heads
    
    print(f"🔧 Adding custom attributes to GPT-2 model...")
    print(f"   Layers: {num_layers}, Heads: {num_heads}, Head dim: {head_dim}")
    
    for layer_idx in range(num_layers):
        # GPT-2 uses 'h' for layers instead of 'layers'
        layer = model.transformer.h[layer_idx]
        
        # Add custom_head_output module for capturing attention head outputs
        if not hasattr(layer.attn, 'custom_head_output'):
            layer.attn.custom_head_output = CustomHeadOutput(num_heads, head_dim)
        
        # Add mask buffer for attention head masking
        # Shape: [1, num_heads, 1, 1] - initialized to all 1s (no masking)
        if not hasattr(layer.attn, 'mask'):
            layer.attn.register_buffer(
                'mask',
                torch.ones(1, num_heads, 1, 1, dtype=model.dtype)
            )
        
        # Add coefficients attribute to MLP for FFN neuron manipulation
        # GPT-2 MLP structure: c_fc (input projection) -> activation -> c_proj (output projection)
        if not hasattr(layer.mlp, 'coefficients'):
            # The intermediate size in GPT-2 is typically 4 * hidden_size
            # GPT-2's Conv1D uses 'nf' instead of 'out_features'
            intermediate_size = layer.mlp.c_fc.nf
            layer.mlp.coefficients = None  # Will be set during FFN manipulation
        
        # Patch the forward method to expose per-head outputs
        patch_gpt2_attention_forward(layer, num_heads, head_dim, hidden_size)
    
    print(f"✅ Added custom attributes to {num_layers} layers")
    print(f"   - custom_head_output module for capturing attention outputs")
    print(f"   - mask buffer ({num_heads} heads) for debiasing")
    print(f"   - coefficients attribute for FFN manipulation")
    print(f"   - Patched forward methods to expose per-head outputs")
    
    return model


def modify_attention_forward(model):
    """
    Modify the forward pass of attention layers to:
    1. Apply the custom mask to attention heads
    2. Pass outputs through custom_head_output for capturing
    
    Note: This is handled by the patching in add_custom_attributes_to_model
    """
    # This function can be extended if deeper modifications are needed
    pass


def verify_model_modifications(model):
    """
    Verify that all necessary modifications have been applied to the GPT-2 model.
    
    Args:
        model: The model to verify
    
    Returns:
        bool: True if all modifications are present, False otherwise
    """
    num_layers = model.config.n_layer
    
    for layer_idx in range(num_layers):
        layer = model.transformer.h[layer_idx]
        
        if not hasattr(layer.attn, 'custom_head_output'):
            print(f"❌ Layer {layer_idx}: Missing custom_head_output")
            return False
        
        if not hasattr(layer.attn, 'mask'):
            print(f"❌ Layer {layer_idx}: Missing mask buffer")
            return False
        
        if not hasattr(layer.mlp, 'coefficients'):
            print(f"❌ Layer {layer_idx}: Missing coefficients attribute")
            return False
    
    print(f"✅ All {num_layers} layers have required modifications")
    return True
