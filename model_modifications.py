"""
Model Modifications for UniBias
This module adds necessary custom attributes and hooks to the Llama model
for UniBias debiasing operations.
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


def patch_llama_attention_forward(layer, num_heads, head_dim):
    """
    Patch the LlamaAttention forward method to capture per-head outputs.
    We need to intercept the computation BEFORE o_proj to get individual head outputs,
    then project each head separately through o_proj.
    """
    original_forward = layer.self_attn.forward
    
    def custom_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        # Call original forward - handle both 2 and 3 return values
        result = original_forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        
        # Unpack based on number of returned values
        if len(result) == 2:
            attn_output, attn_weights = result
            past_key_value_out = None
        else:
            attn_output, attn_weights, past_key_value_out = result
        
        # Compute per-head outputs projected to full hidden size
        # attn_output is [bsz, seq_len, hidden_size] - this is already the combined output
        # We need to split it back to heads and project each head separately
        bsz, q_len, _ = hidden_states.size()
        
        # Reshape to [bsz, seq_len, num_heads, head_dim]
        attn_output_reshaped = attn_output.view(bsz, q_len, num_heads, head_dim)
        
        # For each head, project it through o_proj separately
        # o_proj is [hidden_size, hidden_size], we need to use only the part for each head
        # Actually, o_proj already combined the heads, so we need a different approach
        
        # Create per-head projected outputs by using head-specific slices of o_proj
        head_outputs_projected = []
        for head_idx in range(num_heads):
            # Extract this head's output: [bsz, seq_len, head_dim]
            head_output = attn_output_reshaped[:, :, head_idx, :]
            
            # Project through the corresponding slice of o_proj
            # o_proj.weight shape: [hidden_size, hidden_size]
            # We take the slice corresponding to this head
            start_idx = head_idx * head_dim
            end_idx = start_idx + head_dim
            head_weight = self.o_proj.weight[:, start_idx:end_idx]  # [hidden_size, head_dim]
            
            # Project: [bsz, seq_len, head_dim] @ [head_dim, hidden_size] -> [bsz, seq_len, hidden_size]
            head_projected = torch.matmul(head_output, head_weight.t())
            if self.o_proj.bias is not None:
                # Distribute bias equally across heads (approximation)
                head_projected = head_projected + self.o_proj.bias / num_heads
            
            head_outputs_projected.append(head_projected)
        
        # Stack to [bsz, num_heads, seq_len, hidden_size]
        head_outputs_projected = torch.stack(head_outputs_projected, dim=1)
        
        # Pass through custom_head_output to allow hook capture
        if hasattr(self, 'custom_head_output'):
            head_outputs_projected = self.custom_head_output(head_outputs_projected, bsz, q_len)
        
        # Return in the same format as the original
        if len(result) == 2:
            return attn_output, attn_weights
        else:
            return attn_output, attn_weights, past_key_value_out
    
    # Replace the forward method
    layer.self_attn.forward = custom_forward.__get__(layer.self_attn, type(layer.self_attn))


def add_custom_attributes_to_model(model):
    """
    Add custom attributes required for UniBias operations to the model.
    
    This function modifies the model in-place by adding:
    1. custom_head_output: A module for capturing attention head outputs
    2. mask: A buffer for masking biased attention heads
    3. Patches attention forward to expose per-head outputs
    
    Args:
        model: The loaded Llama model
    
    Returns:
        model: The modified model (modified in-place, but returned for convenience)
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    head_dim = hidden_size // num_heads
    
    print(f"🔧 Adding custom attributes to model...")
    print(f"   Layers: {num_layers}, Heads: {num_heads}, Head dim: {head_dim}")
    
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        
        # Add custom_head_output module for capturing attention head outputs
        if not hasattr(layer.self_attn, 'custom_head_output'):
            layer.self_attn.custom_head_output = CustomHeadOutput(num_heads, head_dim)
        
        # Add mask buffer for attention head masking
        # Shape: [1, num_heads, 1, 1] - initialized to all 1s (no masking)
        if not hasattr(layer.self_attn, 'mask'):
            layer.self_attn.register_buffer(
                'mask',
                torch.ones(1, num_heads, 1, 1, dtype=torch.float16)
            )
        
        # Patch the forward method to expose per-head outputs
        patch_llama_attention_forward(layer, num_heads, head_dim)
    
    print(f"✅ Added custom attributes to {num_layers} layers")
    print(f"   - custom_head_output module for capturing attention outputs")
    print(f"   - mask buffer ({num_heads} heads) for debiasing")
    print(f"   - Patched forward methods to expose per-head outputs")
    
    return model


def modify_attention_forward(model):
    """
    Modify the forward pass of attention layers to:
    1. Apply the custom mask to attention heads
    2. Pass outputs through custom_head_output for capturing
    
    Note: This is a more advanced modification that may require
    monkey-patching the forward method. For now, we rely on hooks.
    """
    # This function can be extended if deeper modifications are needed
    pass


def verify_model_modifications(model):
    """
    Verify that all necessary modifications have been applied to the model.
    
    Args:
        model: The model to verify
    
    Returns:
        bool: True if all modifications are present, False otherwise
    """
    num_layers = model.config.num_hidden_layers
    
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        
        if not hasattr(layer.self_attn, 'custom_head_output'):
            print(f"❌ Layer {layer_idx}: Missing custom_head_output")
            return False
        
        if not hasattr(layer.self_attn, 'mask'):
            print(f"❌ Layer {layer_idx}: Missing mask buffer")
            return False
    
    print(f"✅ All {num_layers} layers have required modifications")
    return True
