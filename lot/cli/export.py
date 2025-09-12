#!/usr/bin/env python3
"""
CLI entry-point for exporting LoT models to ONNX format with phase information.
This version avoids in-place operations to comply with ONNX static graph requirements.
"""
import argparse
import pathlib
import torch
import yaml
from lot.model.transformer import Transformer

class ExportWrapper(torch.nn.Module):
    """
    Wrapper class that avoids in-place operations for ONNX compatibility.
    Returns logits and the final phase values without updating buffers.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Run the model but avoid in-place operations
        logits, info = self.forward_without_inplace(x)
        
        # Extract phase information from all layers
        layers_info = info["layers"]
        
        # Collect phases without in-place updates
        phase_outputs = []
        for i, layer_info in enumerate(layers_info):
            if "phase" in layer_info:
                phase_outputs.append(layer_info["phase"])
            else:
                # For vanilla attention, create a dummy phase tensor
                num_heads = self.model.blocks[i].attn.num_heads
                phase_outputs.append(torch.zeros(num_heads))
        
        # Stack phases across layers
        phase_outputs = torch.stack(phase_outputs)
        
        return logits, phase_outputs
    
    def forward_without_inplace(self, x):
        """
        Custom forward pass that avoids in-place operations.
        This is a modified version of the transformer's forward method.
        """
        B, T = x.shape
        if T > self.model.pos.size(1):
            # Create new position embeddings without in-place assignment
            new_pos = torch.nn.Parameter(
                torch.empty(1, T, self.model.embed_dim, 
                           device=self.model.pos.device, 
                           dtype=self.model.pos.dtype)
            )
            torch.nn.init.trunc_normal_(new_pos, std=0.02)
            # Use the new positions for this forward pass only
            x_embedded = self.model.embed(x) + new_pos[:, :T, :]
        else:
            x_embedded = self.model.embed(x) + self.model.pos[:, :T, :]
        
        infos = []
        current_x = x_embedded
        
        # Process each block without in-place operations
        for blk in self.model.blocks:
            # Use a modified block forward that avoids in-place ops
            current_x, info = self.block_forward_without_inplace(blk, current_x)
            infos.append(info)
        
        return self.model.head(self.model.norm(current_x)), {"layers": infos}
    
    def block_forward_without_inplace(self, block, x, mask=None):
        """
        Modified block forward that avoids in-place operations.
        """
        # Modified attention forward without in-place phase updates
        h, info = self.attention_forward_without_inplace(block.attn, block.norm1(x), mask)
        x = x + h
        x = x + block.mlp(block.norm2(x))
        return x, info
    
    def attention_forward_without_inplace(self, attn, x, mask=None):
        """
        Modified attention forward that avoids in-place phase updates.
        For LatticeMultiHeadAttention, this calculates phase without updating buffers.
        """
        if hasattr(attn, 'phase') and hasattr(attn, 'intrinsic_freq'):
            # This is a LatticeMultiHeadAttention
            from lot.model.components import kuramoto_step
            
            # Calculate new phase without in-place update
            new_phase = kuramoto_step(
                attn.phase, 
                attn.intrinsic_freq, 
                attn.coupling, 
                attn.nbr_idx, 
                attn.nbr_w
            )
            
            # Use the new phase for this forward pass only
            phase_bias = 0.1 * torch.cos(new_phase).view(attn.num_heads, 1, 1)
            
            # Continue with the rest of the attention computation
            B, T, C = x.shape
            qkv = attn.qkv(x).reshape(B, T, 3, attn.num_heads, attn.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            scores = torch.matmul(q, k.transpose(-2, -1)) * attn.scale
            scores = scores + phase_bias
            
            if mask is not None:
                scores = scores + mask.unsqueeze(0).unsqueeze(0)
                
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = torch.dropout(attn_weights, p=attn.dropout, train=attn.training)
            
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2).reshape(B, T, C)
            
            return attn.out_proj(out), {"attn": attn_weights, "phase": new_phase}
        else:
            # This is a VanillaMultiHeadAttention
            return attn(x, mask=mask)

def main():
    parser = argparse.ArgumentParser(description="Export a LoT checkpoint to ONNX with phase outputs")
    parser.add_argument("checkpoint", type=pathlib.Path, help="Lightning .pt checkpoint")
    parser.add_argument("--config", type=pathlib.Path, required=True, help="Training YAML")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="Output ONNX file path")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length for export")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    # Load config and model architecture
    cfg = yaml.safe_load(args.config.read_text())
    
    # Create model instance
    model = Transformer(**cfg["model"])
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    # Adjust position embeddings if needed
    saved_pos = ckpt["state_dict"]["model.pos"]          # [1, T_ckpt, C]
    T_ckpt = saved_pos.shape[1]
    if T_ckpt > model.pos.shape[1]:
        model.pos = torch.nn.Parameter(
            torch.empty(1, T_ckpt, model.embed_dim, dtype=model.pos.dtype)
        )
        torch.nn.init.trunc_normal_(model.pos, std=0.02)

    # Load state dict
    state = {k.partition("model.")[2]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("model.")}
    model.load_state_dict(state, strict=True)
    model.eval()
    
    # Create wrapper for export
    export_model = ExportWrapper(model)
    
    # Create dummy input
    dummy_input = torch.randint(0, cfg["model"]["vocab_size"], (1, args.seq_len), dtype=torch.long)
    
    # Export to ONNX
    torch.onnx.export(
        export_model,
        dummy_input,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits", "phases"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
            "phases": {0: "num_layers"}
        },
        verbose=False
    )
    
    print(f"Model successfully exported to {args.output}")
    print("Outputs:")
    print("  - logits: (batch_size, sequence_length, vocab_size)")
    print("  - phases: (num_layers, num_heads)")
    
    # Verify the exported model
    try:
        import onnx
        onnx_model = onnx.load(str(args.output))
        onnx.checker.check_model(onnx_model)
        
        # Print model information
        print(f"\nONNX model information:")
        print(f"  - Inputs: {[input.name for input in onnx_model.graph.input]}")
        print(f"  - Outputs: {[output.name for output in onnx_model.graph.output]}")
        print(f"  - Opset version: {onnx_model.opset_import[0].version}")
        
        print("ONNX model check passed!")
    except ImportError:
        print("ONNX package not available, skipping model verification")
    except Exception as e:
        print(f"ONNX model check failed: {e}")

if __name__ == "__main__":
    main()