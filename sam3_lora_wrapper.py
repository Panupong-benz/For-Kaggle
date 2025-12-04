"""
LoRA Wrapper for SAM3 Official Training Pipeline
Integrates LoRA into the official SAM3 training from facebookresearch/sam3

This module wraps SAM3 models with LoRA layers and integrates with the
official Hydra-based training pipeline.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Set
import math


class LoRALayer(nn.Module):
    """
    LoRA layer for efficient fine-tuning.

    LoRA decomposes weight updates into low-rank matrices:
    h = W0*x + (B*A)*x * (alpha/rank)

    where W0 is frozen and only A, B are trainable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA: (B @ A @ x) * scaling"""
        return (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Freeze original weights
        self.linear = original_linear
        for param in self.linear.parameters():
            param.requires_grad = False

        # Add LoRA
        self.lora = LoRALayer(
            in_features=original_linear.in_features,
            out_features=original_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: frozen linear + LoRA"""
        return self.linear(x) + self.lora(x)


def apply_lora_to_sam3(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
    apply_to_vision_encoder: bool = True,
    apply_to_text_encoder: bool = True,
    apply_to_detr_encoder: bool = True,
    apply_to_detr_decoder: bool = True,
) -> nn.Module:
    """
    Apply LoRA to SAM3 model components.

    Args:
        model: SAM3 model (Sam3ImageModel)
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
        target_modules: Module names to target (default: attention projections)
        apply_to_vision_encoder: Apply to vision backbone
        apply_to_text_encoder: Apply to text encoder
        apply_to_detr_encoder: Apply to detector encoder
        apply_to_detr_decoder: Apply to detector decoder

    Returns:
        Model with LoRA applied
    """

    if target_modules is None:
        # Default: attention projections
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

    target_set = set(target_modules)
    lora_count = 0

    def should_apply_lora(name: str) -> bool:
        """Check if LoRA should be applied to this module."""

        # Component-level filtering
        if "backbone.vision_backbone" in name or "backbone.trunk" in name:
            if not apply_to_vision_encoder:
                return False

        if "backbone.language_backbone" in name or "text_encoder" in name:
            if not apply_to_text_encoder:
                return False

        if "detector.encoder" in name:
            if not apply_to_detr_encoder:
                return False

        if "detector.decoder" in name or "query_decoder" in name:
            if not apply_to_detr_decoder:
                return False

        # Module name filtering
        module_basename = name.split('.')[-1]
        return module_basename in target_set

    # Apply LoRA to matching modules
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and should_apply_lora(name):
            # Get parent module
            *parent_path, attr_name = name.split('.')
            parent = model
            for p in parent_path:
                parent = getattr(parent, p)

            # Replace with LoRA
            lora_linear = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            setattr(parent, attr_name, lora_linear)
            lora_count += 1

    print(f"Applied LoRA to {lora_count} modules")
    return model


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters for optimizer."""
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend([module.lora_A, module.lora_B])
    return lora_params


def save_lora_weights(model: nn.Module, save_path: str):
    """Save only LoRA weights (not full model)."""
    lora_state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state[f"{name}.lora_A"] = module.lora_A.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.cpu()

    torch.save(lora_state, save_path)
    print(f"Saved {len(lora_state)} LoRA weights to {save_path}")
    return len(lora_state)


def load_lora_weights(model: nn.Module, load_path: str):
    """Load LoRA weights into model."""
    lora_state = torch.load(load_path, map_location='cpu')

    # Load into model
    missing, unexpected = model.load_state_dict(lora_state, strict=False)

    print(f"Loaded LoRA weights from {load_path}")
    print(f"  Loaded: {len(lora_state)} tensors")
    if missing:
        print(f"  Missing: {len(missing)} keys")
    if unexpected:
        print(f"  Unexpected: {len(unexpected)} keys")


def count_parameters(model: nn.Module) -> dict:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total,
        "trainable": trainable,
        "percentage": 100.0 * trainable / total if total > 0 else 0.0,
    }


# ============================================================================
# Integration with SAM3 Training
# ============================================================================

def wrap_sam3_model_with_lora(
    model_builder_fn,
    lora_config: dict,
):
    """
    Wrapper for SAM3 model builder that applies LoRA.

    Use this in Hydra config:
    ```yaml
    trainer:
      model:
        _target_: sam3_lora_wrapper.wrap_sam3_model_with_lora
        model_builder_fn:
          _target_: sam3.model_builder.build_sam3_image_model
          bpe_path: ${paths.bpe_path}
          device: cpus
          eval_mode: false
        lora_config:
          rank: 8
          alpha: 16
          dropout: 0.0
          target_modules: ["q_proj", "k_proj", "v_proj", "out_proj"]
          apply_to_vision_encoder: true
          apply_to_text_encoder: true
          apply_to_detr_encoder: true
          apply_to_detr_decoder: true
    ```

    Args:
        model_builder_fn: Function that builds the base SAM3 model
        lora_config: LoRA configuration dict

    Returns:
        SAM3 model with LoRA applied
    """
    # Build base model
    model = model_builder_fn

    # Apply LoRA
    model = apply_lora_to_sam3(
        model,
        rank=lora_config.get("rank", 8),
        alpha=lora_config.get("alpha", 16),
        dropout=lora_config.get("dropout", 0.0),
        target_modules=lora_config.get("target_modules"),
        apply_to_vision_encoder=lora_config.get("apply_to_vision_encoder", True),
        apply_to_text_encoder=lora_config.get("apply_to_text_encoder", True),
        apply_to_detr_encoder=lora_config.get("apply_to_detr_encoder", True),
        apply_to_detr_decoder=lora_config.get("apply_to_detr_decoder", True),
    )

    # Print parameter stats
    stats = count_parameters(model)
    print(f"\nParameter Statistics:")
    print(f"  Total: {stats['total']:,}")
    print(f"  Trainable: {stats['trainable']:,}")
    print(f"  Percentage: {stats['percentage']:.3f}%")

    return model
