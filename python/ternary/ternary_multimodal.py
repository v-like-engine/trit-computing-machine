"""
Ternary Multimodal Agent - Vision + Language with Ternary Weights.

Combines ternary CNNs for vision encoding with ternary GPT for
language generation to create multimodal capabilities:
- Image captioning
- Visual question answering (VQA)
- Image-to-text generation
- Multimodal reasoning
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from .ternary_gpt import TernaryGPT, TernaryGPTConfig

# Try to import CNN and neural components
try:
    from .cnn_models import TernaryResNet
    _has_cnn = True
except ImportError:
    _has_cnn = False
    TernaryResNet = None

# Always use the transformer module's TernaryLinear (handles both APIs)
from .ternary_transformer import TernaryLinear, TernaryLayerNorm


class TernaryVisionEncoder:
    """
    Vision encoder using ternary CNN.

    Extracts visual features from images for multimodal processing.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
        threshold: float = 0.3
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.threshold = threshold

        # Number of patches
        self.num_patches = (image_size // patch_size) ** 2

        # Use ResNet as backbone (if available)
        if _has_cnn:
            from .cnn_models import create_ternary_resnet18
            self.backbone = create_ternary_resnet18(num_classes=1000)
        else:
            # Simple fallback: use random projection
            self.backbone = None
            # Create simple conv-like projection
            self.simple_proj = TernaryLinear(image_size * image_size * 3, 512, threshold)

        # Projection to embed_dim
        # ResNet-18 has 512 output features from global pooling
        self.projection = TernaryLinear(512, embed_dim, threshold)

    def forward(
        self,
        images: np.ndarray,
        training: bool = False
    ) -> np.ndarray:
        """
        Extract visual features.

        Args:
            images: (batch_size, 3, height, width) images
            training: Whether in training mode

        Returns:
            features: (batch_size, embed_dim) visual features
        """
        # Pass through CNN backbone or simple projection
        if self.backbone is not None:
            # Use full CNN backbone
            x = images

            # Forward through conv layers (manually, since we need features)
            for layer in self.backbone.layers[:-1]:  # Exclude final linear
                if hasattr(layer, 'forward'):
                    if isinstance(layer, (type(self.backbone.layers[0]),)):  # Conv-like layers
                        x = layer.forward(x, training)
                    else:
                        x = layer.forward(x)

            # Global average pooling
            features = np.mean(x, axis=(2, 3))  # (batch_size, 512)
        else:
            # Simple fallback: flatten and project
            batch_size = images.shape[0]
            flattened = images.reshape(batch_size, -1)
            features = self.simple_proj.forward(flattened, training)

        # Project to embedding dimension
        features = self.projection.forward(features, training)

        return features


class TernaryMultimodalAgent:
    """
    Multimodal agent combining vision and language.

    Architecture:
    - Vision encoder: Ternary CNN → visual features
    - Cross-modal projection: Map vision to language space
    - Language decoder: Ternary GPT for text generation

    Capabilities:
    - Image captioning
    - Visual question answering
    - Multimodal dialogue
    """

    def __init__(
        self,
        vision_config: Optional[Dict] = None,
        language_config: Optional[TernaryGPTConfig] = None,
        threshold: float = 0.3
    ):
        # Default configs
        if vision_config is None:
            vision_config = {
                'image_size': 224,
                'patch_size': 16,
                'embed_dim': 768
            }

        if language_config is None:
            language_config = TernaryGPTConfig(
                vocab_size=50257,
                max_len=1024,
                embed_dim=768,
                num_layers=12,
                num_heads=12,
                ff_dim=3072,
                threshold=threshold
            )

        self.vision_encoder = TernaryVisionEncoder(
            image_size=vision_config['image_size'],
            patch_size=vision_config['patch_size'],
            embed_dim=vision_config['embed_dim'],
            threshold=threshold
        )

        self.language_model = TernaryGPT(language_config)

        # Cross-modal projection
        self.vision_proj = TernaryLinear(
            vision_config['embed_dim'],
            language_config.embed_dim,
            threshold
        )

        # Vision prefix tokens (learnable)
        self.vision_prefix_len = 1  # Number of visual tokens to prepend
        self.vision_prefix_embed = np.random.randn(
            self.vision_prefix_len,
            language_config.embed_dim
        ) * 0.02

    def encode_image(
        self,
        images: np.ndarray,
        training: bool = False
    ) -> np.ndarray:
        """
        Encode images to language embedding space.

        Args:
            images: (batch_size, 3, height, width)
            training: Whether in training mode

        Returns:
            visual_embeds: (batch_size, 1, embed_dim)
        """
        # Extract visual features
        visual_features = self.vision_encoder.forward(images, training)

        # Project to language space
        visual_embeds = self.vision_proj.forward(visual_features, training)

        # Add batch dimension
        visual_embeds = visual_embeds[:, np.newaxis, :]  # (batch, 1, embed_dim)

        return visual_embeds

    def forward(
        self,
        images: np.ndarray,
        input_ids: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Forward pass for multimodal input.

        Args:
            images: (batch_size, 3, height, width)
            input_ids: (batch_size, text_len) text token indices
            training: Whether in training mode

        Returns:
            logits: (batch_size, 1 + text_len, vocab_size)
        """
        batch_size = images.shape[0]

        # Encode images
        visual_embeds = self.encode_image(images, training)

        # Get text embeddings
        text_embeds = self.language_model.token_embedding.forward(input_ids, training)

        # Concatenate: [visual_token, text_tokens]
        combined_embeds = np.concatenate([visual_embeds, text_embeds], axis=1)

        # Add positional encoding
        combined_embeds = self.language_model.pos_encoding.forward(combined_embeds)

        # Create causal mask
        from .ternary_transformer import create_causal_mask
        total_len = 1 + input_ids.shape[1]
        mask = create_causal_mask(total_len)

        # Pass through transformer blocks
        x = combined_embeds
        for block in self.language_model.blocks:
            x = block.forward(x, mask=mask, training=training)

        # Final layer norm
        x = self.language_model.ln_f.forward(x, training)

        # Project to vocabulary
        logits = self.language_model.lm_head.forward(x, training)

        return logits

    def caption_image(
        self,
        image: np.ndarray,
        max_length: int = 50,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[int]:
        """
        Generate caption for an image.

        Args:
            image: (3, height, width) single image
            max_length: Maximum caption length
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold

        Returns:
            caption_ids: List of token IDs
        """
        # Add batch dimension
        images = image[np.newaxis, ...]  # (1, 3, H, W)

        # Encode image
        visual_embeds = self.encode_image(images, training=False)

        # Start with empty text (or BOS token)
        caption_ids = []

        for _ in range(max_length):
            # Create input_ids
            if len(caption_ids) == 0:
                # Just use visual embedding
                input_ids = np.array([[0]])  # Dummy token
                text_embeds = self.language_model.token_embedding.forward(input_ids, training=False)
                combined_embeds = visual_embeds
            else:
                input_ids = np.array([caption_ids])
                text_embeds = self.language_model.token_embedding.forward(input_ids, training=False)
                combined_embeds = np.concatenate([visual_embeds, text_embeds], axis=1)

            # Add positional encoding
            combined_embeds = self.language_model.pos_encoding.forward(combined_embeds)

            # Create causal mask
            from .ternary_transformer import create_causal_mask
            seq_len = combined_embeds.shape[1]
            mask = create_causal_mask(seq_len)

            # Forward through transformer
            x = combined_embeds
            for block in self.language_model.blocks:
                x = block.forward(x, mask=mask, training=False)

            x = self.language_model.ln_f.forward(x, training=False)
            logits = self.language_model.lm_head.forward(x, training=False)

            # Get next token logits
            next_logits = logits[0, -1, :] / temperature

            # Top-p sampling
            sorted_indices = np.argsort(next_logits)[::-1]
            sorted_logits = next_logits[sorted_indices]

            # Softmax
            max_logit = np.max(sorted_logits)
            exp_logits = np.exp(sorted_logits - max_logit)
            probs = exp_logits / np.sum(exp_logits)

            # Cumulative probability
            cumsum_probs = np.cumsum(probs)
            cutoff_idx = np.searchsorted(cumsum_probs, top_p)

            # Sample from top-p
            top_indices = sorted_indices[:cutoff_idx + 1]
            top_probs = probs[:cutoff_idx + 1]
            top_probs = top_probs / np.sum(top_probs)

            next_token = np.random.choice(top_indices, p=top_probs)

            # Stop if EOS token (assume EOS = 50256 for GPT-2)
            if next_token == 50256:
                break

            caption_ids.append(int(next_token))

        return caption_ids

    def visual_qa(
        self,
        image: np.ndarray,
        question_ids: List[int],
        max_length: int = 50,
        temperature: float = 0.7
    ) -> List[int]:
        """
        Answer a question about an image.

        Args:
            image: (3, height, width) single image
            question_ids: List of token IDs for question
            max_length: Maximum answer length
            temperature: Sampling temperature

        Returns:
            answer_ids: List of token IDs
        """
        # Add batch dimension
        images = image[np.newaxis, ...]

        # Encode image
        visual_embeds = self.encode_image(images, training=False)

        # Encode question
        question_ids_array = np.array([question_ids])
        text_embeds = self.language_model.token_embedding.forward(question_ids_array, training=False)

        # Combine: [visual, question]
        combined_embeds = np.concatenate([visual_embeds, text_embeds], axis=1)

        # Generate answer autoregressively
        answer_ids = []

        for _ in range(max_length):
            # Add positional encoding
            current_embeds = self.language_model.pos_encoding.forward(combined_embeds)

            # Create mask
            from .ternary_transformer import create_causal_mask
            seq_len = current_embeds.shape[1]
            mask = create_causal_mask(seq_len)

            # Forward through transformer
            x = current_embeds
            for block in self.language_model.blocks:
                x = block.forward(x, mask=mask, training=False)

            x = self.language_model.ln_f.forward(x, training=False)
            logits = self.language_model.lm_head.forward(x, training=False)

            # Sample next token
            next_logits = logits[0, -1, :] / temperature
            probs = np.exp(next_logits - np.max(next_logits))
            probs = probs / np.sum(probs)

            next_token = np.random.choice(len(probs), p=probs)

            # Stop if EOS
            if next_token == 50256:
                break

            answer_ids.append(int(next_token))

            # Append to sequence
            next_embed = self.language_model.token_embedding.forward(
                np.array([[next_token]]),
                training=False
            )
            combined_embeds = np.concatenate([combined_embeds, next_embed], axis=1)

        return answer_ids

    def get_model_size(self) -> Dict[str, float]:
        """Get total model size."""
        vision_params = 11_000_000  # Approximate ResNet-18 params
        language_params = self.language_model.get_num_params()['total']
        projection_params = 512 * self.language_model.config.embed_dim

        total_params = vision_params + language_params + projection_params

        # Ternary size
        float32_size = total_params * 4 / (1024 * 1024)
        ternary_size = total_params * 0.27 / 8 / (1024 * 1024)

        return {
            'total_params': total_params,
            'vision_params': vision_params,
            'language_params': language_params,
            'float32_mb': float32_size,
            'ternary_mb': ternary_size,
            'compression': float32_size / ternary_size
        }


def create_ternary_multimodal_small() -> TernaryMultimodalAgent:
    """Create small multimodal agent (GPT-2 small + ResNet-18)."""
    language_config = TernaryGPTConfig(
        vocab_size=50257,
        max_len=1024,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        ff_dim=3072
    )

    return TernaryMultimodalAgent(language_config=language_config)


def create_ternary_multimodal_tiny() -> TernaryMultimodalAgent:
    """Create tiny multimodal agent for testing/demo."""
    vision_config = {
        'image_size': 224,
        'patch_size': 16,
        'embed_dim': 128
    }

    language_config = TernaryGPTConfig(
        vocab_size=256,
        max_len=256,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        ff_dim=512
    )

    return TernaryMultimodalAgent(
        vision_config=vision_config,
        language_config=language_config
    )
