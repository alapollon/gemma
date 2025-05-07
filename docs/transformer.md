

### **Documentation for `gemma/transformer.py`**

#### **Tranformers python Overview**
This file implements the Gemma Transformer, a configurable Transformer architecture used for various machine learning tasks. The file includes definitions for configuration, initialization, caching, forward passes, attention masks, and multimodal input handling.

---

### **Imports in `transformer.py`**
1. **`dataclasses`**:
   - Used for defining data classes such as `TransformerConfig`.
2. **`enum`**:
   - Used for defining enumerations such as `QueryPreAttentionNormalisation`.
3. **`typing` and `typing.Iterable`**:
   - Provides type hints for better code readability and static type checking.
4. **`warnings`**:
   - Used for raising deprecation warnings.
5. **`einops`**:
   - Library for flexible and readable tensor operations; used in `_include_vision_embeddings`.
6. **`flax.linen as nn`**:
   - Flax library for building neural networks.
7. **`jax` and `jax.numpy as jnp`**:
   - JAX is used for numerical computation and machine learning.
8. **`gemma.layers`**:
   - Contains the `RMSNorm` layer used for normalization in the Transformer.
9. **`gemma.modules`**:
   - Provides embedding, attention, and block modules used extensively in the Transformer.
10. **`gemma.multimodal.vision as gemma_vision`**:
   - Provides vision-related functionalities, including `SigLiPFromPatches` and vision token checks.

---

### **Key Classes**
#### 1. **`QueryPreAttentionNormalisation`**
   - Enum to define strategies for query normalization in attention mechanisms.
   - Options:
     - **`BY_ONE_OVER_SQRT_HEAD_DIM`**: Scales the query by `1/sqrt(head_dim)`.
     - **`BY_EMBED_DIM_DIV_NUM_HEADS`**: Scales the query by `embed_dim // num_heads`.
     - **`BY_ONE_OVER_SQRT_EMBED_DIM_DIV_NUM_HEADS`**: Scales the query by `1/sqrt(embed_dim // num_heads)`.

#### 2. **`TransformerConfig`**
   - Configuration class for defining parameters of the Transformer architecture.
   - **Attributes**:
     - Model properties such as `num_layers`, `embed_dim`, `hidden_dim`, `num_heads`, etc.
     - Attention properties such as `attention_types`, `attn_logits_soft_cap`, and `sliding_window_size`.
     - Multimodal properties such as `vision_encoder`.
   - **Methods**:
     - `query_pre_attn_scalar`: Computes a scalar multiplier for the query before attention.
     - Predefined configurations (`gemma_2b`, `gemma_7b`, etc.) for different model sizes.
     - `from_params`: Infers configuration from model parameters.
     - `init_cache`: Initializes cache for attention mechanisms.

#### 3. **`Transformer`**
   - Implements the Transformer model.
   - **Attributes**:
     - `config`: Instance of `TransformerConfig` defining the model's configuration.
   - **Methods**:
     - `setup`: Initializes the model components, including embeddings and attention blocks.
     - `__call__`: Forward pass through the Transformer.
     - `_include_vision_embeddings`: Adds vision embeddings to the input sequence.
     - `_check_tokens_for_vision`: Validates vision token formatting.

---

### **Functions**
#### 1. **`make_attention_layers_types`**
   - Generates attention type patterns for layers.
   - **Parameters**:
     - `pattern`: Tuple of attention types (e.g., LOCAL_SLIDING, GLOBAL).
     - `num_layers`: Total number of layers.
   - **Returns**: A tuple of attention types repeated for `num_layers`.

#### 2. **Attention Mask Functions**
   - **`compute_sequence_attention_mask`**: Generates sequence attention masks (causal or bidirectional).
   - **`compute_attention_masks`**: Computes causal attention masks for a given timestep.
   - **`make_causal_attn_mask`**: Creates a causal attention mask.
   - **`make_causal_with_prefix_attn_mask`**: Creates a causal mask with a prefix for bidirectional attention.
   - **`make_block_mask`**: Generates block masks for bidirectional segments.
   - **`add_bidirectional_mask`**: Adds bidirectional masks to attention masks.

#### 3. **Position and Input Handling**
   - **`build_positions_from_mask`**: Computes position indices for position encodings.
   - **`mm_input_length`**: Returns the number of multimodal tokens in the input.

---

### **Imported Invocations**
1. **`gemma.modules`**
   - **`LayerCache`**: Used as the `Cache` type for attention mechanisms.
   - **`AttentionType`**: Enum for specifying attention mechanisms (e.g., LOCAL_SLIDING, GLOBAL).
   - **`Embedder`**: Used in `setup` to initialize the embedding layer.
   - **`Block`**: Represents a Transformer block; used in `setup`.

2. **`gemma.layers`**
   - **`RMSNorm`**: Final normalization layer applied to the Transformer output.

3. **`gemma.multimodal.vision`**
   - **`SigLiPFromPatches`**: Vision encoder; used if multimodal input is enabled.
   - **`check_mask`**: Validates vision tokens.
   - **`check_special_vision_token`**: Checks for special tokens like BEGIN_IMAGE_TOKEN.

4. **`einops`**
   - Used in `_include_vision_embeddings` for rearranging vision embeddings.

5. **`jax` and `jax.numpy`**
   - Numerical computation for attention masks, embeddings, and forward passes.

---

### **Deprecation Warning**
The `Transformer` class is marked as deprecated, with a recommendation to use `gm.nn.GemmaXX`. Users are advised to refer to the [documentation](https://gemma-llm.readthedocs.io/) for updated implementations.

---

Would you like me to provide documentation for specific sections in a docstring format, or do you need additional exploration of imported modules?
