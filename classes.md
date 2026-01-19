# Zonos Classes Reference

This document describes the classes involved in Zonos inference and streaming TTS as implemented in this repo. It focuses on what each class does, how it is wired, and the data flow between them. Functions that are central to class behavior are mentioned where they impact class logic, but the primary focus is on classes.

## Overview of the streaming inference path

The main runtime path is:

1) `Zonos` loads a model and autoencoder, builds the backbone, and prepares a `PrefixConditioner` that turns conditioning features into a prefix sequence of embeddings.
2) `make_cond_dict` produces tensors for the conditioner inputs. `Zonos.prepare_conditioning` runs the `PrefixConditioner` twice (conditional and unconditional) and concatenates them for classifier free guidance.
3) `Zonos.generate` performs autoregressive decoding with a delay pattern across codebooks, using `sample_from_logits` to sample tokens per step.
4) The resulting codebook tokens are reverted to their true time order and decoded by `DACAutoencoder.decode` into waveform audio.
5) Optional speaker embeddings come from `SpeakerEmbeddingLDA` and related classes.

The backbone used for decoding is either `TorchZonosBackbone` (pure PyTorch transformer) or `MambaSSMZonosBackbone` (Mamba SSM or hybrid). Both backbones consume the same `InferenceParams` cache structure.

## zonos/model.py

### Zonos (torch.nn.Module)

**Purpose**: Central model wrapper for inference. It composes the autoencoder, transformer or Mamba backbone, conditioning stack, and sampling logic. It also handles CFG (classifier free guidance), CUDA graph capture, and codebook-specific heads.

**Initialization**:
- Inputs: `config: ZonosConfig`, `backbone_cls` (defaults to the first available backbone class).
- Creates `DACAutoencoder`, backbone instance, and `PrefixConditioner` using `config`.
- Uses `autoencoder.num_codebooks` (DAC uses 9) to size the per codebook layers.
- Builds `self.embeddings`: one `nn.Embedding(1026, d_model)` per codebook (codebook size plus EOS and masked tokens).
- Builds `self.heads`: one `nn.Linear(d_model, 1025)` per codebook (codebook size plus EOS).
- Prepares CUDA graph capture state (`_cg_*` fields) used by `_decode_one_token`.
- Optionally registers `_pad_embeddings_and_heads` to pad weights to a vocab multiple.

**Key fields**:
- `autoencoder`: `DACAutoencoder` for codebook encoding and decoding.
- `backbone`: `TorchZonosBackbone` or `MambaSSMZonosBackbone` depending on config.
- `prefix_conditioner`: `PrefixConditioner` that emits prefix embeddings.
- `eos_token_id`, `masked_token_id`: token ids used in codebooks.
- `embeddings` and `heads`: per codebook embedding and prediction heads.
- `device` property: returns the current module device from the first parameter.

**from_pretrained**:
- Downloads `config.json` and `model.safetensors` from a Hugging Face repo, then delegates to `from_local`.
- Uses `DEFAULT_DEVICE` and loads model weights into a freshly initialized `Zonos` instance.

**from_local**:
- Loads `ZonosConfig` from JSON and selects a backbone from `BACKBONES`.
- Prefers the pure torch backbone if the config uses a transformer (no `ssm_cfg`) and if the torch backbone is available.
- Loads safetensors into the model state dict and moves the model to `torch.float16` on the requested device.
- Moves the DAC autoencoder weights to the same device via `model.autoencoder.dac.to(device)`.

**_pad_embeddings_and_heads**:
- Pads embedding and head weights to `pad_vocab_to_multiple_of` using `pad_weight_` after loading a state dict.
- Ensures tensor dimensions align for better kernel efficiency.

**make_speaker_embedding**:
- Lazy loads a `SpeakerEmbeddingLDA` model if needed.
- Produces the LDA projected speaker embedding from a waveform (ignores the base embedding).
- Returns shape `[1, emb_dim]` in `float16`.

**embed_codes**:
- Sums the per codebook embeddings for each codebook channel.
- Input shape: `[batch, n_codebooks, seq]`. Output shape: `[batch, seq, d_model]`.

**apply_heads**:
- Applies each per codebook head to the hidden states and stacks the outputs.
- Input shape: `[batch, seq, d_model]`. Output shape: `[batch, n_codebooks, seq, vocab]`.

**_compute_logits**:
- Runs the backbone, then the per codebook heads.
- Uses only the last time step of the backbone output.
- Applies CFG if `cfg_scale != 1.0` by splitting the batch into conditional and unconditional halves.
- Forces logits for indices >= 1025 to `-inf` to ignore padding.
- Returns logits in `float32` to stabilize sampling.

**_decode_one_token**:
- Single step decode that constructs hidden states and calls `_compute_logits`.
- If `cfg_scale == 1.0`, skips duplication and runs a single batch.
- If CFG is enabled, duplicates hidden states to create conditional and unconditional batches.
- Supports CUDA graph capture for stable batch sizes on CUDA. It runs 3 warmup iterations, captures a graph that includes `embed_codes` and `_compute_logits`, and replays it on later steps with the same batch size.
- If CUDA graph is disabled or not on CUDA, it runs a normal forward path.

**_prefill**:
- Used to prefill the cache with prefix conditioning and optional audio prefix codes.
- If CFG is enabled, expands `input_ids` to match the doubled batch size.
- Concatenates prefix hidden states with embedded codebook tokens and computes logits for the next token.

**setup_cache**:
- Builds `InferenceParams` by allocating caches from the backbone and initializing lengths per sample.
- Pads `max_seqlen` to a multiple of 8 via `find_multiple`.

**prepare_conditioning**:
- Runs `PrefixConditioner` for both conditional and unconditional dictionaries.
- If `uncond_dict` is not provided, it is built by copying required keys from `cond_dict`.
- Concatenates conditional and unconditional outputs along the batch dimension for CFG usage. This doubles the batch size for the decoder.

**can_use_cudagraphs**:
- Returns true only for CUDA and when the backbone class name includes `_mamba_ssm`. This matches the code path that supports CUDA graphs.

**generate**:
- The main autoregressive decoding loop. It asserts `cfg_scale != 1.0` and uses CFG throughout.
- Inputs include prefix conditioning embeddings, optional audio prefix codebook tokens, CFG scale, sampling settings, and a callback.
- Computes `audio_seq_len = prefix_audio_len + max_new_tokens` and `seq_len = prefix_conditioning_len + audio_seq_len + 9` to size the inference cache.
- Allocates inference caches with `batch_size * 2` to account for conditional and unconditional batches.
- Builds a codebook tensor `codes` filled with `unknown_token = -1` of shape `[batch, 9, seq]` and optionally inserts prefix tokens.
- Applies `apply_delay_pattern` to shift each codebook stream by a different offset. This allows parallel token prediction per time step.
- Prefills the backbone cache using `_prefill`, then samples the first new token.
- Uses `torch.compile` on `_decode_one_token` when CUDA graphs are not used, and disables compile when graphs are enabled or `disable_torch_compile` is true.
- Iterates steps to sample new tokens with `sample_from_logits`, adds a `logit_bias` that only allows EOS on codebook 0, tracks `remaining_steps` to enforce a 9 step EOS tail across codebooks, and uses `masked_token_id` to fill in masked codebooks during EOS staging.
- Updates `InferenceParams.seqlen_offset` and `lengths_per_sample` on each step.
- Supports a progress bar and a callback per generated frame; if the callback returns false, generation stops early.
- After generation, calls `revert_delay_pattern` to undo the codebook delay, masks invalid tokens >= 1024, and trims the delayed tail.
- Returns codebook tensor shaped `[batch, 9, audio_len]` suitable for `DACAutoencoder.decode`.

## zonos/autoencoder.py

### DACAutoencoder

**Purpose**: Wrapper around the Descript DAC autoencoder used to convert audio to codebook tokens and back.

**Initialization**:
- Loads the pretrained DAC model `descript/dac_44khz`.
- Stores codebook size, number of codebooks, and sampling rate.
- Sets the model to eval and disables gradients.

**preprocess**:
- Resamples input waveform to 44.1 kHz and right pads to a multiple of 512 samples.
- Used before encoding to ensure alignment with DAC hop size.

**encode**:
- Calls `dac.encode` and returns `audio_codes` with shape `[batch, n_codebooks, frames]`.

**decode**:
- Runs DAC decode under autocast (float16 on GPU, full precision on CPU).
- Returns waveform as shape `[batch, 1, samples]` in `float32`.

## zonos/conditioning.py

### Conditioner (torch.nn.Module)

**Purpose**: Base class for all conditioning modules. It standardizes projection and unconditional handling.

**Initialization**:
- Inputs: `output_dim`, `name`, `cond_dim`, `projection`, `uncond_type`.
- Builds `self.project` as identity, linear, or MLP.
- Creates a learned unconditional vector when `uncond_type == "learned"` (initialized to zeros).

**forward**:
- If `inputs` is `None`, returns the unconditional vector shaped `[1, 1, output_dim]` (using `uncond_vector.data`).
- Otherwise calls `apply_cond` and then `project`.

### EspeakPhonemeConditioner

**Purpose**: Converts text to phonemes and embeds them for conditioning.

**Key behavior**:
- Uses eSpeak via `phonemizer` to produce phoneme strings.
- Tokenizes phonemes to symbol ids and embeds them with `nn.Embedding`.
- Tokenization adds BOS and EOS tokens and left pads to the longest sequence in the batch.
- Text is normalized (numbers, punctuation, and Japanese readings) before phonemization and results are cached.
- Returns embeddings shaped `[batch, seq, output_dim]`.

### FourierConditioner

**Purpose**: Converts continuous values to Fourier features.

**Key behavior**:
- Requires `output_dim` to be even so it can split into cosine and sine halves.
- Stores a random weight matrix of shape `[output_dim // 2, input_dim]` as a buffer.
- Normalizes inputs to `[0, 1]` based on `min_val` and `max_val`.
- Computes cosine and sine features and concatenates them to `[batch, seq, output_dim]`.

### IntegerConditioner

**Purpose**: Converts integer inputs into embeddings.

**Key behavior**:
- Uses an embedding table sized `max_val - min_val + 1`.
- Expects inputs of shape `[batch, seq, 1]` and embeds `x - min_val`.

### PassthroughConditioner

**Purpose**: For conditioners that already provide dense features.

**Key behavior**:
- Validates feature dimension and returns the input unchanged.

### PrefixConditioner

**Purpose**: Builds the full prefix conditioning sequence by combining multiple conditioners.

**Initialization**:
- Takes a `PrefixConditionerConfig` and `output_dim`.
- Instantiates each conditioner from the config with `build_conditioners`.
- Tracks `required_keys` for conditioners that do not have learned unconditional vectors.
 - Adds a `LayerNorm` over the concatenated conditioning sequence.

**forward**:
- Verifies required conditioning keys exist in `cond_dict`.
- Calls each conditioner with its entry from `cond_dict` (or `None` for unconditional).
- Expands single batch conditioners to the max batch size.
- Concatenates results along the sequence dimension and applies projection and layer norm.
- Output shape: `[batch, cond_seq_len, output_dim]`.

## zonos/config.py

### InferenceParams (dataclass)

**Purpose**: Holds inference cache state and offsets used during incremental decoding.

**Fields**:
- `max_seqlen`, `max_batch_size` and dynamic offsets (`seqlen_offset`, `batch_size_offset`).
- `key_value_memory_dict` containing per layer caches.
- `lengths_per_sample`: `[batch]` tensor tracking sequence lengths.

**reset**:
- Resets offsets and clears `lengths_per_sample`.

### BackboneConfig (dataclass)

**Purpose**: Configuration for transformer or Mamba backbones.

**Fields**:
- Model dimensions (`d_model`, `d_intermediate`, `attn_mlp_d_intermediate`).
- Structure (`n_layer`, `attn_layer_idx`).
- Attention config (`attn_cfg`).
- Norm and residual behavior (`rms_norm`, `residual_in_fp32`, `norm_epsilon`).
- `ssm_cfg` controls Mamba or hybrid settings.

### PrefixConditionerConfig (dataclass)

**Purpose**: Configuration for building `PrefixConditioner`.

**Fields**:
- `conditioners`: list of conditioner specs.
- `projection`: how to project concatenated conditioner outputs.

### ZonosConfig (dataclass)

**Purpose**: Top level configuration for the full model.

**Fields**:
- `backbone` and `prefix_conditioner` sub configs.
- Token ids for EOS and masked tokens.
- Optional padding multiple for vocab sizes.

**from_dict**:
- Converts nested dictionaries into `BackboneConfig` and `PrefixConditionerConfig` instances.

## zonos/backbone/_torch.py

### TorchZonosBackbone (torch.nn.Module)

**Purpose**: Pure PyTorch transformer backbone used for decoding.

**Initialization**:
- Builds `n_layer` `TransformerBlock` instances.
- Defines a final `LayerNorm`.

**allocate_inference_cache**:
- Precomputes rotary embeddings for a fixed max position (16384) based on `head_dim = d_model / num_heads` and stores them on the module.
- Creates per layer KV caches, stored in `InferenceParams.key_value_memory_dict`.

**forward**:
- Computes input positions from `InferenceParams.lengths_per_sample` to support streaming.
- Applies rotary embeddings and passes hidden states through each layer.
- Returns normalized hidden states of shape `[batch, seq, d_model]`.

### TransformerBlock (torch.nn.Module)

**Purpose**: One transformer block with attention and feed forward sublayers.

**Key behavior**:
- Pre norm architecture: `x + Attention(norm(x))`, then `x + MLP(norm(x))`.
- `allocate_inference_cache` returns an empty KV cache tensor shaped `[batch, max_seqlen, 2, num_heads_kv, head_dim]`.

### Attention (torch.nn.Module)

**Purpose**: Multi head attention with rotary embeddings and KV cache for streaming.

**Key behavior**:
- Uses grouped query attention: `num_heads` for queries and `num_heads_kv` for KV.
- Projects input to Q, K, V via a single linear projection and reshapes to `[batch, seqlen, heads, head_dim]`.
- Applies rotary embeddings to Q and K.
- Updates KV cache via `_update_kv_cache` with `InferenceParams` offsets and returns a full prefix cache to enable streaming attention.
- Runs `scaled_dot_product_attention` with causal masking when `seqlen > 1` and `enable_gqa=True`.

### FeedForward (torch.nn.Module)

**Purpose**: Gated MLP (SwiGLU style) with two linear layers.

**Key behavior**:
- First linear projects to `2 * d_intermediate` then splits into `y` and `gate`.
- Applies `silu` to the gate and multiplies by `y`.
- Second linear projects back to `d_model`.

## zonos/backbone/_mamba_ssm.py

### MambaSSMZonosBackbone (torch.nn.Module)

**Purpose**: Backbone built from Mamba SSM or hybrid blocks.

**Initialization**:
- Builds layers using `mamba_ssm.create_block`.
- Supports transformer or hybrid architectures based on config.

**allocate_inference_cache**:
- Delegates to each Mamba block to create its cache.

**forward**:
- Iterates through blocks, maintaining a `residual` term used by fused add and norm.
- Applies the final layer norm using `layer_norm_fn` with optional RMS norm and FP32 residual.

## zonos/speaker_cloning.py

### logFbankCal (torch.nn.Module)

**Purpose**: Computes log mel spectrogram features for speaker embedding models.

**Key behavior**:
- Uses `torchaudio.transforms.MelSpectrogram`.
- Applies log and mean normalization across time.
- Outputs log mel features shaped `[batch, n_mels, frames]`.

### ASP (torch.nn.Module)

**Purpose**: Attentive statistics pooling used in speaker models.

**Key behavior**:
- Reshapes and applies a small attention network over time.
- Produces concatenated mean and standard deviation vectors (mu and sigma) over time.

### SimAMBasicBlock (torch.nn.Module)

**Purpose**: Residual block with SimAM attention.

**Key behavior**:
- Two conv layers with batch norm and ReLU.
- Optional downsample if stride or channel mismatch.
- SimAM computes a per element attention map from variance with a small lambda stabilizer.

### BasicBlock (torch.nn.Module)

**Purpose**: Standard residual block without SimAM.

**Key behavior**:
- Two conv layers with batch norm and ReLU and optional downsample.

### Bottleneck (torch.nn.Module)

**Purpose**: ResNet style bottleneck block for 2D convs.

**Key behavior**:
- 1x1, 3x3, 1x1 conv sequence with residual shortcut.
- Uses expansion factor 4 in the output channels.

### ResNet (torch.nn.Module)

**Purpose**: Generic ResNet front end for speaker embedding.

**Initialization**:
- Supports 1D, 2D, or 3D convs based on `feat_dim`.
- Builds four residual stages with increasing channels and stride.
 - Uses an initial 3x3 conv, batch norm, and ReLU before the residual stages.

### ResNet293 (factory)

**Purpose**: Returns a `ResNet` with `SimAMBasicBlock` and a fixed depth of [10, 20, 64, 3].

### ResNet293_based (torch.nn.Module)

**Purpose**: Full speaker embedding network using `ResNet293` plus ASP pooling.

**Key behavior**:
- Applies feature extraction with `featCal` (log mel).
- Passes through `ResNet293`, then `ASP` pooling.
- Optional dropout before a final linear projection to embedding space.

### SEModule (torch.nn.Module)

**Purpose**: Squeeze and excitation module used in ECAPA TDNN.

**Key behavior**:
- Uses global average pooling and two 1x1 convs to generate a channel gate.

### Bottle2neck (torch.nn.Module)

**Purpose**: Res2Net style bottleneck used in ECAPA TDNN.

**Key behavior**:
- Splits channels into multiple groups, applies dilated convs, and recombines.
- Applies SE gating and a residual connection.

### ECAPA_TDNN (torch.nn.Module)

**Purpose**: Alternative speaker encoder architecture.

**Key behavior**:
- Computes feature maps with stacked `Bottle2neck` blocks.
- Aggregates multi layer features and applies attentive statistics pooling.
- Produces a 192 dim embedding.

### SpeakerEmbedding (torch.nn.Module)

**Purpose**: Loads the ResNet293 based speaker encoder and exposes a simple forward API.

**Initialization**:
- Loads weights from `ResNet293_SimAM_ASP_base.pt` and installs `logFbankCal`.
- Runs in eval mode and disables gradients.
 - Exposes a `dtype` property and caches resamplers per input sample rate.

**prepare_input**:
- Converts stereo to mono and resamples to 16 kHz.
 - Accepts 1D or 2D waveforms and averages channels when needed.

**forward**:
- Produces a speaker embedding tensor of shape `[batch, emb_dim]` on the original device.
- Keeps the embedding model in eval mode with gradients disabled.

### SpeakerEmbeddingLDA (torch.nn.Module)

**Purpose**: Adds an LDA projection on top of `SpeakerEmbedding` to produce a lower dimensional embedding.

**Initialization**:
- Downloads base speaker encoder and LDA weights from Hugging Face.
- Instantiates a float32 linear layer sized from the LDA state dict.

**forward**:
- Returns a tuple of `(base_embedding, lda_embedding)` in float32.

## Files without classes

The following files are used heavily in the class implementations but define no classes:

- `zonos/codebook_pattern.py` provides `apply_delay_pattern` and `revert_delay_pattern` used by `Zonos.generate`.
- `zonos/sampling.py` provides sampling utilities used by `Zonos.generate`.
- `zonos/utils.py` provides device utilities and padding helpers used by `Zonos`.
- `gradio_interface.py` provides the UI and orchestration functions but no classes.
- `zonos/backbone/__init__.py` registers available backbone classes.
