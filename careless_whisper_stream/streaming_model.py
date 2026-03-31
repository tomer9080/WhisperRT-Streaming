import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Optional
from .model import AudioEncoder, TextDecoder, Whisper

from .streaming_decoding import DecodingTask, DecodingOptions, DecodingResult
from .streaming_transcribe import transcribe as transcribe_function
from .transcribe import transcribe as offline_transcribe_function
from .decoding import decode as non_causal_decode_function
from .audio import SpectrogramStream
from .timing import add_word_timestamps

from dataclasses import replace

from pytorch_lightning import LightningModule


class LoraLayer(LightningModule):
    def __init__(self, input_dim, output_dim, rank=8, alpha=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lora_A = nn.Parameter(torch.zeros(input_dim, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, output_dim))

        self.alpha = rank if alpha is None else alpha
        self.rank = rank
        self.scale = self.alpha / self.rank

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return x @ (self.lora_A @ self.lora_B) * self.scale


class LoraLinearLayer(LightningModule):
    def __init__(self, base_layer: nn.Linear, rank: int = 8, bias: int = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.base_layer = base_layer

        self.lora_layer = LoraLayer(base_layer.in_features, base_layer.out_features, rank=rank)
        self.aggregate_lora = True

    def turn_on_lora(self):
        self.aggregate_lora = True
    
    def turn_off_lora(self):
        self.aggregate_lora = False

    def forward(self, x: Tensor):
        if not self.aggregate_lora:
            return self.base_layer(x)
        
        return self.base_layer(x) + self.lora_layer(x)


class LoRAMultiHeadAttention(LightningModule):
    def __init__(self, n_head, query, key, value, out, rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.n_head = n_head
        self.query = LoraLinearLayer(query, rank)
        self.key = LoraLinearLayer(key, rank, bias=False)
        self.value = LoraLinearLayer(value, rank)
        self.out = LoraLinearLayer(out, rank)

    def forward(
        self,
        x: Tensor,
        xa: Tensor = None,
        mask: Tensor = None,
        kv_cache: dict = None,
        *args, **kwargs
    ):    
        q = self.query(x)

        if kv_cache is None or xa is None or self.key not in kv_cache:
            k = self.key(x if xa is None else xa)
            v = self.value(x if xa is None else xa)
            
        else:
            k = kv_cache[self.key]
            v = kv_cache[self.value]

        wv, qk = self.qkv_attention(q, k, v, mask, kv_cache)

        return self.out(wv), qk

    def qkv_attention(
            self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor = None, kv_cache: dict = None
    ):
        n_batch, n_ctx, n_state = q.shape
        _, k_ctx, _ = k.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 3, 1) * scale
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        qk = q @ k

        # apply causal mask
        if mask is not None:
            
            # kv_cache for beam search decoding case
            if kv_cache is not None and "beam_indices" in kv_cache.keys():
                for i in range(n_batch):
                    qk[i] = qk[i] + mask[kv_cache["beam_indices"][1][i * n_ctx]:kv_cache["beam_indices"][1][i * n_ctx] + n_ctx, :k_ctx]
            
            # For training, encoder/decoder causal masks
            elif k_ctx == n_ctx:
                qk = qk + mask[:n_ctx, :n_ctx]
            
            # kv_cache in the greedy decoding case
            else:
                qk = qk + mask[k_ctx - n_ctx:k_ctx, :k_ctx]

        qk = qk.float()

        w = F.softmax(qk, dim=-1).to(q.dtype)

        return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()


class StreamingAudioEncoder(AudioEncoder):

    def __init__(self, n_mels, n_ctx, n_state, n_head, n_layer, cache_gran, gran, rank, extra_gran_blocks):
        super().__init__(n_mels, n_ctx, n_state, n_head, n_layer)

        self.gran = gran
        self.extra_gran_blocks = extra_gran_blocks

        for block in self.blocks:
            block.attn = LoRAMultiHeadAttention(self.n_head,
                                                block.attn.query, 
                                                block.attn.key,
                                                block.attn.value,
                                                block.attn.out,
                                                rank)

        self.use_stream = False
        self.use_mask = False

        # mask for training
        matrix_size = n_ctx
        block_size = gran
        extra_blocks = extra_gran_blocks
        mask = torch.full((matrix_size, matrix_size), float('-inf'))

        for i in range(0, matrix_size, block_size):
            if (i // block_size) <= extra_blocks:
                zero_cols = (block_size * (extra_blocks + 1))
            else:
                zero_cols = (block_size * (extra_blocks + 1)) + ((i // block_size) - extra_blocks) * block_size

            mask[i:i + block_size, :zero_cols] = 0

        self.register_buffer("mask", mask, persistent=False)

    def _use_stream(self, use_stream: bool):
        self.use_stream = use_stream

    def _use_mask(self, use_mask: bool):
        self.use_mask = use_mask

    def _update_granularity(self, gran: int, extra_gran_blocks: int):
        self.gran = gran
        self.extra_gran_blocks = extra_gran_blocks

    def forward(self, x: Tensor, index: list = [0, 1500], kv_cache = None, mask = True):
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        
        if self.use_stream:
            offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
            x = x[:, offset:offset + self.gran + (int(offset == 0) * (self.extra_gran_blocks * self.gran))] # offset
            x = (x + self.positional_embedding[offset:offset + self.gran + (int(offset == 0) * (self.extra_gran_blocks * self.gran))]).to(x.dtype)
        else: # use during training
            x = x[:, index[0]:index[1]] # offset
            x = (x + self.positional_embedding[index[0]:index[1]]).to(x.dtype)

        for block in self.blocks:
            chosen_mask = mask[..., :index[1], :index[1]] if isinstance(mask, Tensor) else self.mask if ((mask is not None) and (self.use_stream)) or ((mask is not None) and (self.use_mask)) else None
            # chosen_mask = mask[..., :index[1], :index[1]] if isinstance(mask, Tensor) else self.mask if (mask is not None) else None
            # print(f"{chosen_mask=}")
            x = block(x, mask=chosen_mask, kv_cache=kv_cache)
            
        x = self.ln_post(x)

        return x

    def _no_mask_forward(self, x: Tensor):
        return super().forward(x)

class StreamingTextDecoder(TextDecoder):
    def __init__(self, n_vocab, n_ctx, n_state, n_head, n_layer, rank):
        super().__init__(n_vocab, n_ctx, n_state, n_head, n_layer)

        self.n_ctx = n_ctx
        self.n_state = n_state

        for block in self.blocks:
            block.attn = LoRAMultiHeadAttention(n_head,
                                                block.attn.query, 
                                                block.attn.key,
                                                block.attn.value,
                                                block.attn.out,
                                                rank)
            
            block.cross_attn = LoRAMultiHeadAttention(n_head,
                                                      block.cross_attn.query, 
                                                      block.cross_attn.key,
                                                      block.cross_attn.value,
                                                      block.cross_attn.out,
                                                      rank)
    
    def forward(self, x: Tensor, xa: Tensor, kv_cache: dict = None, dump_type: str = None, **kwargs):
        """
        x : torch.LongTensor, shape = (batch_size, <= n_ctx)
            the text tokens
        xa : torch.Tensor, shape = (batch_size, n_audio_ctx, n_audio_state)
            the encoded audio features to be attended on
        """
        if kv_cache is not None and "beam_indices" in kv_cache.keys():
            x = self.token_embedding(x) + self.positional_embedding.unsqueeze(0).expand(x.shape[0], self.positional_embedding.shape[0], self.positional_embedding.shape[1])[kv_cache["beam_indices"][0], kv_cache["beam_indices"][1]].view(x.shape[0], -1, self.n_state)
        else:
            offset = next(iter(kv_cache.values())).shape[1] if kv_cache else 0
            x = self.token_embedding(x) + self.positional_embedding[offset : offset + x.shape[-1]]
        
        x = x.to(xa.dtype)

        for block in self.blocks:
            x = block(x, xa, mask=self.mask, kv_cache=kv_cache)

        x = self.ln(x)
        logits = (
            x @ torch.transpose(self.token_embedding.weight.to(x.dtype), 0, 1)
        ).float()
        
        return logits


class StreamingWhisper(Whisper):
    def __init__(self, dims, cache_gran: bool = True, gran: int = 16, rank: int = 0, extra_gran_blocks: int = 0, random_masked_model: bool = False):
        super().__init__(dims)

        self.cache_gran = cache_gran
        self.gran = gran
        self.rank = rank
        self.extra_gran_blocks = extra_gran_blocks
        self.random_masked_model = random_masked_model

        if not self.random_masked_model:
            print(f"Running a streaming whisper model, using chunk size: {gran * 20}[msec] and {extra_gran_blocks} extra chunks for initialization.")
        else:
            print(f"Running a random masked streaming whisper model, using chunk size: {gran * 20}[msec], {extra_gran_blocks} extra chunks for initialization and random masking during training.")

        # The only difference is a streaming encoder
        self.encoder = StreamingAudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            cache_gran=cache_gran,
            gran=gran,
            rank=rank,
            extra_gran_blocks=extra_gran_blocks
        )

        self.decoder = StreamingTextDecoder(
            self.dims.n_vocab,
            self.dims.n_text_ctx,
            self.dims.n_text_state,
            self.dims.n_text_head,
            self.dims.n_text_layer,
            rank=rank
        )

        # Advisor params - Dropped.
        self.advisor_type = None
        self.n_advisor_class = 0
        self.decoder_advisor = None

        self.decoding_task = None
        self.spec_streamer = SpectrogramStream()
        
        self.use_ca_cache_hook = True # relevant only when ca_kv_cache is installed

    def reset(self, use_stream: bool = True, clean_task: bool = True):
        self.encoder._use_stream(use_stream)
        del self.decoding_task # trigger clean encoder kv caching
        self.decoding_task = None
        self.spec_streamer.reset()

    @torch.no_grad()
    def decode(self, mel: Tensor, options: DecodingOptions = DecodingOptions(), use_frames: bool = False, **kwargs) -> DecodingResult:
        if kwargs: options = replace(options, **kwargs)
        
        if use_frames: # mel is frames of audio, need to calc mel
            mel = self.spec_streamer.calc_mel_with_new_frame(mel).squeeze(0)

        # if not a random masked model - force granularity.
        if self.encoder.gran != options.gran and not self.random_masked_model:
            print(f"Encoder gran & options gran differ. forcing options to be on encoder's gran: {self.encoder.gran}")
            options.gran = self.encoder.gran
        
        if self.random_masked_model:
            # print(f"Random masked model - using options granularity {options.gran} without forcing it to be the encoder's granularity {self.encoder.gran}")
            self.encoder._update_granularity(options.gran, options.look_ahead_blocks)

        if not self.decoding_task:
            self.decoding_task = DecodingTask(self, options)
        
        return self.decoding_task.run(mel.unsqueeze(0))

    def _turn_off_lora(self):
        for _, layer in self.encoder.named_modules():
            if isinstance(layer, LoraLinearLayer):
                layer.turn_off_lora()
        
        for _, layer in self.decoder.named_modules():
            if isinstance(layer, LoraLinearLayer):
                layer.turn_off_lora()
    
    def _turn_on_lora(self):
        for _, layer in self.encoder.named_modules():
            if isinstance(layer, LoraLinearLayer):
                layer.turn_on_lora()
        
        for _, layer in self.decoder.named_modules():
            if isinstance(layer, LoraLinearLayer):
                layer.turn_on_lora()

    def _cancel_streaming_mode(self):
        self._turn_off_lora()
        self.encoder._use_stream(False)
    
    def _revert_streaming_mode(self):
        self._turn_on_lora()
        self.encoder._use_stream(True)

    @torch.no_grad()
    def non_causal_decode(self, mel: Tensor, options: DecodingOptions = DecodingOptions(), **kwargs) -> DecodingResult:
        self._cancel_streaming_mode()
        results = non_causal_decode_function(self, self.encoder._no_mask_forward(mel), options, **kwargs)
        self._revert_streaming_mode()        
        return results

    @torch.no_grad()
    def non_causal_transcribe(self, audio, **kwargs):
        self._cancel_streaming_mode()
        results = offline_transcribe_function(self, audio, **kwargs)
        self._revert_streaming_mode()
        return results
    
    @torch.no_grad()
    def _offline_word_timestamps(self, segments, tokenizer, mel, num_frames, last_speech_timestamp):
        add_word_timestamps(segments=segments,
                            model=self,
                            tokenizer=tokenizer,
                            mel=mel,
                            num_frames=num_frames,
                            last_speech_timestamp=last_speech_timestamp)
        return segments

    def remove_encoder_kv_cache_hooks(self):
        for hook in self.encoder._forward_hooks.values():
            hook.remove()

    def install_encoder_kv_cache_hooks(self, cache = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_audio_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]

        def install_hooks(layer: nn.Module):
            if isinstance(layer, LoRAMultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))
        
        self.encoder.apply(install_hooks)
        return cache, hooks
    
    def install_decoder_kv_cache_hooks(self, cache = None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                cache[module] = output
            else:
                if "beam_indices" not in cache.keys() or all([index == (cache[module].shape[1]) for index in cache["beam_indices"][1]]):
                    cache[module] = torch.cat([cache[module], output], dim=1).detach()
                else:
                    for _, (beam, index, output_index) in enumerate(zip(*cache["beam_indices"])):
                        if index < cache[module].shape[1]:
                            cache[module][beam, index] = output[beam, output_index]
                        else:
                            cache[module] = torch.cat([cache[module], output[:, (output_index):]], dim=1).detach()

            return cache[module]

        for name, module in self.decoder.named_modules():
            if isinstance(module, LoRAMultiHeadAttention) and "attn" in name and "cross" not in name:
                hooks.append(module.key.register_forward_hook(save_to_cache))
                hooks.append(module.value.register_forward_hook(save_to_cache))

        return cache, hooks

    def install_cross_attn_kv_cache_hooks(self, cache=None):
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if not self.use_ca_cache_hook:
                return cache[module]
            
            if module not in cache or output.shape[1] > self.dims.n_audio_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            
            return cache[module]
        
        def check_if_calculation_is_needed(module, _):
            if not self.use_ca_cache_hook:
                return cache[module]

        for name, module in self.decoder.named_modules():
            if isinstance(module, LoRAMultiHeadAttention) and "cross_attn" in name:
                hooks.append(module.key.register_forward_hook(save_to_cache))
                hooks.append(module.key.register_forward_pre_hook(check_if_calculation_is_needed))
                hooks.append(module.value.register_forward_hook(save_to_cache))
                hooks.append(module.value.register_forward_pre_hook(check_if_calculation_is_needed))
        
        return cache, hooks

    # For non-causal decoding compatibility
    def install_kv_cache_hooks(self, cache: Optional[dict] = None):
        """
        The `MultiHeadAttention` module optionally accepts `kv_cache` which stores the key and value
        tensors calculated for the previous positions. This method returns a dictionary that stores
        all caches, and the necessary hooks for the key and value projection modules that save the
        intermediate tensors to be reused during later calculations.

        Returns
        -------
        cache : Dict[nn.Module, torch.Tensor]
            A dictionary object mapping the key/value projection modules to its cache
        hooks : List[RemovableHandle]
            List of PyTorch RemovableHandle objects to stop the hooks to be called
        """
        cache = {**cache} if cache is not None else {}
        hooks = []

        def save_to_cache(module, _, output):
            if module not in cache or output.shape[1] > self.dims.n_text_ctx:
                # save as-is, for the first token or cross attention
                cache[module] = output
            else:
                cache[module] = torch.cat([cache[module], output], dim=1).detach()
            return cache[module]
        
        def install_hooks(layer: nn.Module):
            if isinstance(layer, LoRAMultiHeadAttention):
                hooks.append(layer.key.register_forward_hook(save_to_cache))
                hooks.append(layer.value.register_forward_hook(save_to_cache))

        self.decoder.apply(install_hooks)
        return cache, hooks

    # refers to function from streaming_decoding, streaming_transcribe library
    transcribe = transcribe_function