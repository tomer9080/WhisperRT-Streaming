from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import re
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from .audio import CHUNK_LENGTH
from .tokenizer import Tokenizer, get_tokenizer
from .utils import compression_ratio

if TYPE_CHECKING:
    from .streaming_model import StreamingWhisper
    from .model import Whisper

from .hyp_buffer import HypothesisBuffer

@dataclass(frozen=False)
class DecodingOptions:
    # whether to perform X->X "transcribe" or X->English "translate"
    task: str = "transcribe"

    # language that the audio is in; uses detected language if None
    language: Optional[str] = None

    # sampling-related options
    temperature: float = 0.0
    sample_len: Optional[int] = None  # maximum number of tokens to sample
    best_of: Optional[int] = None  # number of independent sample trajectories, if t > 0
    beam_size: Optional[int] = None  # number of beams in beam search, if t == 0
    patience: Optional[float] = None  # patience in beam search (arxiv:2204.05424)

    # "alpha" in Google NMT, or None for length norm, when ranking generations
    # to select which to return among the beams or best-of-N samples
    length_penalty: Optional[float] = None

    # text or tokens to feed as the prompt or the prefix; for more info:
    # https://github.com/openai/whisper/discussions/117#discussioncomment-3727051
    prompt: Optional[Union[str, List[int]]] = None  # for the previous context
    prefix: Optional[Union[str, List[int]]] = None  # to prefix the current context

    # list of tokens ids (or comma-separated token ids) to suppress
    # "-1" will suppress a set of symbols as defined in `tokenizer.non_speech_tokens()`
    suppress_tokens: Optional[Union[str, Iterable[int]]] = "-1"
    suppress_blank: bool = True  # this will suppress blank outputs

    # timestamp sampling options
    without_timestamps: bool = True  # use <|notimestamps|> to sample text tokens only
    max_initial_timestamp: Optional[float] = 1.0

    # implementation details
    fp16: bool = False  # use fp16 for most of the calculation

    # Advisor & Streaming params
    advised: bool = False
    attentive_advisor: bool = False
    use_sa: bool = False
    ctx: int = 3 # CA / SA ctx given to advisor.
    gran: int = 15 # granularity of encoder embeddings (!) 15 = 300 msec (each frame equals 20msec)
    pad_audio_features: bool = True
    single_frame_mel: bool = True
    pad_input: bool = False

    look_ahead_blocks: int = 0
    maximal_seconds_context: int = 30 # after 30 seconds, reset the mel.

    use_kv_cache: bool = False
    use_ca_kv_cache: bool = False

    # streaming decoding args
    stream_decode: bool = True
    tokens_per_frame: int = 2
    n_tokens_look_back: int = 2
    streaming_timestamps: bool = True
    wait_for_all: bool = False
    force_first_tokens_timestamps: bool = False

    # localagreement params
    localagreement: bool = False

@dataclass(frozen=False)
class DecodingResult:
    audio_features: Tensor
    language: str
    language_probs: Optional[Dict[str, float]] = None
    tokens: List[int] = field(default_factory=list)
    retired_tokens: List[int] = field(default_factory=list)
    text: str = ""
    full_text: str = ""
    avg_logprob: float = np.nan
    no_speech_prob: float = np.nan
    temperature: float = np.nan
    compression_ratio: float = np.nan
    timestamps: Dict[Tuple, Tuple] = field(default_factory=tuple)
    timed_tokens: List[int] = field(default_factory=list)
    timed_text: str = ""


class Inference:
    def logits(self, tokens: Tensor, audio_features: Tensor) -> Tensor:
        """Perform a forward pass on the decoder and return per-token logits"""
        raise NotImplementedError

    def rearrange_kv_cache(self, source_indices) -> None:
        """Update the key-value cache according to the updated beams"""
        raise NotImplementedError

    def cleanup_caching(self) -> None:
        """Clean up any resources or hooks after decoding is finished"""
        pass

    def flush_tokens_from_cache(self) -> None:
        """flush irrelevant tokens from cache during streaming"""
        pass


class PyTorchInference(Inference):
    def __init__(self, model: "StreamingWhisper", initial_token_length: int, use_kv_cache: bool = False, dump_type: str = None, n_tokens_look_back: int = 2, use_beam: bool = False):
        self.model: "StreamingWhisper" = model
        self.initial_token_length = initial_token_length
        self.hooks = []
        # custom
        self.use_kv_cache = use_kv_cache
        self.kv_cache = {} if use_kv_cache else None
        self.dump_type = dump_type
        self.n_tokens_look_back = n_tokens_look_back
        self.cached_logits = None
        self.use_beam = use_beam

        key_modules = [block.attn.key for block in self.model.decoder.blocks]
        value_modules = [block.attn.value for block in self.model.decoder.blocks]
        self.kv_modules = key_modules + value_modules

    def logits(self, tokens: Tensor, audio_features: Tensor, first_prediction: bool = False, beam_indices: list[list] = None) -> Tensor:
        
        if not self.kv_cache and self.use_kv_cache:
            self.kv_cache, self.hooks = self.model.install_decoder_kv_cache_hooks()
        
        if tokens.shape[-1] > self.initial_token_length and self.use_kv_cache:
            # only need to use the last token except in the first forward pass
            if not self.use_beam:
                tokens = tokens[:, -self.n_tokens_look_back:] if first_prediction else tokens[:, -1:]
            else:
                n_beams = tokens.shape[0]
                self.kv_cache["beam_indices"] = beam_indices # an elegant way to send it to decoder ? 
                tokens = tokens[beam_indices[0], beam_indices[1]].view(n_beams, -1)
                

        return self._concat_logits_if_needed(self.model.decoder(tokens, audio_features, kv_cache=self.kv_cache))

    def _concat_logits_if_needed(self, logits: Tensor):
        if not self.use_kv_cache: return logits

        if self.cached_logits is None:
            self.cached_logits = logits
            
            return logits
        
        if not self.use_beam: # greedy
            self.cached_logits = torch.cat([self.cached_logits, logits], dim=1)
            return self.cached_logits

        # beam kv_cache
        n_beams, n_ctx, n_vocab = logits.shape
        
        for i, (beam, index, output_index) in enumerate(zip(*self.kv_cache["beam_indices"])):
            if index < self.cached_logits.shape[1]:
                self.cached_logits[beam, index] = logits[beam, output_index]
            else:
                self.cached_logits = torch.cat([self.cached_logits, logits[:, (output_index):]], dim=1)

        return self.cached_logits

    def cleanup_caching(self):
        if not self.use_kv_cache:
            return
        
        for hook in self.hooks:
            hook.remove()

        del self.kv_cache
        del self.hooks
        self.kv_cache = {}
        self.hooks = []

    def flush_tokens_from_cache(self):
        for key in self.kv_cache.keys():
            if key == "beam_indices": continue
            self.kv_cache[key] = self.kv_cache[key][:, :-self.n_tokens_look_back].detach()

        self.cached_logits = self.cached_logits[:, :-self.n_tokens_look_back].detach()

    def rearrange_kv_cache(self, source_indices):
        if not self.use_kv_cache: return 

        if source_indices != list(range(len(source_indices))):
            
            for module in self.kv_modules:
                # update the key/value cache to contain the selected sequences
                self.kv_cache[module] = self.kv_cache[module][source_indices].detach()
            
            self.cached_logits = self.cached_logits[source_indices].detach()


class SequenceRanker:
    def rank(
        self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]
    ) -> List[int]:
        """
        Given a list of groups of samples and their cumulative log probabilities,
        return the indices of the samples in each group to select as the final result
        """
        raise NotImplementedError


class MaximumLikelihoodRanker(SequenceRanker):
    """
    Select the sample with the highest log probabilities, penalized using either
    a simple length normalization or Google NMT paper's length penalty
    """

    def __init__(self, length_penalty: Optional[float]):
        self.length_penalty = length_penalty

    def rank(self, tokens: List[List[Tensor]], sum_logprobs: List[List[float]]):
        def scores(logprobs, lengths):
            result = []
            for logprob, length in zip(logprobs, lengths):
                if self.length_penalty is None:
                    penalty = length
                else:
                    # from the Google NMT paper
                    penalty = ((5 + length) / 6) ** self.length_penalty
                result.append(logprob / penalty)
            return result

        # get the sequence with the highest score
        lengths = [[len(t) for t in s] for s in tokens]
        
        return [np.argmax(scores(p, l)) for p, l in zip(sum_logprobs, lengths)]


class TokenDecoder:
    def reset(self):
        """Initialize any stateful variables for decoding a new sequence"""

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        """Specify how to select the next token, based on the current trace and logits

        Parameters
        ----------
        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        sum_logprobs : Tensor, shape = (n_batch)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Tensor, shape = (n_batch, current_sequence_length + 1)
            the tokens, appended with the selected next token

        completed : bool
            True if all sequences has reached the end of text

        """
        raise NotImplementedError

    def finalize(
        self, tokens: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Sequence[Sequence[Tensor]], List[List[float]]]:
        """Finalize search and return the final candidate sequences

        Parameters
        ----------
        tokens : Tensor, shape = (n_audio, n_group, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence

        sum_logprobs : Tensor, shape = (n_audio, n_group)
            cumulative log probabilities for each sequence

        Returns
        -------
        tokens : Sequence[Sequence[Tensor]], length = n_audio
            sequence of Tensors containing candidate token sequences, for each audio input

        sum_logprobs : List[List[float]], length = n_audio
            sequence of cumulative log probabilities corresponding to the above

        """
        raise NotImplementedError


class GreedyDecoder(TokenDecoder):
    def __init__(self, temperature: float, eot: int):
        self.temperature = temperature
        self.eot = eot

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()

        logprobs = F.log_softmax(logits.float(), dim=-1)

        current_logprobs = logprobs[torch.arange(logprobs.shape[0]), next_tokens]
        sum_logprobs += current_logprobs * (tokens[:, -1] != self.eot)

        next_tokens[tokens[:, -1] == self.eot] = self.eot
        if (next_tokens[-1] == self.eot).all():
            return tokens, True
        tokens = torch.cat([tokens, next_tokens[:, None]], dim=-1)

        completed = (tokens[:, -1] == self.eot).all()
        return tokens, completed

    def finalize(self, tokens: Tensor, sum_logprobs: Tensor):
        # make sure each sequence has at least one EOT token at the end
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()


class BeamSearchDecoder(TokenDecoder):
    def __init__(
        self,
        beam_size: int,
        eot: int,
        inference: Inference,
        patience: Optional[float] = None,
    ):
        self.beam_size = beam_size
        self.eot = eot
        self.inference = inference
        self.patience = patience or 1.0
        self.max_candidates: int = round(beam_size * self.patience)
        self.finished_sequences = None

        assert (
            self.max_candidates > 0
        ), f"Invalid beam size ({beam_size}) or patience ({patience})"

    def reset(self):
        self.finished_sequences = None

    def update(
        self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor
    ) -> Tuple[Tensor, bool]:
        if tokens.shape[0] % self.beam_size != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.beam_size} != 0")

        n_audio = tokens.shape[0] // self.beam_size
        if self.finished_sequences is None:  # for the first update
            self.finished_sequences = [{} for _ in range(n_audio)]

        logprobs = F.log_softmax(logits.float(), dim=-1)
        next_tokens, source_indices, finished_sequences = [], [], []
        for i in range(n_audio): # in our case n_audio = 1
            scores, sources, finished = {}, {}, {}

            # STEP 1: calculate the cumulative log probabilities for possible candidates
            for j in range(self.beam_size): # go over each trajectory of beam
                idx = i * self.beam_size + j
                prefix = tokens[idx].tolist()
                for logprob, token in zip(*logprobs[idx].topk(self.beam_size + 1)):
                    new_logprob = (sum_logprobs[idx] + logprob).item()
                    sequence = tuple(prefix + [token.item()])
                    scores[sequence] = new_logprob
                    sources[sequence] = idx

            # STEP 2: rank the candidates and keep the top beam_size sequences for each audio
            saved = 0
            for sequence in sorted(scores, key=scores.get, reverse=True):
                if sequence[-1] == self.eot:
                    finished[sequence] = scores[sequence]
                else:
                    sum_logprobs[len(next_tokens)] = scores[sequence]
                    next_tokens.append(sequence)
                    source_indices.append(sources[sequence])

                    saved += 1
                    if saved == self.beam_size:
                        break

            finished_sequences.append(finished)

        tokens = torch.tensor(next_tokens, device=tokens.device)
        self.inference.rearrange_kv_cache(source_indices)

        # add newly finished sequences to self.finished_sequences
        assert len(self.finished_sequences) == len(finished_sequences)
        for previously_finished, newly_finished in zip(
            self.finished_sequences, finished_sequences
        ):
            for seq in sorted(newly_finished, key=newly_finished.get, reverse=True):
                if len(previously_finished) >= self.max_candidates:
                    break  # the candidate list is full
                previously_finished[seq] = newly_finished[seq]

        # mark as completed if all audio has enough number of samples
        completed = all(
            len(sequences) >= self.max_candidates
            for sequences in self.finished_sequences
        )
        return tokens, completed

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        # collect all finished sequences, including patience, and add unfinished ones if not enough
        sum_logprobs = sum_logprobs.cpu()
        for i, sequences in enumerate(self.finished_sequences):
            if (
                len(sequences) < self.beam_size
            ):  # when not enough sequences are finished
                for j in list(np.argsort(sum_logprobs[i]))[::-1]:
                    sequence = preceding_tokens[i, j].tolist() + [self.eot]
                    sequences[tuple(sequence)] = sum_logprobs[i][j].item()
                    if len(sequences) >= self.beam_size:
                        break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in sequences.keys()]
            for sequences in self.finished_sequences
        ]
        sum_logprobs: List[List[float]] = [
            list(sequences.values()) for sequences in self.finished_sequences
        ]
        return tokens, sum_logprobs


class StreamingDecoder(TokenDecoder):
    def __init__(self,
                 temperature: float,
                 tokens_per_frame: int,
                 eot: int,
                 inference: Inference,
                 n_tokens_look_back: int = 2,
                 streaming_timestamps: bool = False
                ):
        self.tokens_per_frame = tokens_per_frame
        self.eot = eot
        self.inference = inference
        self.last_logits: Tensor = None
        self.temperature = temperature
        self.tokens_look_back = n_tokens_look_back
        self.check_token_index = -n_tokens_look_back - 1
        self.streaming_timestamps = streaming_timestamps
        self.timestamps_map = {-1: 0}
        self.check_tokens_override = None
        self.commited_words = None
        self.discarded_words = None
        self._times = []
        self.transcript_buffer = HypothesisBuffer()
        self._reset_timestamps()

    def _insert_timestamps(self,  audio_features: Tensor, tokens: Tensor, enc_emb_gran: int):
        """
        Given the tokens and stable lists. 
        If we send complete, it means that we should put a timestamp on the tokens.
        """
        # Given T, and a gran g, forward pass T/g times.
        # Observe each vector, and try to find swap points for each token.
        # Overall, extra T/g forward passes.
        examined_token_index = 3 # this index is the first non initial token index to be predcited
        for i in range(enc_emb_gran, audio_features.shape[-1] + 1, enc_emb_gran):
            i_logits = self.inference.logits(tokens, audio_features[:, :i])
            
            if tokens[0, examined_token_index] in i_logits.argmax(dim=-1): # stamp of this token.
                self.timestamps_map[examined_token_index - 3] = i * 20
                examined_token_index += 1
            
            if examined_token_index >= tokens.shape[-1]: break

    def _mark_check_tokens(self, check: bool = True):
        self.check_tokens_override = check

    def _check_last_tokens(self, logits: Tensor, tokens: Tensor, next_tokens: Tensor, check_tokens: bool):
        stable = []
        
        if not check_tokens:
            return stable, tokens

        examine_tokens_indices = range(tokens.shape[1] + self.check_token_index, tokens.shape[1] - 1)
        for i in examine_tokens_indices:
            
            token_index = i + 1
            token_prob_index = i
            
            examined_token = tokens[:, token_index] # This is the predicted token for this index.
            
            # It means that the prediction is stabilizing. We can move on, check next token.
            # or - Prob is down, but token has still highest prob. Continue.
            if self.last_logits[:, token_prob_index, examined_token] <= logits[:, token_prob_index, examined_token] \
                or \
                (self.last_logits[:, token_prob_index, examined_token] > logits[:, token_prob_index, examined_token] and next_tokens[:, token_prob_index] == examined_token):
                stable.append(True)
                continue
            
            else:
                tokens = tokens[:, :token_index] # Crop next tokens - Irrelevant after flush.
                stable.append(False)
                break
        
        return stable, tokens

    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor, check_tokens: bool = False, index: int = 0, first_frame: bool = False) -> Tuple[Tensor, bool]: 
        # calc next_tokens, logprob, completed
        if self.temperature == 0:
            next_tokens = logits.argmax(dim=-1)
        else:
            next_tokens = Categorical(logits=logits / self.temperature).sample()
        
        logprobs = F.log_softmax(logits.float(), dim=-1)
        completed = next_tokens[0, -1] == self.eot

        # means we are at the beginning, append tokens greedily.
        if first_frame:
            if next_tokens[0, -1] != self.eot:
                tokens = torch.cat([tokens, next_tokens[None, :, -1]], dim=-1)
                self.last_logits = logits
            return tokens, completed

        # Otherwise, check if last tokens are stable.
        check_tokens = check_tokens if self.check_tokens_override is None else self.check_tokens_override
        stable, tokens = self._check_last_tokens(logits, tokens, next_tokens, check_tokens)

        if all(stable) and next_tokens[0, -1] != self.eot:
            tokens = torch.cat([tokens, next_tokens[None, :, -1]], dim=-1)
            
            if tokens.shape[-1] - 1 - 4 not in self.timestamps_map.keys(): # mark start of timestamp
                self.timestamps_map[tokens.shape[-1] - 1 - 4] = index - 1

        sum_logprobs += logprobs[:, -1, next_tokens[0, -1]]

        # take last tokens logits to compare on next decode step
        self.last_logits = logits.clone().detach()

        return tokens, completed

    def _reset_timestamps(self):
        self.timestamp_tokens_indices = [None, None]
        self.timestamps_indices = [None, None]

    def reset(self):
        self.check_token_index = -self.tokens_look_back - 1

    def finalize(self, tokens, sum_logprobs):
        tokens = F.pad(tokens, (0, 1), value=self.eot)
        return tokens, sum_logprobs.tolist()
    
    def local_agreement_words(self, tokens: Tensor, tokenizer: Tokenizer, frame_index: int):
        words_list = tokenizer.decode(tokens).split()
        self._times.extend([frame_index for _ in range(len(words_list) - len(self._times))])
        print(f"frame_index: {frame_index=}\n{self._times=}")
        self.transcript_buffer.insert(words_list, self._times)
        self.commited_words = self.transcript_buffer.flush()
        print(f"{self.commited_words=}\n-------------------------------")
        return [tokenizer.encode(" ".join(self.commited_words))]


class BeamStreamingDecoder(TokenDecoder):
    def __init__(self,
                 temperature: float,
                 tokens_per_frame: int,
                 eot: int,
                 inference: Inference,
                 n_tokens_look_back: int = 2,
                 n_beams: int = 1,
                 pad_token: int = None,
                 wait_for_all: bool = False,
                ):
        self.tokens_per_frame = tokens_per_frame
        self.eot = eot
        self.pad_token = pad_token # sotlm token
        self.inference = inference
        self.last_logits: list = []
        self.temperature = temperature
        self.tokens_look_back = n_tokens_look_back
        self.check_token_index = [4 + 1 for _ in range(n_beams)] # Here I'll save the last index that is relevant for checking.
        self.n_beams = n_beams
        self.sum_logprobs = torch.zeros(n_beams)
        self.timestamps_map = {}
        self.wait_for_all = wait_for_all
        self.finished_sequences = {}
        self.check_tokens_override = None
        self.commited_words = None
        self.discarded_words = None
        self.transcript_buffer = HypothesisBuffer()
        self._times = []

    def _mark_check_tokens(self, check: bool = True):
        self.check_tokens_override = check

    def _get_last_valid_token_index(self, prefix: Tensor):
        indices = torch.where((prefix == self.eot) | (prefix == self.pad_token))[0]
        last_valid_token_index = (prefix.shape[-1] - 1) if indices.shape == torch.Size([0]) else (indices.min().item() - 1)
        last_valid_token_index = max(last_valid_token_index, 3)
        
        return last_valid_token_index

    def _check_last_tokens(self, prefix: Tensor, logits: Tensor, check_tokens: bool):
        """
        given the last k logits, and the tokens, determine where we should proceed from.
        prefix - a list with the tokens, including padding tokens and eot. Start from eot / pad token index. of shape (l)
        logits - of shape (l, n_vocab). The logits of the specific beam.
        """
        last_valid_token_index = self._get_last_valid_token_index(prefix)
        
        if not check_tokens: 
            return last_valid_token_index, True

        examined_prob_indices = range(max(last_valid_token_index - self.tokens_look_back, 3), last_valid_token_index)
        for examined_prob_index in examined_prob_indices:
            
            examined_token_index = examined_prob_index + 1
            examined_token = prefix[examined_token_index]
            examined_candidates = logits[examined_prob_index].topk(self.n_beams).indices

            if examined_token not in examined_candidates: # Means the last predicted token is out of out topk.
                # pad with irrelevant tokens
                prefix[examined_token_index:] = self.pad_token
                return examined_prob_index, False
            
        # If nothing was returned during the check, it means that all tokens were stable.
        # Continue decoding from the last valid token index.
        return last_valid_token_index, True

    def _check_last_tokens_new(self, prefix: Tensor, logits: Tensor, check_tokens: bool, beam: int):
        """
        check two conditions instead one:
        1. token it still in beam given new chunk.
        2. token probability gradient is positive.
        TODO:
        1. Save for each prefix a probability trajectory per token instead of the whole logits
        2. Consider the case when two prefixes might use different tokens but  when decoding they are the same.
        """
        last_valid_token_index = self._get_last_valid_token_index(prefix)

        if not check_tokens: 
            return last_valid_token_index, True
        
        examined_prob_indices = range(max(last_valid_token_index - self.tokens_look_back, 3), last_valid_token_index)
        for examined_prob_index in examined_prob_indices:
            examined_token_index = examined_prob_index + 1
            examined_token = prefix[examined_token_index]
            examined_candidates = logits[examined_prob_index].topk(self.n_beams).indices

            if examined_token not in examined_candidates: # Means the last predicted token is out of out topk.
                # pad with irrelevant tokens
                prefix[examined_token_index:] = self.pad_token
                return examined_prob_index, False
            
            # instead of saving the full last logits
            # we can save only the probabilities of the last predicted tokens, and check if they are going up or down. If they are going down, it means that the prediction is not stable, and we should flush.
            if self.last_logits[beam].shape[-1] > examined_prob_index and self.last_logits[beam][examined_prob_index] > logits[examined_prob_index, examined_token]: # Prob went down, flush
            # if self.last_logits[beam].shape[-1] > examined_prob_index and \
            # torch.exp(self.last_logits[beam][examined_prob_index]) - torch.exp(logits[examined_prob_index, examined_token]) < -0.2: # Prob went down, flush
                prefix[examined_token_index:] = self.pad_token
                return examined_prob_index, False

        # If nothing was returned during the check, it means that all tokens were stable.
        # Continue decoding from the last valid token index.
        return last_valid_token_index, True

    def update(self, tokens: Tensor, logits: Tensor, sum_logprobs: Tensor, first_frame: bool = False, check_tokens: bool = False) -> Tuple[Tensor, bool]: 
        if tokens.shape[0] % self.n_beams != 0:
            raise ValueError(f"{tokens.shape}[0] % {self.n_beams} != 0")

        scores, sources, finished = {}, {}, {}
        next_tokens, source_indices, finished_sequences = [], [], {}
        logprobs = F.log_softmax(logits.float(), dim=-1)
        
        for beam in range(self.n_beams):
            prefix = tokens[beam] # tokens in this beam.
            check_tokens = check_tokens if self.check_tokens_override is None else self.check_tokens_override

            # new addition
            sampling_index, stable = self._check_last_tokens_new(prefix, logits[beam], check_tokens, beam) if not first_frame else ((tokens.shape[-1] - 1), False)

            prefix = prefix.tolist() # using list, easier to append to.

            # Calculate candidates from the last token index we should check.
            for logprob, token in zip(*logprobs[beam, sampling_index].topk(self.n_beams + 1)):
                new_logprob = (logprobs[beam, range(3, sampling_index - 1), tokens[beam, 4:sampling_index]].sum() + logprob).item()
                
                token_index = sampling_index + 1
                if token_index == len(prefix):
                    prefix.append(token.item())
                else:
                    prefix[token_index] = token.item()

                sequence = prefix[:prefix.index(self.pad_token)] if self.pad_token in prefix else prefix
                sequence = tuple(sequence)
                scores[sequence] = new_logprob
                sources[sequence] = beam

        # After all beams were checked, and tokens were calculated
        # Get top n_beams sequences.
        saved = 0
        for sequence in sorted(scores, key=scores.get, reverse=True):
            if self.wait_for_all and sequence[-1] == self.eot:
                finished[sequence] = scores[sequence]
            sum_logprobs[len(next_tokens)] = scores[sequence]
            next_tokens.append(Tensor(sequence).long())
            source_indices.append(sources[sequence])

            saved += 1
            if saved == self.n_beams:
                break

        tokens = torch.nn.utils.rnn.pad_sequence(next_tokens, batch_first=True, padding_value=self.pad_token).to(tokens.device)
        self.inference.rearrange_kv_cache(source_indices)

        # new addition
        self.last_logits = []
        for i, source in enumerate(source_indices):
            self.last_logits.append(logits[source, range(0, tokens[i].shape[-1] - 1), tokens[i][1:]])

        if not self.wait_for_all: # greedy stop mode.
            completed = any([self.eot in s for s in next_tokens]) # Greedy stop - Believe any beam that says enough.
            return tokens, completed

        # If we wait for EOT in all beams. regular beam stop mode.
        for sequence in sorted(finished, key=finished.get, reverse=True):
            if len(finished_sequences) >= self.n_beams: break
            finished_sequences[sequence] = finished[sequence]

        # we have enough trajectories that reached EOT, in this specific frame.
        self.finished_sequences = finished_sequences
        completed = len(self.finished_sequences) >= self.n_beams
        return tokens, completed

    def reset(self):
        self.check_token_index = -self.tokens_look_back

    def finalize(self, preceding_tokens: Tensor, sum_logprobs: Tensor):
        if not self.wait_for_all:
            preceding_tokens = F.pad(preceding_tokens, (0, 1), value=self.eot)
            return preceding_tokens, sum_logprobs.tolist()
        
        sum_logprobs = sum_logprobs.cpu()
        if len(self.finished_sequences) < self.n_beams:
            for j in list(np.argsort(sum_logprobs[0]))[::-1]:
                sequence = preceding_tokens[0][j].tolist() + [self.eot]
                self.finished_sequences[tuple(sequence)] = sum_logprobs[0][j].item()
                if len(self.finished_sequences) >= self.n_beams: break

        tokens: List[List[Tensor]] = [
            [torch.tensor(seq) for seq in self.finished_sequences.keys()]
        ]
        
        sum_logprobs: List[List[float]] = [
            list(self.finished_sequences.values())
        ]

        return tokens, sum_logprobs

    def local_agreement_words(self, tokens: Tensor, tokenizer: Tokenizer, frame_index: int):
        words_list = tokenizer.decode(tokens).split()
        self._times.extend([frame_index for _ in range(len(words_list) - len(self._times))])
        print(f"frame_index: {frame_index=}\n{self._times=}")
        self.transcript_buffer.insert(words_list, self._times)
        self.commited_words = self.transcript_buffer.flush()
        print(f"{self.commited_words=}\n-------------------------------")
        return [tokenizer.encode(" ".join(self.commited_words))]


class LogitFilter:
    def apply(self, logits: Tensor, tokens: Tensor) -> None:
        """Apply any filtering or masking to logits in-place

        Parameters
        ----------
        logits : Tensor, shape = (n_batch, vocab_size)
            per-token logits of the probability distribution at the current step

        tokens : Tensor, shape = (n_batch, current_sequence_length)
            all tokens in the context so far, including the prefix and sot_sequence tokens

        """
        raise NotImplementedError


class SuppressBlank(LogitFilter):
    def __init__(self, tokenizer: Tokenizer, sample_begin: int):
        self.tokenizer = tokenizer
        self.sample_begin = sample_begin

    def apply(self, logits: Tensor, tokens: Tensor):
        if tokens.shape[1] == self.sample_begin:
            logits[..., self.tokenizer.encode(" ") + [self.tokenizer.eot]] = -np.inf


class SuppressTokens(LogitFilter):
    def __init__(self, suppress_tokens: Sequence[int]):
        self.suppress_tokens = list(suppress_tokens)

    def apply(self, logits: Tensor, tokens: Tensor):
        logits[..., self.suppress_tokens] = -np.inf


class DecodingTask:
    inference: Inference
    sequence_ranker: SequenceRanker
    decoder: TokenDecoder
    logit_filters: List[LogitFilter]

    def __init__(self, model: "StreamingWhisper", options: DecodingOptions):
        self.model = model

        language = options.language or "en"
        tokenizer = get_tokenizer(
            model.is_multilingual,
            num_languages=model.num_languages,
            language=language,
            task=options.task,
        )
        self.tokenizer: Tokenizer = tokenizer
        self.options: DecodingOptions = self._verify_options(options)

        self.n_group: int = options.beam_size or options.best_of or 1
        self.n_ctx: int = model.dims.n_text_ctx
        self.sample_len: int = options.sample_len or model.dims.n_text_ctx // 2

        self.sot_sequence: Tuple[int] = tokenizer.sot_sequence
        if self.options.without_timestamps:
            self.sot_sequence = tokenizer.sot_sequence_including_notimestamps

        self.initial_tokens: Tuple[int] = self._get_initial_tokens()
        self.sample_begin: int = len(self.initial_tokens)
        self.sot_index: int = self.initial_tokens.index(tokenizer.sot)

        # inference: implements the forward pass through the decoder, including kv caching
        dump_type = "ATT" if self.options.attentive_advisor else None
        self.inference = PyTorchInference(model, len(self.initial_tokens), self.options.use_kv_cache, dump_type=dump_type, n_tokens_look_back=options.n_tokens_look_back, use_beam=options.beam_size > 0)

        # sequence ranker: implements how to rank a group of sampled sequences
        self.sequence_ranker = MaximumLikelihoodRanker(options.length_penalty)

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.stream_decode and options.beam_size in [None, 0]:
            print(f"Initialized StreamingDecoder with temperature {options.temperature} and tokens_per_frame {options.tokens_per_frame}")
            self.decoder = StreamingDecoder(
                options.temperature, options.tokens_per_frame, tokenizer.eot, self.inference, options.n_tokens_look_back, options.streaming_timestamps
            )
        elif options.stream_decode and options.beam_size > 0:
            print(f"Initialized BeamStreamingDecoder with beam size {options.beam_size} and temperature {options.temperature}")
            self.decoder = BeamStreamingDecoder(
                options.temperature, options.tokens_per_frame, tokenizer.eot, self.inference, options.n_tokens_look_back, options.beam_size, tokenizer.sot_lm, options.wait_for_all
                # options.temperature, options.tokens_per_frame, tokenizer.eot, self.inference, options.n_tokens_look_back, options.beam_size, tokenizer.eot, options.wait_for_all
            )
        elif options.beam_size is not None and options.beam_size > 0:
            self.decoder = BeamSearchDecoder(options.beam_size, tokenizer.eot, self.inference, options.patience)
            print(f"Initialized BeamSearchDecoder with beam size {options.beam_size} and patience {options.patience}")

            if self.options.localagreement:
                self.decoder = BeamStreamingDecoder(
                    options.temperature, options.tokens_per_frame, tokenizer.eot, self.inference, options.n_tokens_look_back, options.beam_size, tokenizer.sot_lm, options.wait_for_all
                )
                self.decoder._mark_check_tokens(False)
        else:
            self.decoder = GreedyDecoder(options.temperature, tokenizer.eot)
            print(f"Initialized GreedyDecoder with temperature {options.temperature}")
            
            if self.options.localagreement:
                self.decoder = self.decoder = StreamingDecoder(
                    options.temperature, options.tokens_per_frame, tokenizer.eot, self.inference, options.n_tokens_look_back, options.streaming_timestamps
                )
                self.decoder._mark_check_tokens(False)

        # logit filters: applies various rules to suppress or penalize certain tokens
        self.logit_filters = []
        if self.options.suppress_blank:
            self.logit_filters.append(SuppressBlank(self.tokenizer, self.sample_begin))
        if self.options.suppress_tokens:
            self.logit_filters.append(SuppressTokens(self._get_suppress_tokens()))
        
        self.mel = None
        self.index = 0

        # repeat text tensors by the group size, for beam search or best-of-n sampling 
        self.tokens: Tensor = torch.tensor([self.initial_tokens]) # no need to use repeat, batch is meaningless in stream.
        self.tokens = self.tokens.repeat_interleave(self.n_group, dim=0).to(model.device)

        n_batch = self.n_group # streaming, batch_size larger than 1 is not allowed. only beam size is relevant
        self.sum_logprobs: Tensor = torch.zeros(n_batch, device=model.device)
        self.no_speech_probs = [np.nan] * n_batch

        # Causal encoder inference
        self._init_enc_kv_caching()

        if options.use_ca_kv_cache:
            self._init_ca_kv_caching()

        self.audio_features = torch.zeros((1, model.dims.n_audio_ctx, model.dims.n_audio_state)).to(model.device)
        self.frame_counter = 0

    def _init_enc_kv_caching(self):
        self.enc_kv_cache, self.enc_hooks = self.model.install_encoder_kv_cache_hooks()
    
    def _init_ca_kv_caching(self):
        self.ca_kv_cache, self.dec_ca_hooks = self.model.install_cross_attn_kv_cache_hooks()

    def _cleanup_encoder_caching(self):
        for hook in self.enc_hooks:
            hook.remove()
        
        del self.enc_kv_cache
        del self.enc_hooks
        self.enc_kv_cache = {}
        self.enc_hooks = []

    def _cleanup_ca_caching(self):
        if not self.options.use_ca_kv_cache:
            return
        
        for hook in self.dec_ca_hooks:
            hook.remove()
        
        del self.ca_kv_cache
        del self.dec_ca_hooks
        self.ca_kv_cache = {}
        self.dec_ca_hooks = []

    def __del__(self):
        self._cleanup_encoder_caching()
        self._cleanup_ca_caching()
        self.inference.cleanup_caching()

    def _verify_options(self, options: DecodingOptions) -> DecodingOptions:
        if options.beam_size is not None and options.best_of is not None:
            raise ValueError("beam_size and best_of can't be given together")
        if options.temperature == 0:
            if options.best_of is not None:
                raise ValueError("best_of with greedy sampling (T=0) is not compatible")
        if options.patience is not None and options.beam_size is None:
            raise ValueError("patience requires beam_size to be given")
        if options.length_penalty is not None and not (
            0 <= options.length_penalty <= 1
        ):
            raise ValueError("length_penalty (alpha) should be a value between 0 and 1")

        return options

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)

        if prefix := self.options.prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt := self.options.prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip())
                if isinstance(prompt, str)
                else prompt
            )
            tokens = (
                [self.tokenizer.sot_prev]
                + prompt_tokens[-(self.n_ctx // 2 - 1) :]
                + tokens
            )

        return tuple(tokens)

    def _get_suppress_tokens(self) -> Tuple[int]:
        suppress_tokens = self.options.suppress_tokens

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(self.tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend(
            [
                self.tokenizer.transcribe,
                self.tokenizer.translate,
                self.tokenizer.sot,
                self.tokenizer.sot_prev,
                self.tokenizer.sot_lm,
            ]
        )
        if self.tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(self.tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))

    def _get_audio_features(self, mel: Tensor, index: list = None):
        
        if self.options.fp16:
            mel = mel.half()

        audio_features: Tensor = self.model.encoder(mel, kv_cache=self.enc_kv_cache, mask=None)
        
        if audio_features.dtype != (
            torch.float16 if self.options.fp16 else torch.float32
        ):
            return TypeError(
                f"audio_features has an incorrect dtype: {audio_features.dtype}"
            )

        # Usually will run with this config
        if self.options.use_ca_kv_cache:
            return audio_features

        # update audio_features
        end_index = (self.mel.shape[-1] // 2) - 1
        
        if end_index == (self.model.encoder.gran * (1 + self.model.encoder.extra_gran_blocks)):
            start_index = 0
        else:
            start_index = end_index - self.model.encoder.gran
        
        if start_index % self.model.encoder.gran != 0:
            modolu_res = start_index % self.model.encoder.gran
            steps = self.model.encoder.gran - modolu_res
            start_index += steps
            end_index = start_index + self.model.encoder.gran

        self.audio_features[:, start_index:end_index] = audio_features

    def _detect_language(self, audio_features: Tensor, tokens: Tensor):
        languages = [self.options.language] * audio_features.shape[0]
        lang_probs = None

        if self.options.language is None or self.options.task == "lang_id":
            lang_tokens, lang_probs = self.model.detect_language(
                audio_features, self.tokenizer
            )
            languages = [max(probs, key=probs.get) for probs in lang_probs]
            if self.options.language is None:
                tokens[:, self.sot_index + 1] = lang_tokens  # write language tokens

        return languages, lang_probs

    def _set_ca_kv_cache(self, value: bool):
        self.model.use_ca_cache_hook = value

    def _main_loop(self, audio_features: Tensor):
        """
        in streaming we use self.tokens, since we need to keep in context the last tokens we got.
        """
        is_first_frame = self.index == (self.options.gran * (1 + self.options.look_ahead_blocks))
        self._set_ca_kv_cache(True)
        beam_indices = None
        # print(f"{audio_features.shape=}")

        try:
            for i in range(self.sample_len // 8):

                if self.tokens.shape[-1] > self.n_ctx: break 

                if self.options.stream_decode and self.options.beam_size > 0 and self.options.use_kv_cache: # We are in beam search. kv cache
                    last_valid_token_indices = [self.decoder._get_last_valid_token_index(self.tokens[i]) for i in range(self.tokens.shape[0])]
                    beam_indices_cols = last_valid_token_indices if i > 0 else sum([list(range(index - self.options.n_tokens_look_back, index + 1)) for index in last_valid_token_indices], [])
                    beam_indices_rows = [i // (self.options.n_tokens_look_back + 1) for i in range(self.options.beam_size * (self.options.n_tokens_look_back + 1))] if i==0 else list(range(self.options.beam_size))
                    beam_indices_cols_input = [item - beam_indices_cols[(j // (len(beam_indices_cols) // self.options.beam_size)) * (len(beam_indices_cols) // self.options.beam_size)] for j, item in enumerate(beam_indices_cols)]
                    beam_indices = [beam_indices_rows, beam_indices_cols, beam_indices_cols_input]
                
                logits = self.inference.logits(self.tokens, audio_features, i==0, beam_indices) # run inference through decoder

                # after the first decoder forward pass, no need to cache
                if i == 0: self._set_ca_kv_cache(False)

                if i == 0 and self.tokenizer.no_speech is not None:  # save no_speech_probs
                    probs_at_sot = logits[:, self.sot_index].float().softmax(dim=-1)
                    self.no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech].tolist()

                if isinstance(self.decoder, StreamingDecoder) or isinstance(self.decoder, BeamStreamingDecoder): # We need tokens for context
                    logits = logits[:, :]
                else: # we need to consider the logits at the last token only
                    logits = logits[:, -1]

                # apply the logit filters, e.g. for suppressing or applying penalty to
                for logit_filter in self.logit_filters:
                    logit_filter.apply(logits, self.tokens)

                # expand the tokens tensor with the selected next tokens
                if isinstance(self.decoder, BeamStreamingDecoder):
                    self.tokens, completed = self.decoder.update(self.tokens, logits, self.sum_logprobs, first_frame=is_first_frame, check_tokens=(i == 0))
                elif isinstance(self.decoder, StreamingDecoder):
                    self.tokens, completed = self.decoder.update(self.tokens, logits, self.sum_logprobs, first_frame=is_first_frame, check_tokens=(i == 0), index=(self.index * 20))
                else:
                    self.tokens, completed = self.decoder.update(self.tokens, logits, self.sum_logprobs)
                
                if completed: # ctx is unlimited, we are limited only by i
                    break
        finally:
            if self.options.use_kv_cache:
                self.inference.flush_tokens_from_cache()
            
            if isinstance(self.decoder, StreamingDecoder) or isinstance(self.decoder, BeamStreamingDecoder):
                
                if is_first_frame and self.options.streaming_timestamps and self.options.force_first_tokens_timestamps:
                    self.decoder._insert_timestamps(audio_features, self.tokens, self.options.gran)
                
                # if self.tokens.shape[1] > logits.shape[1]:
                #     self.tokens = self.tokens[:, :logits.shape[1]]
                
                self.decoder.reset()

        return self.sum_logprobs, self.no_speech_probs

    def _pad_audio_features(self, audio_features: Tensor):
        to_pad = 1500 - audio_features.shape[1]
        padding = torch.zeros((1, to_pad, audio_features.shape[2])).to(audio_features.device)
        audio_features = torch.cat([audio_features, padding], dim=1).to(audio_features.device)
        return audio_features

    def _empty_results(self):
        return DecodingResult(
                audio_features=None,
                language="en",
                tokens=[],
                text="",
                avg_logprob=None,
                no_speech_prob=None,
                temperature=self.options.temperature,
                compression_ratio=0,
            )

    def _caching_inner_reset(self):
        # Encoder kv cache clean
        self._cleanup_encoder_caching()
        self._init_enc_kv_caching()

        # Decoder CA kv-cache
        if self.options.use_ca_kv_cache:
            self._cleanup_ca_caching()
            self._init_ca_kv_caching()
        
        # Decoder SA kv-cache
        if self.options.use_kv_cache:
            self.inference.cleanup_caching()
            # Caching will be triggered on the logits function

    def _reset_after_maximal_context(self, new_mel_frame: Tensor):
        print("Reset context...")
        num_old_mels = (self.options.gran * (self.options.look_ahead_blocks) * 2) + 2 if self.options.look_ahead_blocks > 0 else (self.options.gran * 2) + 2
        self.mel = torch.cat([self.mel[..., -num_old_mels:], new_mel_frame], dim=-1)
        self.options.prefix = self.tokens[:, len(self.sot_sequence):].tolist()[0][-self.options.n_tokens_look_back-5:]
        
        # save retired tokens
        self.retired_tokens = self.tokens[:, :len(self.sot_sequence)]

        print(f"Modifying tokens! {self.options.prefix=}")
        self.initial_tokens = self._get_initial_tokens()
        print("Modified tokens!")
        
        self.tokens = torch.tensor([list(self.initial_tokens)]) # no need to use repeat, batch is meaningless in stream.
        self.tokens = self.tokens.repeat_interleave(self.n_group, dim=0).to(self.model.device)
        
        print(f"Modifying tokens! {self.tokens=}")
        self._caching_inner_reset()
        self.audio_features = torch.zeros((1, self.model.dims.n_audio_ctx, self.model.dims.n_audio_state)).to(self.model.device)
        print("Finished reset...")

    def _clean_transcription_timestamps(self, text: str) -> str:
        """
        Removes punctuation and their immediately following timestamps 
        from a BPE-encoded transcription string.
        """
        # Pattern explanation:
        # [,.:;!?]      -> Matches the punctuation mark
        # <\|           -> Matches the opening tag <|
        # \d+           -> Matches one or more digits (d or dd)
        # \.            -> Matches the decimal point
        # \d+           -> Matches one or more digits after the decimal
        # \|>           -> Matches the closing tag |>
        pattern = r'[,.:;!?]<\|\d+\.\d+\|>'

        # 1. Remove the punctuation + timestamp pairs
        cleaned = re.sub(pattern, '', text)

        # 2. Fix potential double spaces left behind and trim edges
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned

    @torch.no_grad()
    def run(self, mel_frame: Tensor) -> List[DecodingResult]:
        """
        mel_frame - a tensor containing the last frame we got from the stream.
        
        The function needs to: 
        1. Take the mel frame.
        2. Extract features from the given frame using kv caching in the encoder.
        3. Note that the caching in the encoder will do it automatically.
        """
        # concat mel
        if self.options.single_frame_mel: # each time we get a single frame of mel
            self.mel = torch.cat([self.mel, mel_frame], dim=-1) if self.mel is not None else mel_frame
        else: # we are getting a whole frame [0, t_curr]
            self.mel = mel_frame
        tokenizer: Tokenizer = self.tokenizer
        n_audio: int = mel_frame.shape[0]

        self.index += self.options.gran # on each decoding call we add more context
        self.frame_counter += 1
        # print(f"Collected {self.frame_counter} frames...")

        if self.mel.shape[-1] >= self.options.maximal_seconds_context * 100: 
            self._reset_after_maximal_context(mel_frame)
        
        # print(f"{self.mel.shape=}")
        if self.mel.shape[-1] < (self.options.gran * 2 * (self.options.look_ahead_blocks + 1)):
            # print("Decoding Task: skipping first frames...")
            return self._empty_results()
        
        # call the main sampling loop
        if not self.options.use_ca_kv_cache:
            self._get_audio_features(self.mel) # encoder forward pass, updates self.audio_features
            audio_features = self.audio_features
            sum_logprobs, no_speech_probs = self._main_loop(audio_features[:, :self.index])
        else:
            audio_features = self._get_audio_features(self.mel)
            sum_logprobs, no_speech_probs = self._main_loop(audio_features)


        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[:: self.n_group]
        no_speech_probs = no_speech_probs[:: self.n_group]

        self.tokens = self.tokens.reshape(n_audio, self.n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, self.n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = self.decoder.finalize(self.tokens, sum_logprobs)
        tokens: List[List[Tensor]] = [[t[self.sample_begin: ((t == tokenizer.eot) | (t == tokenizer.sot_lm)).nonzero()[0, 0]] for t in s] for s in tokens]
        
        
        # if any of the suggested beams is empty (no token), it means that the predicted token was EOT. Add it, so ML Ranker won't fail.
        for i in range(len(tokens[0])):
            if tokens[0][i].shape[0] > 0: continue
            else: tokens[0][i] = torch.tensor([self.tokenizer.eot]).to(tokens[0][i].device)

        # select the top-ranked sample in each group
        selected = self.sequence_ranker.rank(tokens, sum_logprobs)
        tokens: List[List[int]] = [t[i].tolist() for i, t in zip(selected, tokens)]
        
        texts: List[str] = [tokenizer.decode(t).strip() for t in tokens]
        sum_logprobs: List[float] = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs: List[float] = [
            lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)
        ]

        fields = (texts, tokens, audio_features, avg_logprobs, no_speech_probs)
        
        self.tokens = self.tokens.squeeze(0)

        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        # apply timestamps
        if (self.options.beam_size == 0 or self.options.beam_size is None) and hasattr(self.decoder, 'timestamps_map'):
            timed_tokens = tokens.copy()[0]
            for i, index in enumerate(sorted(self.decoder.timestamps_map.keys())):
                timed_tokens.insert(index + i + 1, self.tokenizer.timestamp_begin + (self.decoder.timestamps_map[index] // 20))
        else:
            timed_tokens = [50257]

        return DecodingResult(
                audio_features=audio_features,
                language="en",
                tokens=tokens,
                text=texts[0],
                avg_logprob=avg_logprobs,
                no_speech_prob=no_speech_probs,
                temperature=self.options.temperature,
                compression_ratio=compression_ratio(texts[0]),
                timestamps=None if not hasattr(self.decoder, 'timestamps_map') else self.decoder.timestamps_map,
                timed_tokens=timed_tokens,
                timed_text=self._clean_transcription_timestamps(self.tokenizer.decode_with_timestamps(timed_tokens))
            )
    

@torch.no_grad()
def decode(
    model: "Whisper",
    mel: Tensor,
    task: DecodingTask,
    options: DecodingOptions = DecodingOptions(),
    **kwargs,
) -> Union[DecodingResult, List[DecodingResult]]:
    """
    Performs decoding of 30-second audio segment(s), provided as Mel spectrogram(s).

    Parameters
    ----------
    model: Whisper
        the Whisper model instance

    mel: torch.Tensor, shape = (80, 3000) or (*, 80, 3000)
        A tensor containing the Mel spectrogram(s)

    options: DecodingOptions
        A dataclass that contains all necessary options for decoding 30-second segments

    Returns
    -------
    result: Union[DecodingResult, List[DecodingResult]]
        The result(s) of decoding contained in `DecodingResult` dataclass instance(s)
    """
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)

    if kwargs:
        options = replace(options, **kwargs)
    
    result = DecodingTask(model, options).run(mel)    
    return result

