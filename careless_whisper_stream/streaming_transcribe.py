import sys
sys.path.append('.')

import time

import argparse
from typing import TYPE_CHECKING, List

import numpy as np
import torch

from .audio import (
    SAMPLE_RATE,
    SpectrogramStream,
    MyStream
)
from .streaming_decoding import DecodingOptions

if TYPE_CHECKING:
    from .streaming_model import StreamingWhisper

        
def transcribe(
    model: "StreamingWhisper" = None,
    output_filename: str = None,
    channels: int = 2,
    language: str = "en",
    simulate_stream: bool = False,
    wav_file: str = None,
    single_frame_mel: bool = True,
    temperature: float = 0,
    beam_size: int = 5,
    stream_decode: bool = True,
    ca_kv_cache: bool = False,
    sa_kv_cache: bool = False,
    use_latency: bool = False,
    get_times: bool = False,
    pad_trim: bool = False,
    max_sec_context: int = 30,
    streaming_timestamps: bool = False,
    force_first_tokens_timestamps: bool = False,
    ms_granularity: int = None,
    extra_initial_blocks: int = None,
    **kwargs
) -> List[str]:
    """
    Open a stream and transcribe it using streaming whisper model

    A very thin implementation of the transcribe function, compared to Whisper implementation.

    Parameters
    ----------
    model: Whisper
        The Whisper model instance

    Returns - 
    -------
    A dict with a text, tokens field with all of the text that was transcribed till the stream stopped.
    """
    model.reset(use_stream=True) # we first reset the model before starting a stream, cleaning any cache.
    model.eval()
    
    # Instantiate streaming instance and open a stream
    ms_gran = model.encoder.gran * 20 if ms_granularity is None else ms_granularity
    assert ms_gran % 20 == 0, "ms_granularity must be a multiple of 20"
    stream_instance = MyStream(ms_gran,
                               channels=channels,
                               filename=output_filename, 
                               simulate_stream=simulate_stream, 
                               wav_file=wav_file, 
                               use_latency=use_latency, 
                               pad_trim=pad_trim,)
    
    stream_instance.open_stream()
    
    # frames - used only when filename is given, in order to save a long wav at the end of the conversation.
    frames = []

    extra_gran_blocks = extra_initial_blocks if extra_initial_blocks is not None else model.encoder.extra_gran_blocks

    # first we'll use
    decoding_options = DecodingOptions(
        language=language,
        gran=(ms_gran // 20),
        single_frame_mel=single_frame_mel,
        without_timestamps=True,
        beam_size=beam_size if temperature == 0 else None,
        temperature=temperature,
        length_penalty=None,
        look_ahead_blocks=extra_gran_blocks,
        patience=None,
        stream_decode=stream_decode,
        use_kv_cache=sa_kv_cache,
        use_ca_kv_cache=ca_kv_cache,
        streaming_timestamps=streaming_timestamps,
        force_first_tokens_timestamps=force_first_tokens_timestamps,
        **kwargs
    )

    streamed_spectrogram = SpectrogramStream(n_mels=model.dims.n_mels) # default values are whisper default values

    texts = []
    times = []
    reset_len = (max_sec_context) * SAMPLE_RATE + 360 # 360 is for the mel padding
    chunk_samples = stream_instance.chunk_size
    full_text = ""
    try:
        for frame in stream_instance.read():
            # save frames for optional save
            frames.extend(frame)
            if len(frames) >= reset_len: # When we surpass the max_sec_context - reset model (positional embeddings constrain us)
                frame = np.concatenate((frames[-360:], frame))
                frames = []
                frames.extend(frame.tolist())
                model.reset(use_stream=True)
                streamed_spectrogram.reset()
                full_text += " " + texts[-1].text if len(texts) > 0 else ""

            if get_times:
                torch.cuda.synchronize()
                start = time.time()

            frame_tensor = torch.from_numpy(frame).pin_memory()
            mel_frame = streamed_spectrogram.calc_mel_with_new_frame(frame_tensor.to(model.device, non_blocking=True))
            
            # decode given the new mel frame and print results
            result = model.decode(mel_frame.squeeze(0), decoding_options)

            if get_times:
                torch.cuda.synchronize()
                end = time.time()
                times.append(end - start)

            # useful for debugging
            # print(full_text +  " "  + result.text)
            
            result.full_text = full_text + " " + result.text
            texts.append(result)

    except KeyboardInterrupt:
        stream_instance.close_stream(frames)
    
    print("Finished capturing audio.")
    
    return texts, times


def cli():
    parser = argparse.ArgumentParser(description="Transcribe streaming audio with customizable options")

    # Model choices
    parser.add_argument("--model", type=str, default="small", help="Model size to transcribe with")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run model inference on.")
    parser.add_argument("--chunk_size", type=int, default=300, help="Chunk size for streaming")
    parser.add_argument("--multilingual", action="store_true", help="Use a multilingual checkpoint if exists.", default=False)

    # Local streaming args
    parser.add_argument("--output_filename", type=str, help="Path to the output audio file when using local streaming")
    parser.add_argument("--channels", type=int, default=2, help="Number of audio channels - relevant for local streaming")

    # Streaming simulation wav file
    parser.add_argument("--wav_file", type=str, help="Optional WAV file path to stream, using a stream simulation")

    # Streaming behavior
    parser.add_argument("--simulate_stream", action="store_true", help="Simulate a stream from a file")
    parser.add_argument("--single_frame_mel", action="store_true", default=True, help="Use single frame MELs")
    parser.add_argument("--stream_decode", action="store_true", default=True, help="Use streaming decode")
    parser.add_argument("--ca_kv_cache", action="store_true", help="Use cross-attention key-value cache")
    parser.add_argument("--sa_kv_cache", action="store_true", help="Use self-attention key-value cache")
    parser.add_argument("--wait_for_all", action="store_true", help="Wait for all results before outputting")
    parser.add_argument("--use_latency", action="store_true", help="Add latency for streaming simulation")
    parser.add_argument("--pad_trim", action="store_true", default=False, help="Enable padding and trimming")
    parser.add_argument("--streaming_timestamps", action="store_true", help="Use timestamps in streaming")
    parser.add_argument("--force_first_tokens_timestamps", action="store_true", help="Force timestamps on first tokens")

    # Model behavior
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--beam_size", type=int, default=5, help="Beam size for beam search decoding")
    parser.add_argument("--language", type=str, default="en", help="Language of transcription")
    parser.add_argument("--max_sec_context", type=int, default=30, help="Max context window size in seconds")

    args = parser.parse_args().__dict__

    from . import load_streaming_model

    model_size: str = args.pop("model")
    chunk_size: int = args.pop("chunk_size")
    multilingual: bool = args.pop("multilingual")
    device: str = args.pop("device")

    model = load_streaming_model(model_size, chunk_size, multilingual, device)

    texts = transcribe(model, **args)
    return texts

if __name__ == "__main__":
    cli()

