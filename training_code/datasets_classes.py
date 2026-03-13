import os
import lmdb
import torch
import pickle
import torchaudio
import numpy as np
import pandas as pd
import careless_whisper_stream
import careless_whisper_stream.tokenizer
from praatio import textgrid
from dataclasses import dataclass
from torchaudio.datasets.utils import _extract_tar
from torchaudio._internal import download_url_to_file
from careless_whisper_stream.tokenizer import Tokenizer
from careless_whisper_stream.audio import SpectrogramStream

from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm

class WAVsTextsDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path: str, sep="\t", custom_len: int = 0):
        super().__init__()

        self.ds_df = pd.read_csv(ds_path, sep=sep, index_col=False)
        self.custom_len = custom_len

    def __len__(self):
        return int(self.custom_len) if 0 < self.custom_len < len(self.ds_df) else int(len(self.ds_df))

    def __getitem__(self, index):
        item = self.ds_df.iloc[index]
        return dict(wav_path=item["wav_path"], text=item["raw_text"])


class WAVsDataset(torch.utils.data.Dataset):
    def __init__(self, ds_path: str, 
                 sep="\t", 
                 tokenizer: careless_whisper_stream.tokenizer = None, 
                 no_labels: bool = False, 
                 custom_len: int = 0,
                 get_streamed_mel: bool = False) -> None:
        super().__init__()

        if not no_labels:
            self.tokenizer = tokenizer if tokenizer else careless_whisper_stream.tokenizer.get_tokenizer(True, language="en", task="transcribe")
        self.ds_df = pd.read_csv(ds_path[0], sep=sep, index_col=False)
        self.sr = 16_000
        self.no_labels = no_labels
        self.custom_len = custom_len
        self.get_streamed_mel = get_streamed_mel

    def __len__(self):
        return int(self.custom_len) if 0 < self.custom_len < len(self.ds_df) else int(len(self.ds_df))

    def _calc_mel(self, audio):
        if self.get_streamed_mel:
            spec_streamer = SpectrogramStream()
            return spec_streamer._simulate_streaming_log_spec(torch.tensor(audio)).squeeze(0)
            
        return careless_whisper_stream.log_mel_spectrogram(audio)

    def __getitem__(self, idx):
        item = self.ds_df.iloc[idx]

        audio = careless_whisper_stream.load_audio(item["wav_path"], sr=self.sr)
        audio = careless_whisper_stream.pad_or_trim(audio.flatten())
        mel = self._calc_mel(audio)
        
        if self.no_labels: return dict(input_ids=mel)
        
        text = item["raw_text"]
        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        # maximal context available on whisper model is 448 tokens
        if len(labels) > 448:
            labels = labels[:448]
            text = text[:447] + [self.tokenizer.eot]

        return dict(input_ids=mel, labels=torch.tensor(labels), dec_input_ids=torch.tensor(text), endpoints=torch.tensor([audio.shape[-1] / self.sr for _ in range(len(labels))]))


@dataclass
class Interval:
    label: str = None
    start: float = 0.0
    end: float = 0.0

class AlignedTextGridDataset(torch.utils.data.Dataset):
    def __init__(self,
                 ds_path: Union[str, list], 
                 tokenizer: Tokenizer = None, 
                 sample_rate: int = 16_000, 
                 custom_len: int = 0,
                 get_streamed_mel: bool = False,
                 gran: int = 15,
                 extra_gran_blocks: int = 0,
                 n_mels: int = 80,
                 multilingual: bool = False,
                 separator='\t',
                 split='train'): # most of the times we train just on english librispeech
        super().__init__()

        self.tokenizer = tokenizer if tokenizer else careless_whisper_stream.tokenizer.get_tokenizer(True, language="en", task="transcribe")
        print("Reading ds")
        self.ds_df = pd.concat([pd.read_csv(path, sep=separator) for path in ds_path], ignore_index=True)
        print("finished Reading ds, its length: ", len(self.ds_df), len(self.alignments_cache))
        self.sr = sample_rate
        self.custom_len = custom_len
        self.get_streamed_mel = get_streamed_mel
        self.gran = gran
        self.extra_gran_blocks = extra_gran_blocks
        self.n_mels = n_mels
        self.multilingual = multilingual

    def __len__(self):
        return int(self.custom_len) if 0 < self.custom_len < len(self.ds_df) else len(self.ds_df)
    
    def _calc_mel(self, audio):
        if self.get_streamed_mel:
            spec_streamer = SpectrogramStream(n_mels=self.n_mels)
            return spec_streamer._simulate_streaming_log_spec(torch.tensor(audio))
            
        return careless_whisper_stream.log_mel_spectrogram(audio)

    def _get_intervals_from_wrd_file(self, path: str):
        with open(path, "r") as file:
            lines = file.readlines()
        
        intervals = []
        for line in lines:
            start, end, label = line.strip().split()
            intervals.append(Interval(label, int(start) / self.sr, int(end) / self.sr))
        
        return intervals

    def __getitem__(self, index):
        item = self.ds_df.iloc[index]

        audio = careless_whisper_stream.pad_or_trim(careless_whisper_stream.load_audio(item["wav_path"], sr=self.sr))
        mel = self._calc_mel(audio)
        
        if ".wrd" in item["tg_path"]:
            text_intervals = self._get_intervals_from_wrd_file(item["tg_path"])
        else:
            path = Path(item["tg_path"]) if not os.path.exists(Path(item["tg_path"]).with_suffix('.punc')) else Path(item["tg_path"]).with_suffix('.punc')
            tg = textgrid.openTextgrid(path, includeEmptyIntervals=False)
            text_intervals = tg.getTier("words") if path.suffix == '.TextGrid' else tg.getTier("words_punctuated")

        tokenizer = self.tokenizer if not self.multilingual else careless_whisper_stream.tokenizer.get_tokenizer(True, language=item["lang"], task="transcribe")

        endpoints = [0, 0, 0]
        tokens = []
        for i, interval in enumerate(text_intervals):
            curr_tokens = self.tokenizer.encode(interval.label if i == 0 else " " + interval.label)
            n_diff = (interval.end - interval.start) / len(curr_tokens)
            endpoints.extend([interval.start + (i + 1) * n_diff for i in range(len(curr_tokens))])
            tokens.extend(curr_tokens)
        
        text = [*tokenizer.sot_sequence_including_notimestamps] + tokens
        labels = text[1:] + [self.tokenizer.eot]
        endpoints.append(endpoints[-1] + 0.5)
        
        assert len(endpoints) == len(labels) == len(text)

        return dict(input_ids=mel,
                    dec_input_ids=torch.tensor(text),
                    labels=torch.tensor(labels),
                    endpoints=torch.tensor(endpoints))


class TIMIT(torch.utils.data.Dataset):
    def __init__(self, ds_path: str, tokenizer: Tokenizer = None, n_state: int = 384) -> None:
                
        self.tokenizer = tokenizer if tokenizer else careless_whisper_stream.tokenizer.get_tokenizer(True, language="en", task="transcribe")

        with open(ds_path, 'rb') as file:
            self.dataset = pickle.load(file)

        self.n_state = n_state
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        audio, sr, text, _, _ = self.dataset[index]
        audio_len = audio.shape[-1]
        assert sr == 16000
        audio = careless_whisper_stream.pad_or_trim(torch.Tensor(audio).flatten())
        mel = careless_whisper_stream.log_mel_spectrogram(audio)

        text = [*self.tokenizer.sot_sequence_including_notimestamps] + self.tokenizer.encode(text)
        labels = text[1:] + [self.tokenizer.eot]

        num_frames = ((audio_len // 16000) * 50) + 2
        mask = torch.ones(1, 1500, self.n_state)
        mask[0, num_frames:, :] = 0

        return dict(
            input_ids=mel,
            labels=labels,
            dec_input_ids=text,
            mask=mask,
        )

class AlignedTextGridDatasetLMDB(torch.utils.data.Dataset):
    def __init__(self,
                 ds_path: Union[str, list], 
                 lmdb_paths: Optional[Dict[str, str]] = None, # e.g., {"librispeech": "path/to/libri.lmdb"}
                 tokenizer = None, 
                 sample_rate: int = 16_000, 
                 custom_len: int = 0,
                 get_streamed_mel: bool = False,
                 gran: int = 15,
                 extra_gran_blocks: int = 0,
                 n_mels: int = 80,
                 multilingual: bool = False,
                 separator='\t',
                 split='train'):
        super().__init__()

        self.tokenizer = tokenizer if tokenizer else careless_whisper_stream.tokenizer.get_tokenizer(True, language="en", task="transcribe")
        print("Reading ds")
        self.ds_df = pd.concat([pd.read_csv(path, sep=separator) for path in ds_path], ignore_index=True)
        print("finished Reading ds, its length: ", len(self.ds_df))
        
        self.sr = sample_rate
        self.custom_len = custom_len
        self.get_streamed_mel = get_streamed_mel
        self.gran = gran
        self.extra_gran_blocks = extra_gran_blocks
        self.n_mels = n_mels
        self.multilingual = multilingual
        
        # LMDB Setup
        self.lmdb_paths = lmdb_paths
        self.envs = {} # Will hold opened environments per worker process

    def _init_lmdb(self, db_name):
        """Initializes LMDB environment for the current process."""
        if db_name not in self.envs:
            print("Initializing LMDB for database:", db_name)
            self.envs[db_name] = lmdb.open(
                self.lmdb_paths[db_name],
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256
            )

    def __len__(self):
        return int(self.custom_len) if 0 < self.custom_len < len(self.ds_df) else len(self.ds_df)
    
    def _calc_mel(self, audio):
        if self.get_streamed_mel:
            spec_streamer = SpectrogramStream(n_mels=self.n_mels)
            return spec_streamer._simulate_streaming_log_spec(torch.tensor(audio))
        return careless_whisper_stream.log_mel_spectrogram(audio)

    def _get_intervals_from_wrd_file(self, path: str):
        with open(path, "r") as file:
            lines = file.readlines()
        
        intervals = []
        for line in lines:
            start, end, label = line.strip().split()
            intervals.append(Interval(label, int(start) / self.sr, int(end) / self.sr))
        return intervals

    def __getitem__(self, index):
        item = self.ds_df.iloc[index]

        # 1. Load Audio
        audio = careless_whisper_stream.pad_or_trim(careless_whisper_stream.load_audio(item["wav_path"], sr=self.sr))
        mel = self._calc_mel(audio)
        
        # 2. Load Intervals (Logic Check: LMDB vs File System)
        text_intervals = None
        
        # Check if this item specifies an LMDB source (e.g., if you added a 'db_name' column to your CSV)
        db_name = 'VOXP-EN-ALIGNED' if 'LIBRI' not in item.get('tg_path') else 'LIBRI-960-ALIGNED'

        if self.lmdb_paths and db_name in self.lmdb_paths:
            self._init_lmdb(db_name)
            with self.envs[db_name].begin(write=False, buffers=True) as txn:
                # We use the tg_path or a unique ID as the key
                wrd_bytes = txn.get(item["tg_path"].encode('ascii'))
                if wrd_bytes:
                    text_intervals = pickle.loads(bytes(wrd_bytes))

        # Fallback to original file logic if not found in LMDB
        used_fallback = False
        if text_intervals is None:
            used_fallback = True
            if ".wrd" in item["tg_path"]:
                text_intervals = self._get_intervals_from_wrd_file(item["tg_path"])
            else:
                # Assuming praatio/textgrid is used here
                path = Path(item["tg_path"]) if not os.path.exists(Path(item["tg_path"]).with_suffix('.punc')) else Path(item["tg_path"]).with_suffix('.punc')
                tg = textgrid.openTextgrid(path, includeEmptyIntervals=False)
                text_intervals = tg.getTier("words") if path.suffix == '.TextGrid' else tg.getTier("words_punctuated")

        # 3. Process Tokens (Original Logic)
        tokenizer = self.tokenizer if not self.multilingual else careless_whisper_stream.tokenizer.get_tokenizer(True, language=item["lang"], task="transcribe")

        endpoints = [0, 0, 0]
        tokens = []
        for i, interval in enumerate(text_intervals):
            curr_tokens = self.tokenizer.encode(interval.label if i == 0 else " " + interval.label)
            n_diff = (interval.end - interval.start) / len(curr_tokens)
            endpoints.extend([interval.start + (i + 1) * n_diff for i in range(len(curr_tokens))])
            tokens.extend(curr_tokens)
        
        text = [*tokenizer.sot_sequence_including_notimestamps] + tokens
        labels = text[1:] + [self.tokenizer.eot]
        endpoints.append(endpoints[-1] + 0.5)
        
        assert len(endpoints) == len(labels) == len(text)

        return dict(input_ids=mel,
                    dec_input_ids=torch.tensor(text),
                    labels=torch.tensor(labels),
                    endpoints=torch.tensor(endpoints))

### Taken from Eyal Cohen's TEDLIUM loader ###

_RELEASE_CONFIGS = {
    "release1": {
        "folder_in_archive": "TEDLIUM_release1",
        "url": "http://www.openslr.org/resources/7/TEDLIUM_release1.tar.gz",
        "checksum": "30301975fd8c5cac4040c261c0852f57cfa8adbbad2ce78e77e4986957445f27",
        "data_path": "",
        "subset": "train",
        "supported_subsets": ["train", "test", "dev"],
        "dict": "TEDLIUM.150K.dic",
    },
    "release2": {
        "folder_in_archive": "TEDLIUM_release2",
        "url": "http://www.openslr.org/resources/19/TEDLIUM_release2.tar.gz",
        "checksum": "93281b5fcaaae5c88671c9d000b443cb3c7ea3499ad12010b3934ca41a7b9c58",
        "data_path": "",
        "subset": "train",
        "supported_subsets": ["train", "test", "dev"],
        "dict": "TEDLIUM.152k.dic",
    },
    "release3": {
        "folder_in_archive": "TEDLIUM_release-3",
        "url": "http://www.openslr.org/resources/51/TEDLIUM_release-3.tgz",
        "checksum": "ad1e454d14d1ad550bc2564c462d87c7a7ec83d4dc2b9210f22ab4973b9eccdb",
        "data_path": "data/",
        "subset": "train",
        "supported_subsets": ["train", "test", "dev"],
        "dict": "TEDLIUM.152k.dic",
    },
}

class TEDLIUM(Dataset):
    def __init__(
        self,
        root: Union[str, Path] = "/mlspeech/data/eyalcohen/datasets",
        release: str = "release1",
        subset: str = "train",
        download: bool = False,
        audio_ext: str = ".sph",
    ) -> None:
        self._ext_audio = audio_ext
        self.root = Path(root)
        self.base_path = ""
        if release in _RELEASE_CONFIGS.keys():
            config = _RELEASE_CONFIGS[release]
            folder_in_archive = Path(config["folder_in_archive"])
            url = Path(config["url"])
            subset = subset if subset else config["subset"]
        else:
            raise RuntimeError(
                f"The release {release} does not match any of the supported TEDLIUM releases {list(_RELEASE_CONFIGS.keys())}"
            )

        if subset not in config["supported_subsets"]:
            raise RuntimeError(
                f"The subset {subset} does not match any of the supported TEDLIUM subsets {config['supported_subsets']}"
            )
        self.base_path = self.root / folder_in_archive

        if release == "release3":
            if subset == "train":
                self._path = self.root / folder_in_archive / config["data_path"]
            else:
                self._path = self.root / folder_in_archive / "legacy" / subset
        else:
            self._path = self.root / folder_in_archive / config["data_path"] / subset

        self.audio_artifacts_dir = self.root / folder_in_archive / "artifacts" / subset / "audio"
        self.audio_artifacts_dir.mkdir(parents=True, exist_ok=True)

        if download:
            archive = self.root / url.name
            if not self._path.is_dir():
                if not archive.is_file():
                    download_url_to_file(url, str(archive), hash_prefix=config["checksum"])
                _extract_tar(str(archive), extract_path=str(self.root))
        else:
            if not os.path.exists(self._path):
                raise RuntimeError(
                    f"The path {self._path} doesn't exist. "
                    "Please check the ``root`` path or set `download=True` to download it"
                )

        self._dict_path = self.root / folder_in_archive / config["dict"]
        self._phoneme_dict = None
        self._filelist = self._prepare_filelist(self._path / "stm")

    def _prepare_filelist(self, stm_path: Path):
        filelist = []
        for file in sorted(stm_path.iterdir()):
            if file.suffix == ".stm":
                with open(file, "r") as f:
                    lines = f.readlines()
                    filelist.extend(
                        [(file.stem, i) for i, line in enumerate(lines) if "ignore_time_segment_in_scoring" not in line]
                    )
        return filelist

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, int, int, int]:
        fileid, line_idx = self._filelist[n]
        audio_path = self.audio_artifacts_dir / f"{fileid}_{line_idx}.wav"
        transcript_path = audio_path.with_suffix(".txt")

        if audio_path.exists() and transcript_path.exists():
            waveform, sample_rate = torchaudio.load(audio_path)
            with open(transcript_path, "r") as f:
                transcript = f.read().strip()
        else:
            waveform, sample_rate, transcript, talk_id, speaker_id, identifier = self._load_tedlium_item(
                fileid, line_idx, self._path
            )
            transcript = transcript.replace(" '", "'")
            torchaudio.save(audio_path, waveform, sample_rate)
            with open(transcript_path, "w") as f:
                f.write(transcript)
        
        # In the TEDLIUM dataset, " '" is used as a separator subword and should be replaced by "'"
        transcript = transcript.replace(" '", "'")
        
        return dict(wav_path=waveform,
                    text=transcript,
                    real_path=str(audio_path))

    def _load_tedlium_item(self, fileid: str, line: int, path: Path) -> Tuple[Tensor, int, str, int, int, int]:
        """Loads a TEDLIUM dataset sample given a file name and corresponding sentence name.

        Args:
            fileid (str): File id to identify both text and audio files corresponding to the sample
            line (int): Line identifier for the sample inside the text file
            path (str): Dataset root path

        Returns:
            (Tensor, int, str, int, int, int):
            ``(waveform, sample_rate, transcript, talk_id, speaker_id, identifier)``
        """
        stm_file_path = path / "stm" / f"{fileid}.stm"
        with open(stm_file_path) as f:
            lines = f.readlines()
        talk_id, _, speaker_id, start_time, end_time, identifier, transcript = lines[line].split(" ", 6)

        wave_path = path / "sph" / f"{fileid}{self._ext_audio}"
        waveform, sample_rate = self._load_audio(wave_path, float(start_time), float(end_time))
        return (waveform, sample_rate, transcript.strip(), talk_id, speaker_id, identifier)

    def _load_audio(
        self, path: Path, start_time: float, end_time: float, sample_rate: int = 16000
    ) -> Tuple[Tensor, int]:
        start_frame = int(start_time * sample_rate)
        end_frame = int(end_time * sample_rate)
        num_frames = end_frame - start_frame

        return torchaudio.load(path, frame_offset=start_frame, num_frames=num_frames)

    def __len__(self) -> int:
        return len(self._filelist)

    @property
    def phoneme_dict(self):
        if not self._phoneme_dict:
            with open(self._dict_path, "r", encoding="utf-8") as f:
                self._phoneme_dict = {line.split()[0]: tuple(line.split()[1:]) for line in f}
        return self._phoneme_dict.copy()

    def get_index_by_name(self, file_name: str) -> int:
        for i, (filelst_name, seg_id) in enumerate(self._filelist):
            if file_name == f"{filelst_name}_{seg_id}":
                return i
        raise ValueError(f"File '{file_name}' not found in the dataset")
    