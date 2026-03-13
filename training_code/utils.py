import argparse
from dataclasses import dataclass


@dataclass(frozen=False)
class Config:
    # DL Args
    learning_rate: float = 0.0005
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-6
    warmup_steps: int = 100
    batch_size: int = 16
    num_worker: int = 16
    num_train_epochs: int = 10
    gradient_accumulation_steps: int = 1
    sample_rate: int = 16000
    ckpt: str = None
    no_logger: bool = False
    dataset: str = None
    name: str = None
    top_k: int = -1
    early_stop: bool = False
    fast_dev_run: int = None
    strategy: str = "ddp"
    seed: int = 3407
    custom_len: int = 0
    lmdb: bool = False
    lmdb_paths: str = None
    self_supervision: bool = False

    # Whisper args
    lang: str = "en"
    size: str = "tiny"

    # LoRA + streaming args
    lora: bool = False
    lora_ckpt: str = None
    rank: int = 16
    gran: int = 15
    extra_gran_blocks: int = 1
    sim_stream: bool = False
    uniform_sampling: bool = False
    streaming_train: bool = False
    streaming_fraction: float = 1
    streaming_random: bool = False
    multilingual: bool = False
    slices_num: int = 20
    random_masking: bool = False

    use_from_ft_ckpt: bool = False

def parse_cmdl():
    # parser
    parser = argparse.ArgumentParser(description="Training whisper models, using different configurations")

    # switches for train.py 
    parser.add_argument('--lora', action="store_true", help="run LoRA training")
    parser.add_argument('--lmdb', action="store_true", help="use lmdb for dataset loading instead of disk reads (if available)")
    parser.add_argument('--no_logger', action="store_true", help="set logger to False")
    parser.add_argument('--use_from_ft_ckpt', action="store_true", help="Use from FT ckpts to resume training for ssl")
    parser.add_argument('--self_supervision', action="store_true", help="Self-supervised training mode")
    
    # variables for trainer
    parser.add_argument('--name', type=str, help="Trained model name", default="model")
    parser.add_argument('--size', type=str, help="Whisper size - can use only [tiny, base, small, medium, large, large-v2]", default="tiny")
    parser.add_argument('--ckpt', type=str, help="ckpt loading to resume training from", default=None)
    parser.add_argument('--fast_dev_run', type=int, help="run few dev runs for sanity checks on lightning trainer", default=None)
    parser.add_argument('--top_k', type=int, help="Top K checkpoints to save, use 1 for the best, -1 for last", default=1)
    parser.add_argument('--early_stop', action="store_true", help="Use early stopping callback")
    parser.add_argument('--custom_len', type=int, help="Number of samples to train on", default=0)
    
    # DL Hyper Parameters
    parser.add_argument('--epochs', type=int, help="Number of training epochs", default=10)
    parser.add_argument('--batch_size', type=int, help="Batch size for training and evaluation. Better be 2^n, where n is a positive integer.", default=16)
    parser.add_argument('--dataset', type=str, nargs='+', help="Name of dataset to load", default=['TIMIT-WORD'])
    parser.add_argument('--learning_rate', type=float, help="Custom learning rate", default=0.0001)
    parser.add_argument('--gacc', type=int, help="Number of gradient accumulation steps", default=1)
    parser.add_argument('--weight_decay', type=float, help="Weight decay factor", default=0.01)
    parser.add_argument('--adam_epsilon', type=float, help="Adam epsilon", default=1e-6)
    parser.add_argument('--warmup_steps', type=int, help="Scheduler warmup steps", default=100)
    parser.add_argument('--num_worker', type=int, help="Data loader workers", default=16)
    parser.add_argument('--strategy', type=str, help="Trainer strategy [ddp, fsdp, ddp_find_unused_parameters_true]", default="ddp")
    
    # LoRA 
    parser.add_argument('--lora_ckpt', type=str, help="ckpt loading (for LoRA training mode only)", default=None)
    parser.add_argument('--rank', type=int, help="LoRA rank", default=16)
    parser.add_argument('--gran', type=int, help="Granularity in encoder frames to calc attention on", default=15)
    parser.add_argument('--extra_gran_blocks', type=int, help="How many extra granularity blocks we add on encoder causal block matrix", default=1)
    parser.add_argument('--streaming_fraction', type=float, help="Fraction of the available streaming sample points to train on.", default=1)
    parser.add_argument('--num_slices', type=int, help="Number of slices to divide the streaming data into.", default=20)
    parser.add_argument('--simulate_stream', action="store_true", help="Spectrogram input is simulated to a stream scenario")
    parser.add_argument('--streaming_train', action="store_true", help="Train sequentially on a stream of data.")
    parser.add_argument('--streaming_random', action="store_true", help="Train using random sample points, not sequentially!")
    parser.add_argument('--random_masking', action="store_true", help="Train using random masking.")
    parser.add_argument('--multilingual', action="store_true", help="Train using multilingual dataset, assuming lang field is available.")

    return parser.parse_args()
