#!/usr/bin/env python3
import os
import torch
import faulthandler, sys
from pathlib import Path
from ds_dict_private import ds_paths
from whisper_module import LoRAStreamedWhisper
from training_code.utils import Config, parse_cmdl
from pytorch_lightning.loggers import  WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping

# faulthandler.enable()
# faulthandler.dump_traceback_later(900, repeat=False, file=sys.stderr)
# print("faulthandler enabled", flush=True)

SEED = 3407
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu" 
seed_everything(SEED, workers=True)
torch.set_float32_matmul_precision("high")

logs_root = "/mlspeech/data/tomer/streaming_whisper/models/logs"
ckpt_root = "/mlspeech/data/tomer/streaming_whisper/models/ckpts"

whisper_lrs: dict[str, float] = {'tiny': 1.5e-3, 'base': 1e-3, 'small': 5e-4, 'medium': 2.5e-4, 'large': 1.75e-4, 'large-v2': 2e-4}

project_names = {
    "lora": "LoRA_whisper_stream",
}

def train_model(log_output_dir, check_output_dir, model_name, train_set, val_set, train_name, project_name, cfg: Config) -> None:

    Path(log_output_dir).mkdir(exist_ok=True)
    Path(check_output_dir).mkdir(exist_ok=True)

    wandblogger = WandbLogger(
        save_dir=log_output_dir,
        name=train_name,
        project=project_names[project_name]
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=cfg.top_k, # Best model save,
        monitor="val/wer",
        every_n_epochs=1
    )

    steps_ckpt_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/steps_checkpoint",
        filename="steps-checkpoint-{step:09d}",
        save_top_k=-1, # Save all
        every_n_train_steps=500
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]
    
    if cfg.early_stop:
        early_stop_callback = EarlyStopping(
            monitor="val/wer",
            min_delta=0.00,
            patience=2,
            mode="min",
        )
        callback_list.append(early_stop_callback)

    # Model mux
    if cfg.lora and cfg.streaming_train:
        model = LoRAStreamedWhisper(cfg, model_name, cfg.lang, train_set, val_set, rank=cfg.rank, enc_emb_gran=cfg.gran, enc_context=cfg.extra_gran_blocks, sim_stream=cfg.sim_stream)
    
    trainer = Trainer(
        accelerator=DEVICE,
        max_epochs=cfg.num_train_epochs,
        callbacks=callback_list,
        logger=wandblogger if not cfg.no_logger else False,
        deterministic=True,
        num_sanity_val_steps=1,
        strategy=cfg.strategy,
        fast_dev_run=cfg.fast_dev_run,
        # devices=8
        # precision="16"
        # accumulate_grad_batches=cfg.gradient_accumulation_steps,
    )

    if cfg.ckpt is None: trainer.fit(model)
    else: trainer.fit(model, ckpt_path=cfg.ckpt)


if __name__ == "__main__":

    project_name = None
    
    args = parse_cmdl()
    
    # Training config
    cfg = Config(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_epsilon=args.adam_epsilon,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        num_worker=args.num_worker,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gacc,
        no_logger=args.no_logger,
        dataset=args.dataset,
        name=args.name,
        top_k=args.top_k,
        sample_rate=16_000,
        ckpt=args.ckpt,
        size=args.size,
        lora=args.lora,
        lora_ckpt=args.lora_ckpt,
        rank=args.rank,
        gran=args.gran,
        extra_gran_blocks=args.extra_gran_blocks,
        sim_stream=args.simulate_stream,
        fast_dev_run=args.fast_dev_run,
        early_stop=args.early_stop,
        strategy=args.strategy,
        streaming_train=args.streaming_train,
        streaming_random=args.streaming_random,
        streaming_fraction=args.streaming_fraction,
        seed=SEED,
        multilingual=args.multilingual,
        custom_len=args.custom_len,
        use_from_ft_ckpt=args.use_from_ft_ckpt,
        lmdb=args.lmdb,
        self_supervision=args.self_supervision,
        slices_num=args.num_slices,
        random_masking=args.random_masking
    )

    if args.lmdb:
        # Set lmdb paths for datasets
        cfg.lmdb_paths = {key: ds_paths[key].get('train-lmdb', None) for key in args.dataset}
        print(f"Using LMDB paths: {cfg.lmdb_paths}")

    if cfg.streaming_train:
        assert cfg.sim_stream == cfg.streaming_train, "When running in full stream mode you must simulate streaming!"
        cfg.sim_stream = True

    lr_addition = f"_LR-{cfg.learning_rate}"
    effective_bsize = cfg.batch_size * cfg.gradient_accumulation_steps

    if cfg.random_masking:
        cfg.name += f"_random-masking{cfg.slices_num}"

    if cfg.lora and cfg.streaming_train:
        dir_name = f"LoRA_streamed_whisper_{cfg.size}_{'-'.join(cfg.dataset)}_{effective_bsize}_{cfg.name}{lr_addition}_r{cfg.rank}_g{cfg.gran}_eg{cfg.extra_gran_blocks}_top{cfg.top_k}_full-stream{cfg.streaming_train}_random-order{cfg.streaming_random}_fraction{cfg.streaming_fraction}"
        project_name = "lora"
    
    # Run trainer
    train_model(
        log_output_dir=os.path.join(logs_root, dir_name),
        check_output_dir=os.path.join(ckpt_root, dir_name),
        model_name=args.size,
        train_set=[ds_paths[key]['train'] for key in args.dataset],
        val_set=[ds_paths[key]['val'] for key in args.dataset],
        train_name=dir_name,
        project_name=project_name,
        cfg=cfg
    )

    
