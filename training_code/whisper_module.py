import sys
sys.path.append("./")
import torch
import random
import evaluate
import careless_whisper_stream
import careless_whisper_stream.tokenizer as whisper_tokenizer

from torch import nn, Tensor
from torch.optim.adamw import AdamW
from training_code.utils import Config
from careless_whisper_stream import StreamingWhisper
from careless_whisper_stream.audio import HOP_LENGTH
from careless_whisper_stream.normalizers import EnglishTextNormalizer
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from training_code.collators import WhisperDataCollatorWithPadding, LoRAWhisperDataCollatorWithPadding
from training_code.datasets_classes import TIMIT, WAVsDataset, AlignedTextGridDataset, AlignedTextGridDatasetLMDB

class WhisperCustomModel(LightningModule):
    def __init__(self, cfg:Config, model_name="tiny", lang="en", train_dataset: str = None, eval_dataset: str = None, task="transcribe") -> None:
            super().__init__()
            self.save_hyperparameters()
            self.task = task
            self.lang = lang
            self.model = careless_whisper_stream.load_model(model_name)
            self.tokenizer: whisper_tokenizer = whisper_tokenizer.get_tokenizer(True, language=lang)

            self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            self.metrics_wer = evaluate.load("wer")
            self.metrics_cer = evaluate.load("cer")

            self.params = self.model

            self.cfg = cfg
            self.__train_dataset = train_dataset
            self.__eval_dataset = eval_dataset

    def forward(self, x):
        return self.model(x)

    def calc_wer_val(self, out: Tensor, labels: Tensor):
        out[out == -100] = self.tokenizer.eot
        labels[labels == -100] = self.tokenizer.eot

        o_list, l_list = [], []
        for o, l in zip(out, labels):
            o = torch.argmax(o, dim=1)
            o_list.append(self.normalizer(self.tokenizer.decode(o)))
            l_list.append(self.normalizer(self.tokenizer.decode(l)))
            
        wer = self.metrics_wer.compute(references=l_list, predictions=o_list)
        return wer

    def training_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        with torch.no_grad():
            audio_features = self.model.encoder(input_ids)

        out, _, _ = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_id):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        audio_features = self.model.encoder(input_ids)
        out, _, _ = self.model.decoder(dec_input_ids, audio_features)

        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))
        wer = self.calc_wer_val(out, labels)

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True, on_epoch=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True, on_epoch=True)

        return {
            "wer": wer,
            "loss": loss
        }

    def configure_optimizers(self):
        model = self.params
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters()
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": self.cfg.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters()
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                        lr=self.cfg.learning_rate,
                        eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = LinearLR(
            self.optimizer, start_factor=0.5, end_factor=0.8, total_iters=self.t_total
        )

        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.t_total = (
                (len(self.__train_dataset) // (self.cfg.batch_size))
                // self.cfg.gradient_accumulation_steps
                * float(self.cfg.num_train_epochs)
            )

    def get_dataset(self, ds_path: str, split: str):
        return TIMIT(ds_path, self.tokenizer)
        
    def train_dataloader(self):
        dataset = self.get_dataset(self.__train_dataset, "train")
        return torch.utils.data.DataLoader(dataset,
                        batch_size=self.cfg.batch_size,
                        drop_last=True, shuffle=True, num_workers=self.cfg.num_worker,
                        collate_fn=WhisperDataCollatorWithPadding()
                        )

    def val_dataloader(self):
        dataset = self.get_dataset(self.__eval_dataset, "val")
        return torch.utils.data.DataLoader(dataset,
                        batch_size=self.cfg.batch_size,
                        num_workers=self.cfg.num_worker,
                        collate_fn=WhisperDataCollatorWithPadding()
                        )
    

class LoRAStreamedWhisper(WhisperCustomModel):
    def __init__(self, cfg: Config, model_name="tiny", lang="en", train_dataset: str = None, eval_dataset: str = None, task="transcribe", rank=8, enc_emb_gran=15, enc_context=1, sim_stream=False, beam_size=None, use_kv_cache=False, use_ca_kv_cache=False, get_times=False, eval_script=False) -> None:
        super().__init__(cfg, model_name, lang, train_dataset, eval_dataset, task)

        self.automatic_optimization = not cfg.streaming_train

        # if model_name != "large-v2" and not eval_script:
        print(f"enc_emb_gran: {enc_emb_gran}")
        if not cfg.use_from_ft_ckpt:
            self.model: StreamingWhisper = careless_whisper_stream.load_streaming_model_for_train(model_name, 
                                                                                    advisor_ckpt_path=None,
                                                                                    advisor_type=None,
                                                                                    rank=rank,
                                                                                    gran=enc_emb_gran,
                                                                                    extra_gran_blocks=enc_context,
                                                                                    )
        else:
            self.model: StreamingWhisper = careless_whisper_stream.load_streaming_model(cfg.size, cfg.gran * 20)
        
        for n, p in self.model.named_parameters():
            if "lora" not in n:
                p.requires_grad = False

        self.normalizer = EnglishTextNormalizer()
        self.model.encoder._use_mask(True)

        self.model_name = model_name
        self.rank = self.model.rank
        self.enc_emb_gran = self.model.gran
        self.enc_context = self.model.extra_gran_blocks
        self.simulate_stream = sim_stream
        self.full_stream = cfg.streaming_train
        self.beam_size = beam_size
        self.language = lang
        self.use_kv_cache = use_kv_cache
        self.use_ca_kv_cache = use_ca_kv_cache
        self.get_times = get_times
        self.eval_script = eval_script
        self.lmdb_paths = cfg.lmdb_paths

        # for stream mode train
        self.num_frames = self.model.dims.n_audio_ctx // self.enc_emb_gran # 1500 // enc_emb_gran
        self.mel_samples = self.enc_emb_gran * 2 * HOP_LENGTH

        self.params = None

        self.last_out = None
        self.__train_dataset = train_dataset
        self.__eval_dataset = eval_dataset

    def _calc_labels(self, labels: Tensor, endpoints: Tensor, index: int, out: Tensor = None):
        if self.cfg.self_supervision and out is not None:
            # get predicted tokens
            pred_tokens = torch.argmax(out, dim=-1)

            # find first eot in predictions
            eot_mask = (pred_tokens == self.tokenizer.eot)
            eot_indices = eot_mask.int().argmax(dim=-1)

            # create new labels
            clone_labels = labels.clone()

            for i in range(labels.shape[0]):
                eot_idx = eot_indices[i].item()
                if eot_idx < labels.shape[1]:
                    clone_labels[i, :eot_idx + 1] = pred_tokens[i, :eot_idx + 1]
                    clone_labels[i, eot_idx] = self.tokenizer.eot
                    clone_labels[i, eot_idx + 1:] = -100
                else:
                    # no eot found, use all predictions
                    clone_labels[i, :] = pred_tokens[i, :]
            
            return clone_labels
            
        else:
            if not self.cfg.random_masking:
                t_seconds = (index + 1) * self.enc_emb_gran * 0.02
            else:
                t_seconds = index * 0.02
            
            # take only relevant labels into account
            mask = (endpoints <= t_seconds) & (endpoints != -100)
            eot_indices = mask.int().argmin(dim=-1)
            
            clone_labels = labels.clone()
            
            # ignore irrelevant labels
            clone_labels[~mask] = -100
            
            # mark eot labels
            clone_labels[range(labels.shape[0]), eot_indices] = self.tokenizer.eot 
            return clone_labels

    def _get_sample_points(self, endpoints: Tensor):
        # base case
        if self.cfg.streaming_fraction == 1:
            return range(self.enc_context, self.num_frames)

        biggest_endpoint = endpoints.max().item()

        # determine last index.
        num_frames = min(int(((biggest_endpoint / 0.02) // self.enc_emb_gran) + int(1 / (self.enc_emb_gran * 0.02)) + 1), self.num_frames) # adding 1 sec of silence
        new_range = range(self.enc_context, num_frames)

        sample_points = random.sample(new_range, k=int(len(new_range) * self.cfg.streaming_fraction) + 1)

        if self.cfg.streaming_random:
            return sample_points

        return sorted(sample_points)

    def _get_sample_points_random_mask(self, endpoints: Tensor):
        biggest_endpoint = endpoints.max().item()
        ls_choices = list(range(5, 55, 5))
        ls_weights = [15, 20, 25, 20, 15, 1, 1, 1, 1, 1]
        lengths = [30]
        curr_sum = 30
        last_index = biggest_endpoint // 0.02

        while curr_sum < last_index:
            l = random.choices(ls_choices, weights=ls_weights, k=1)[0]
            
            if curr_sum + l > last_index:
                l = (int(last_index - curr_sum) // 5) * 5
                lengths.append(l)
                break

            lengths.append(l)
            curr_sum += l
        
        sample_points = [sum(lengths[:i]) for i in range(1, len(lengths) + 1)]
        mask = torch.full((sample_points[-1], sample_points[-1]), float("-inf"))
        start = 0
        for l in lengths:
            end = start + l
            mask[start:end, :end] = 0
            start = end
        

        # Now sample self.cfg.slices_num points from sample_points, if there are less, return all.
        if len(sample_points) <= self.cfg.slices_num:
            sample_points = sorted(sample_points)
        else:
            sample_points = sorted(random.sample(sample_points, k=self.cfg.slices_num))

        return sample_points, mask

    def _forward_step_stream(self, batch, step):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()
        endpoints = batch["endpoints"]

        if step == "train":
            optimizer = self.optimizers()

        # forward
        if self.cfg.random_masking:
            sample_points, mask_value = self._get_sample_points_random_mask(endpoints)
            mask_value = mask_value.to(input_ids.device)
        else:
            sample_points = self._get_sample_points(endpoints)

        for i in sample_points:
            if self.cfg.random_masking:
                audio_features = self.model.encoder(input_ids[..., :i * 2], index=[0, i], mask=mask_value)
            else:
                audio_features = self.model.encoder(input_ids[..., :(i + 1) * (self.enc_emb_gran * 2)], index=[0, (i + 1) * self.enc_emb_gran], mask=True)
            out = self.model.decoder(dec_input_ids, audio_features, dump_type="None")

            if step == "train":
                optimizer.zero_grad()

            # loss calc
            frame_labels = self._calc_labels(labels, endpoints, i, out if self.cfg.self_supervision else None)
            loss = self.loss_fn(out.view(-1, out.size(-1)), frame_labels.view(-1))

            # optimizer step if relevant.
            if step == "train":
                loss.backward()
                optimizer.step() # might move optimizer step to out of the loop for faster training

        if step == "train":
            return loss
        
        return out, loss

    def _forward_step(self, batch, step):
        input_ids = batch["input_ids"]
        labels = batch["labels"].long()
        dec_input_ids = batch["dec_input_ids"].long()

        # self.model.encoder.reset()
        audio_features = self.model.encoder(input_ids, mask=True) # use mask for fast learning, simulates stream mode
        out = self.model.decoder(dec_input_ids, audio_features, dump_type="None")

        # loss calc
        loss = self.loss_fn(out.view(-1, out.size(-1)), labels.view(-1))

        if step == "train":
            return loss
        
        return out, loss

    def training_step(self, batch, batch_id):
        if self.full_stream:
            loss = self._forward_step_stream(batch, "train")
        else:
            loss = self._forward_step(batch, "train")

        self.log("train/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_id):
        out, loss = self._forward_step(batch, "val")
        wer = self.calc_wer_val(out, batch["labels"])

        self.log("val/loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val/wer", wer, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        return {
            "wer": wer,
            "loss": loss
        }

    def predict_step(self, batch, batch_id):
        wavs = batch["wav_path"]
        text = batch["text"]

        results, times = self.model.transcribe(wav_file=wavs[0],
                                            simulate_stream=True,
                                            beam_size=self.beam_size,
                                            language=self.language,
                                            use_ca_kv_cache=self.use_ca_kv_cache,
                                            use_sa_kv_cache=self.use_kv_cache,
                                            get_times=self.get_times)

        return [res.text for res in results], text, wavs[0], times

    def on_train_epoch_start(self):
        random.seed(self.current_epoch + self.cfg.seed)

    def on_train_epoch_end(self):        
        model_sched = self.lr_schedulers()
        model_sched.step(self.trainer.callback_metrics["val/loss"])

    def get_dataset(self, ds_path, split):
        print(f"Stream simulation mode: {self.simulate_stream}")
        
        if self.full_stream and not self.cfg.self_supervision:
            return AlignedTextGridDatasetLMDB(ds_path=ds_path, lmdb_paths=self.lmdb_paths, get_streamed_mel=True, gran=self.enc_emb_gran, extra_gran_blocks=self.enc_context, n_mels=self.model.dims.n_mels, multilingual=self.cfg.multilingual)
        
        return WAVsDataset(ds_path=ds_path, get_streamed_mel=self.simulate_stream)
    
    def configure_optimizers(self):
        model = self.model
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad],
                "weight_decay": self.cfg.weight_decay,
                "lr": self.cfg.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if not p.requires_grad],
                "weight_decay": 0.0,
                "lr": 0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                        eps=self.cfg.adam_epsilon)
        self.optimizer = optimizer

        scheduler = ReduceLROnPlateau(
            self.optimizer, 'min', patience=2, factor=0.5
        )

        self.scheduler = scheduler

        return [optimizer], [{"scheduler": scheduler, "monitor": "val/loss"}]
    
    def train_dataloader(self):
        dataset = self.get_dataset(self.__train_dataset, "train")
        return torch.utils.data.DataLoader(dataset,
                        batch_size=self.cfg.batch_size,
                        drop_last=True, shuffle=True, num_workers=self.cfg.num_worker, pin_memory=True,
                        collate_fn=LoRAWhisperDataCollatorWithPadding())

    def val_dataloader(self):
        dataset = self.get_dataset(self.__eval_dataset, "val")
        return torch.utils.data.DataLoader(dataset,
                        batch_size=self.cfg.batch_size,
                        num_workers=self.cfg.num_worker, pin_memory=True,
                        collate_fn=LoRAWhisperDataCollatorWithPadding())

    def on_save_checkpoint(self, checkpoint):
        checkpoint["dims"] = self.model.dims.__dict__
