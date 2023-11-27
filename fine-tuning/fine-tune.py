import os
import torch
from datasets import load_dataset, DatasetDict
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Audio
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from transformers import WhisperProcessor
import evaluate
import argparse

# login(token=token, add_to_git_credential=True, write_permission=True)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class WhisperFineTuning:
    def __init__(
        self,
        model_name="openai/whisper-small",
        language="Portuguese",
        dataset_name="mozilla-foundation/common_voice_11_0",
    ) -> None:
        self.language = language
        self.model_name = model_name
        self.dataset_name = dataset_name
        # NOTE: Only reliable amount for preparing dataset. Depending on the configuration, you may increase this number.
        self.cpu_count = 1

        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.tokenizer = WhisperTokenizer.from_pretrained(
            model_name, language=language, task="transcribe"
        )
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []

        self.metric = evaluate.load("wer")

        self.common_voice = None

    def load_dataset(self, load_path="./common_voice") -> None:
        self.common_voice = DatasetDict.load_from_disk(load_path)

    def prepare_batch(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]

        # encode target text to label ids
        batch["labels"] = self.tokenizer(batch["sentence"]).input_ids
        return batch

    def prepare_dataset(self, save=True, save_path="./common_voice"):
        self.common_voice = DatasetDict()
        self.common_voice["train"] = load_dataset(
            self.dataset_name,
            "pt",
            split="train+validation",
            use_auth_token=True,
        )
        self.common_voice["test"] = load_dataset(
            self.dataset_name, "pt", split="test", use_auth_token=True
        )

        self.common_voice = self.common_voice.remove_columns(
            [
                "accent",
                "age",
                "client_id",
                "down_votes",
                "gender",
                "locale",
                "path",
                "segment",
                "up_votes",
            ]
        )

        self.common_voice = self.common_voice.cast_column(
            "audio", Audio(sampling_rate=16000)
        )

        self.common_voice = self.common_voice.map(
            self.prepare_batch,
            remove_columns=self.common_voice.column_names["train"],
            num_proc=self.cpu_count,
        )

        if save:
            self.common_voice.save_to_disk(save_path)

    def train(
        self,
        repo_name="./whisper-small-pt",
        batch_size=32,
        lr=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=True,
    ):
        if self.common_voice is None:
            raise ValueError("Dataset not loaded. Use load_dataset() first.")

        processor = WhisperProcessor.from_pretrained(
            self.model_name, language=self.language, task="transcribe"
        )
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        training_args = Seq2SeqTrainingArguments(
            output_dir=repo_name,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
            learning_rate=lr,
            warmup_steps=warmup_steps,
            max_steps=max_steps,
            gradient_checkpointing=gradient_checkpointing,
            fp16=fp16,
            evaluation_strategy=evaluation_strategy,
            per_device_eval_batch_size=per_device_eval_batch_size,
            predict_with_generate=predict_with_generate,
            generation_max_length=generation_max_length,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            report_to=["tensorboard"],
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            push_to_hub=push_to_hub,
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=self.model,
            train_dataset=self.common_voice["train"],
            eval_dataset=self.common_voice["test"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=processor.feature_extractor,
        )

        trainer.train()

    def compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}


def main():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Fine-tuning Whisper for Speech to Text"
    )

    # Add arguments for WhisperFineTuning initialization
    parser.add_argument(
        "--model_name", type=str, default="openai/whisper-small", help="Model name"
    )
    parser.add_argument(
        "--language", type=str, default="Portuguese", help="Language for transcription"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="mozilla-foundation/common_voice_11_0",
        help="Dataset name",
    )

    # Add arguments for prepare_dataset
    parser.add_argument(
        "--prepare_dataset", action="store_true", help="Prepare the dataset"
    )
    parser.add_argument(
        "--save_dataset", action="store_true", help="Save the prepared dataset"
    )
    parser.add_argument(
        "--dataset_save_path",
        type=str,
        default="./common_voice",
        help="Path to save the prepared dataset",
    )

    # Add arguments for train
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--repo_name",
        type=str,
        default="./whisper-small-pt",
        help="Repository name for training",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--warmup_steps", type=int, default=500, help="Warmup steps for training"
    )
    parser.add_argument(
        "--max_steps", type=int, default=4000, help="Max steps for training"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Use gradient checkpointing",
        default=True,
    )
    parser.add_argument("--fp16", action="store_true", help="Use fp16")
    parser.add_argument(
        "--evaluation_strategy",
        type=str,
        default="steps",
        help="Evaluation strategy for training",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--predict_with_generate",
        action="store_true",
        help="Predict with generate",
        default=True,
    )
    parser.add_argument(
        "--generation_max_length",
        type=int,
        default=225,
        help="Max length for generation",
    )
    parser.add_argument(
        "--save_steps", type=int, default=1000, help="Save steps for training"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=1000, help="Eval steps for training"
    )
    parser.add_argument(
        "--logging_steps", type=int, default=25, help="Logging steps for training"
    )
    parser.add_argument(
        "--load_best_model_at_end",
        action="store_true",
        help="Load best model at end of training",
        default=True,
    )
    parser.add_argument(
        "--greater_is_better",
        action="store_true",
        default=False,
        help="Greater is better for metric",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=True,
        help="Push to hub after training",
    )

    # ... add other training arguments here ...

    # Parse the arguments
    args = parser.parse_args()

    # Initialize WhisperFineTuning
    whisper = WhisperFineTuning(
        model_name=args.model_name,
        language=args.language,
        dataset_name=args.dataset_name,
    )

    # Prepare dataset
    if args.prepare_dataset:
        whisper.prepare_dataset(
            save=args.save_dataset, save_path=args.dataset_save_path
        )

    # Train
    if args.train:
        whisper.load_dataset(load_path=args.dataset_save_path)
        whisper.train(repo_name=args.repo_name, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
