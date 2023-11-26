import os
from datasets import load_dataset, DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from datasets import Audio 

# Execute nvidia-smi command to check if GPU is connected
gpu_info = os.system('nvidia-smi')

# login(token=token, add_to_git_credential=True, write_permission=True)

common_voice = DatasetDict()

common_voice["train"] = load_dataset("mozilla-foundation/common_voice_11_0", "pt", split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset("mozilla-foundation/common_voice_11_0", "pt", split="test", use_auth_token=True)

common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny", language="Portuguese", task="transcribe")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="Portuguese", task="transcribe")

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=1)

common_voice.save_to_disk("./common_voice")