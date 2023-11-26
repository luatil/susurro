REM https://huggingface.co/luatil/whisper-model-pt
if not exist "models/whisper-model-pt" (
    ct2-transformers-converter --model luatil/whisper-model-pt --output_dir models/whisper-model-pt --copy_files tokenizer_config.json preprocessor_config.json generation_config.json
)


REM https://huggingface.co/brunoqgalvao/whisper-small-pt-br
if not exist "models/whisper-small-pt-br" (
    ct2-transformers-converter --model brunoqgalvao/whisper-small-pt-br --output_dir models/whisper-small-pt-br --copy_files vocab.json tokenizer_config.json preprocessor_config.json generation_config.json config.json added_tokens.json normalizer.json special_tokens_map.json
)

REM jonatasgrosman/whisper-small-pt-cv11-v7
if not exist "models/whisper-small-pt-cv11-v7" (
    ct2-transformers-converter --model jonatasgrosman/whisper-small-pt-cv11-v7 --output_dir models/whisper-small-pt-cv11-v7 
)