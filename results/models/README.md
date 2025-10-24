# Loading saved models

The final trained models are available at `s3://data-mixture-optimization-models/models/`. Each model is in a folder, consisting of `config.yaml` and `model.pt`

Please git clone the Olmo codebase and install required dependencies there https://github.com/allenai/OLMo. In particular, see `requirements.txt`

Place both `load_model_example.py` and `load_model.py` under the Olmo top folder. Then use the first script to load the model.
For tokenizer, please use `/data_mixing_experiments/olmo/GPT2TokenizerFast-gpt2`
