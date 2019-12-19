# Spelling Corrector Experimental models
This is experimental project for research of spelling corrector models
We researched different configurations of systems for spelling corrections:

Language model:
- TFKuznetsovELMO (Best by quality but slow)
- PyTorchELMO (Good for speed and good for quality)
- TransformerELMO (Nice for speed but bad quality)
- RusVecores ELMO (Nice for speed but bad quality)
- TFHubELMO (Not good solution)

Error model and candidates generator:
- Levenshtein
- Weighted Levenshtein

Reranker for corrections making:
- Summator (decision selected for hypotheses with maximal sum of language_model score and error_score)
- Logistic Regression (decision is made by trainable Regression algorithm which uses multiple features includin language model score and error score and many other)

##Root documentation
http://192.168.10.188:8081/index.php/Spelling_Corrector

## Table with comparison of models on different datasets
https://docs.google.com/spreadsheets/d/1QzYF2O5z1nQR8gic8uFTx4TnB242EV3uNXPjl-xTppg/edit#gid=0

# Basic usage
1. How to prepare spelling corrector object with specific implementation underneath

`git clone https://github.com/acriptis/spelling_corrector_experiments`

`pip install -r requirements.txt`

then you can launch tests:

`python tests/test_spelling_correctors.py`

but it requires about 30GB of RAM and loads disck space (model ELMO40inKuz consu,es about 4.5GB of disk space,
TorchELMO40in model consumes about 3GB of disk space)
(But it loads all models, )

# As Server usage

TODO How to run spelling corrector as server
- deepepavlov riseapi?
- django rest api service
- flask

# How to train your own components
TODO
## Language model
TODO 
## ReRanker
TODO
## Spelling Corrector Candidates Generator
TODO describe how to configure

