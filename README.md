
## train example files
aied_essay/examples/bert_demo.py
aied_essay/examples/xlnet_demo.py
aied_essay/examples/electra_demo.py
aied_essay/examples/roberta_demo.py
aied_essay/examples/bart_demo.py


## predict example files
aied_essay/examples/aied_predict.py


### environment configuration


1. virtual environment installation
```sh
conda create --name=aied python=3.7.5
source activate aied
```

2. dependence package installation

first step:
`conda install tensorflow-gpu==1.13.1  cudatoolkit=10.0.130=0`

second step:

`pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html`
third step:

`pip install -r requirements_gpu.txt`


### source of model

en_bert_large_cased   
[reference](https://arxiv.org/abs/1810.04805)  
[model](https://huggingface.co/bert-large-cased)  

en_roberta_large  
[reference](https://arxiv.org/abs/1907.11692)  
[model](https://huggingface.co/roberta-large)  

en_electra_large  
[reference](https://arxiv.org/abs/2003.10555)   
[model](https://huggingface.co/google/electra-large-discriminator)   

en_xlnet_large_cased   
[reference](https://arxiv.org/abs/1906.08237)   
[model](https://huggingface.co/xlnet-large-cased)  

en_bart_large   
[reference](https://arxiv.org/abs/1910.13461)  
[model](https://huggingface.co/facebook/bart-large)  
