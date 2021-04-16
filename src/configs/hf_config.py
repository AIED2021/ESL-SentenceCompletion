import os
base_path = os.path.dirname(os.path.realpath(__file__))
from os.path import join


## en_bert_large_base_cased
en_bert_large_cased_config = {"model_dir": join(base_path, '../data/en_bert_large_cased'),
                             "save_dir": 'model/en_bert_large_cased'}

## en_roberta_large
en_roberta_large_config = {"model_dir": join(base_path, '../data/en_roberta_large'),
                             "save_dir": 'model/en_roberta_large'}

# en_xlnet_large_cased
en_xlnet_large_cased_config = {"model_dir": join(base_path, '../data/en_xlnet_large_cased'),
                             "save_dir": 'model/en_xlnet_large_cased'}

## google_electra_large
en_google_electra_large_config = {"model_dir": join(base_path, '../data/en_google_electra_large'), "save_dir": 'model/en_google_electra_large'}

## en_bart_large
en_bart_large_config = {"model_dir": join(base_path, '../data/en_bart_large'),
                             "save_dir": 'model/en_bart_large',"token_type_ids_disable":True}