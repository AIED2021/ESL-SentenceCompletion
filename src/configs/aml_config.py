
from src.models import *
from src.configs import *

en_model_dict = {
        "en_bert_large_cased": {"model_class": BERT, "config": en_bert_large_cased_config},
        "en_roberta_large": {"model_class": ROBERTA, "config": en_roberta_large_config},
        "en_xlnet_large_cased": {"model_class": XLNet, "config": en_xlnet_large_cased_config},
        "en_electra_large": {"model_class": ELECTRA, "config": en_google_electra_large_config},
        "en_bart_large":{"model_class":BART,"config":en_bart_large_config}
        }


model_dict.update(en_model_dict)

default_model_list = list(model_dict.keys())
