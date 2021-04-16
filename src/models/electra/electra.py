from src.models.hf_base import HFBase
from transformers import BertForSequenceClassification, BertModel, BertTokenizer,ElectraTokenizer,AutoModelForSequenceClassification
from transformers import AdamW

class ELECTRA(HFBase):
    def __init__(self,config):
        super().__init__(config)
        self.model_name = 'electra'

    def get_tokenizer(self):
        tokenizer = ElectraTokenizer.from_pretrained(self.model_dir)
        return tokenizer