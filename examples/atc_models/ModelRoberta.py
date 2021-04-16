import os
import argparse
import pickle
import pandas as pd
import numpy as np
from random import sample
import json

import sys
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

class AutoFillBlank:

    def __init__(self):
        # from fairseq.data.data_utils import collate_tokens
        # from fairseq.models.roberta import RobertaModel
        self.model = self.load_roberta_model()

    def predict(self, json_dict):
        choce_dict = json_dict["choice_dict"]
        sent = []
        label = []
        for k, v in choce_dict.items():
            label.append(k)
            sent.append(v)
        texts = sent
        batch = collate_tokens([self.model.encode(text) for text in texts], pad_idx=1)
        # OutputProbs = list(self.model.demo_text_list(sent))
        qq_probs = self.model.predict('english_head', batch)
        OutputProbs = qq_probs[:,1].tolist()

        res = label[OutputProbs.index(max(OutputProbs))]

        OutputProbs = self.SoftMax(OutputProbs)
        OutputProbs = self.SoftMax(OutputProbs)
        OutputProbs = self.SoftMax(OutputProbs)

        prob_dict = dict(zip(label, OutputProbs))
        max_prob = max(OutputProbs)
        if max_prob >= 0:
            json_dict["ans"] = res
            json_dict["prob_dict"] = prob_dict
            json_dict["state"] = 0
            return 0, json_dict
        else:
            json_dict["ans"] = res
            json_dict["prob_dict"] = prob_dict
            json_dict["state"] = -7
            return -7, json_dict


    def load_roberta_model(self,):
        print("start load roberta model")
        model_path = ""
        roberta = RobertaModel.from_pretrained(
            os.path.join(path, 'checkpoints'),
            checkpoint_file='checkpoint_best.pt',
            data_name_or_path= os.path.join(path, 'english-bin'),
        )
        roberta.eval()  # disable dropout
        print("load roberta model succ!")
        return roberta
    def SoftMax(self, x, g=0, t=0.1):
        x = np.array(x)
        log_x = (x + g) / t
        exp_x = np.exp(log_x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x