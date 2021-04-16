import os
import jieba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from pytorchtools import EarlyStopping
from sklearn.metrics import *

from src.models.base_model import BaseModel
from src.utils.metrics_utils import get_model_metrics
from src.utils.data_utils import load_df

class TorchDataSet(Dataset):
    def __init__(self, l_label_id):
        self.l_label_id = l_label_id
        
    def __len__(self):
        return len(self.l_label_id)

    def __getitem__(self, idx):
        n_label, ln_id = self.l_label_id[idx]
        return n_label, np.array(ln_id)


class TorchBase(BaseModel):
    def __init__(self, O_CONFIG):
        super().__init__(O_CONFIG)
        self._net = None
        self._optimizer = None

        self._run_logger = O_CONFIG["run_logger"]
        self._b_print = O_CONFIG["b_print"]

        self._b_use_cuda = O_CONFIG["use_cuda"]

        self._loss_fun = nn.CrossEntropyLoss() 

        

        self._s_model_name = ""
        
        self._ls_stop_token = O_CONFIG["stop_token"]

        self._s_model_with_weight_path = ""
        self._s_best_model_with_weight_path = ""
        self._s_weight_file = ""
        self.model_path = ""

        self._embed_mgr = None
        self._n_vocab_num = -1
        self._n_embed_dim = -1
        self._np_embed_matrix = None
        self._np_pad_vec = None
        self._load_word_vec(O_CONFIG)

    
    def _load_weight(self, s_weigth_path):
        if os.path.exists(s_weigth_path):
            self._net.load_state_dict(torch.load(s_weigth_path))
        else:
            s_msg = "weight file %s not exist" % s_weigth_path
            self._run_logger and self._run_logger.info(s_msg)
            self._b_print and print(s_msg)

     # Load word vector
    def _load_word_vec(self, O_CONFIG):
        o_embed_config = O_CONFIG["embed_config"]
        s_embed_path = o_embed_config["emb_path"]

        self._n_embed_dim = o_embed_config["emb_dim"]      
        self._embed_mgr = o_embed_config["emb_class"](s_embed_path, self._n_embed_dim)
        self._n_vocab_num = len(self._embed_mgr.get_vocabs())  
        self._np_embed_matrix = self._embed_mgr.get_emb_matrix()
        self._m_word_id = self._embed_mgr.get_tokenizer().word_index
        self._n_pad_id = self._n_vocab_num - 1

        O_CONFIG["embed_dim"] = self._n_embed_dim
        O_CONFIG["embed_pretrained"] = torch.Tensor(self._np_embed_matrix)

        s_msg = "load word vector success. vocab:%d embed:%d" % (self._n_vocab_num, self._n_embed_dim) 
        self._run_logger and self._run_logger.info(s_msg)
        self._b_print and print(s_msg)

    def get_torch_data(self, train_path, dev_path, test_path):
        # train text -> token -> id vector
        train_data_loader, df_train = self._get_torch_data_loader(train_path)
        O_IN_TRAIN = {
            "b_train": True, 
            "data_loader": train_data_loader, 
            "n_epoch": 0,
            "df": df_train
        }
        s_msg = "Get trian data"
        self._run_logger and self._run_logger.info(s_msg)
        self._b_print and print(s_msg)

        # test text -> token -> id vector
        test_data_loader, df_test = self._get_torch_data_loader(test_path)
        O_IN_TEST = { 
            "b_train": False, 
            "data_loader": test_data_loader, 
            "n_epoch": 0,
            "df": df_test
        }
        s_msg = "Get test data"
        self._run_logger and self._run_logger.info(s_msg)
        self._b_print and print(s_msg)

        
        valiad_data_loader, df_valiad = self._get_torch_data_loader(dev_path)
        O_IN_VALIAD = {
            "b_train": False, 
            "data_loader": valiad_data_loader, 
            "n_epoch": 0,
            "df": df_valiad
        }
        s_msg = "Get valiad data"
        self._run_logger and self._run_logger.info(s_msg)
        self._b_print and print(s_msg)

        return O_IN_TRAIN, O_IN_TEST, O_IN_VALIAD

    def train(self, train_path, dev_path, test_path): 
        O_IN_TRAIN, O_IN_TEST, O_IN_VALIAD = self.get_torch_data(train_path, dev_path, test_path)

        # class imbalance
        df_train = O_IN_TRAIN["df"]
        ln_train_label = list(df_train["label"])
        n_train_neg_num = ln_train_label.count(0)
        n_train_pos_num = ln_train_label.count(1)
        

        f_val = -1.0
        if n_train_neg_num/n_train_pos_num < 0.5 or n_train_neg_num/n_train_pos_num > 2.0:
            f_val = n_train_neg_num/n_train_pos_num
           
        if f_val > 0.0:
            t_weight = torch.Tensor([1.0, f_val])
            if self._b_use_cuda:
                t_weight = t_weight.cuda()
            self._loss_fun = nn.CrossEntropyLoss()
     
            s_msg = "train data: %s %s, sample weight %s" % (str(n_train_pos_num), str(n_train_neg_num), str(t_weight))
            self._run_logger and self._run_logger.info(s_msg)
            self._b_print and print(s_msg)

        if self._b_use_cuda:
            self._net = self._net.cuda()

            s_msg = "net use cuda"
            self._run_logger and self._run_logger.info(s_msg)
            self._b_print and print(s_msg)

        s_msg = str(self._net)
        self._run_logger and self._run_logger.info(s_msg)
        self._b_print and print(s_msg)

        f_valiad_avg_batch_best_acc = 0.0
        n_patience_count = 0 

        for n_epoch in range(self.epochs):
            # train batch
            O_IN_TRAIN["n_epoch"] = n_epoch

            O_OUT_TRAIN = self._batch_train_test(O_IN_TRAIN)
            O_OUT_TRAIN["s_info"] = "Train"
            O_OUT_TRAIN["b_out_excel"] = False
            O_OUT_TRAIN["s_excel_in_path"] = train_path
            O_OUT_TRAIN["s_excel_out_path"] = ""
            
            np_y_true_train, np_y_pred_train = O_OUT_TRAIN["np_y_true"], O_OUT_TRAIN["np_y_pred"]
            f_loss, n_batch = O_OUT_TRAIN["f_loss"], O_OUT_TRAIN["n_batch"]
            f_train_acc = accuracy_score(np_y_true_train, np_y_pred_train)
            
            # valiad batch 
            O_IN_VALIAD["n_epoch"] = n_epoch

            O_OUT_VALIAD = self._batch_train_test(O_IN_VALIAD)
            O_OUT_VALIAD["s_info"] = "Valiad"
            O_OUT_VALIAD["b_out_excel"] = False
            O_OUT_VALIAD["s_excel_in_path"] = dev_path
            O_OUT_VALIAD["s_excel_out_path"] = ""
            
            np_y_true_valiad, np_y_pred_valiad = O_OUT_VALIAD["np_y_true"], O_OUT_VALIAD["np_y_pred"]
            f_valiad_loss = O_OUT_VALIAD["f_loss"]
            
            if n_epoch == self.epochs - 1:
                O_OUT_TRAIN["b_out_excel"] = True
                O_OUT_VALIAD["b_out_excel"] = True

            # if (n_epoch + 1) % 10 == 0:
                # self._show_statistic_info(O_OUT_TRAIN)
                # self._show_statistic_info(O_OUT_VALIAD)

            # lf_valiad_loss.append(f_valiad_loss)
            # if best save self.model
            f_valiad_batch_avg_acc = O_OUT_VALIAD["batch_avg_acc"]
            if f_valiad_batch_avg_acc > f_valiad_avg_batch_best_acc:
                n_patience_count = 0
                f_valiad_avg_batch_best_acc = f_valiad_batch_avg_acc

                torch.save(self._net, self._s_best_model_with_weight_path) # Save whole model
                s_msg = "save best model success, valiad avg batch acc %.7f" % (f_valiad_avg_batch_best_acc)
                self._run_logger and self._run_logger.info(s_msg)
                self._b_print and print(s_msg)
            else:
                n_patience_count += 1

            s_msg = "save model epoch:%d\tstep:%d\ttrain loss:%.7f\ttrain acc:%.7f\tvaliad loss:%.7f valiad avg batch acc:%.7f" %\
                (n_epoch + 1, n_batch + 1, f_loss, f_train_acc, f_valiad_loss, f_valiad_batch_avg_acc)    
            # torch.save(self._net.state_dict(), self._s_weight_file)
            self._run_logger and self._run_logger.info(s_msg)
            self._b_print and print(s_msg)

            if n_patience_count > self.patience:
                s_msg = "early stop, patience %d, best valiad acc:%.7f" % (self.patience, f_valiad_avg_batch_best_acc)
                self._run_logger and self._run_logger.info(s_msg)
                self._b_print and print(s_msg)
                break

        # torch.save(self._net.cpu(), self._s_model_with_weight_path) # Save whole model
        # s_msg = "save model success %s" % self._s_model_with_weight_path
        self._run_logger and self._run_logger.info(s_msg)
        self._b_print and print(s_msg)
        return self.evaluate(test_path)

    def load_model(self,model_path):
        self._net = torch.load(model_path)

    def demo(self, text):
        lf_porb = self.demo_text_list([text])[0]
        return lf_porb
       
    def demo_text_list(self,text_list):
        ls_token_all = self._text_2_tokens(text_list)

        ln_id_all = self._token_2_id(ls_token_all)

        l_label_id = [(0, e) for e in ln_id_all]
        
        data_demo_set = TorchDataSet(l_label_id)
        data_demo_loader = DataLoader(dataset=data_demo_set, batch_size=self.batch_size, shuffle=False)

        self._net.eval()   

        np_y_pred = None
        with torch.no_grad():
            for i, (t_y_batch, t_x_batch) in enumerate(data_demo_loader):
                t_x_batch = t_x_batch.long()
                
                if self._b_use_cuda:
                    t_x_batch = t_x_batch.cuda()  
                t_batch_out_onehot = self._net(t_x_batch)
                
                if self._b_use_cuda:
                    t_batch_out_onehot = t_batch_out_onehot.cpu()

                np_y_bantch_onehot = t_batch_out_onehot.detach().numpy() # [batch_size, class_num]
                np_y_pred = np_y_bantch_onehot if np_y_pred is None else np.concatenate((np_y_pred, np_y_bantch_onehot),axis=0)
        if self.num_labels==2:
            pred_list = np_y_pred[:,1]
        else:
            pred_list = np.argmax(np_y_pred, axis=1).flatten()
        return pred_list

    def _text_2_tokens(self, ls_text, s_mode="jieba", b_trunc=True):
        # text convert tokens
        ls_token_all = []
        for i, s_text in enumerate(ls_text):
            ls_token_one = []
            if s_mode == "jieba": 
                for e in jieba.cut(s_text):
                    if e not in self._ls_stop_token:
                        ls_token_one.append(e)
        
            ls_token_all.append(ls_token_one)

        return ls_token_all
    
    def _token_2_id(self, ls_token_all):
        ln_id_all = []
        for i, ls_token_one in enumerate(ls_token_all):
                    
            ln_id_one = [0 for j in range(self.max_len)]
            n_row = 0
            for j, s_token in enumerate(ls_token_one):
                if n_row >= self.max_len:
                    break
                if s_token in self._m_word_id.keys():
                    ln_id_one[n_row] = self._m_word_id[s_token]
                    n_row += 1 

            ln_id_all.append(ln_id_one)

            # if (i+1) % 1000 == 0:
            #     s_msg = "token convert id,  num:%d" % (i+1)
            #     self._run_logger and self._run_logger.info(s_msg)
            #     self._b_print and print(s_msg)

        return ln_id_all   

    def _get_torch_data_loader(self, s_data_excel_path):
        # s_msg = "get torch data loader for %s" % s_data_excel_path
        # self._run_logger and self._run_logger.info(s_msg)
        # self._b_print and print(s_msg)
        
        df = load_df(s_data_excel_path)
        ls_label = df["label"]
        ls_text = [e if isinstance(e, str) else str(e) for e in df["text"]]
        
        s_msg = "text len %d" % len(ls_text)
        self._run_logger and self._run_logger.info(s_msg)
        self._b_print and print(s_msg)

        ls_token_all = self._text_2_tokens(ls_text)

        ln_id_all = self._token_2_id(ls_token_all)

        l_label_id = [(e1, e2) for e1, e2 in zip(ls_label, ln_id_all)]
        
        data_set = TorchDataSet(l_label_id)
        data_loader = DataLoader(dataset=data_set, batch_size=self.batch_size, shuffle=True)

        s_msg = "load text num %d" % len(l_label_id)
        self._run_logger and self._run_logger.info(s_msg)
        self._b_print and print(s_msg)

        return data_loader, df

    def _batch_train_test(self, O_IN):
        n_epoch = O_IN["n_epoch"]
        b_train = O_IN["b_train"]  
        data_loader = O_IN["data_loader"]

        if b_train:
            self._net.train()
        else:
            self._net.eval()     

        f_acc_sum = 0
        n_batch_num = 0

        np_y_true, np_y_pred, np_y_onehot = None, None, None
        for i, (t_y_batch, t_x_batch) in enumerate(data_loader):
            # t_x_batch, t_y_batch = t_x_batch.type(torch.LongTensor), t_y_batch.type(torch.LongTensor)
            t_x_batch, t_y_batch = t_x_batch.long(), t_y_batch.long()
            
            if self._b_use_cuda:
                t_x_batch, t_y_batch = t_x_batch.cuda(), t_y_batch.cuda()   
            t_out_onehot = self._net(t_x_batch)
            t_loss = self._loss_fun(t_out_onehot, t_y_batch)
            
            if b_train:
                self._optimizer.zero_grad()
                t_loss.backward()
                self._optimizer.step()
            
            if self._b_use_cuda:
                t_loss, t_y_batch, t_out_onehot = t_loss.cpu(), t_y_batch.cpu(), t_out_onehot.cpu()

              
            np_y_batch_true = t_y_batch.numpy()
            np_y_batch_pred = torch.topk(t_out_onehot, 1)[1].view(-1).numpy()
            np_y_bantch_onehot = t_out_onehot.detach().numpy()

            np_y_true = np_y_batch_true if np_y_true is None else np.append(np_y_true, np_y_batch_true)
            np_y_pred = np_y_batch_pred if np_y_pred is None else np.append(np_y_pred, np_y_batch_pred) 
            np_y_onehot = np_y_bantch_onehot if np_y_onehot is None else np.append(np_y_onehot, np_y_bantch_onehot)

            f_acc_batch = accuracy_score(np_y_batch_true, np_y_batch_pred)

            f_acc_sum += f_acc_batch
            n_batch_num += 1
            # if (i + 1) % 1 == 0:
            #     s_info = "Train\t" if b_train else "Test(Valiad)"
            #     s_info += "epoch: %d\tstep: %d\tbatch loss:%.7f\tbatch acc:%.7f" % (n_epoch +1, i + 1, t_loss.item(), f_acc_batch)
            #     print(s_info)
            
                
        O_OUT = {"np_y_true": np_y_true, "np_y_pred": np_y_pred, "np_y_onehot": np_y_onehot, "f_loss": t_loss.item(), "n_batch": i, "batch_avg_acc": f_acc_sum/n_batch_num}
        return O_OUT
    
    def _show_statistic_info(self, O_IN):
        s_info = O_IN["s_info"]
        s_excel_in_path = O_IN["s_excel_in_path"]
        s_excel_out_path = O_IN["s_excel_out_path"]
        np_y_true = O_IN["np_y_true"]
        np_y_pred = O_IN["np_y_pred"]
        b_out_excel = O_IN["b_out_excel"]
        
        f_acc = accuracy_score(np_y_true, np_y_pred)
        f_auc = roc_auc_score(np_y_true, np_y_pred)
        matrix = confusion_matrix(np_y_true, np_y_pred)
        s_report = classification_report(np_y_true, np_y_pred,digits=4)
        s_mat = "\n"
        for line in matrix :
            for n in line:
                s_mat += "%d " % n
            s_mat += "\n"

        s_msg = ""
        s_msg += "\n%s" % s_info
        s_msg += "\n Acc %.7f" % f_acc
        s_msg += "\n Auc %.7f" % f_auc
        s_msg += "\nConfusion matrix:"
        s_msg += s_mat
        s_msg += ""
        s_msg += "Result"
        s_msg += s_report
        s_msg += ""
       
        self._run_logger and self._run_logger.info(s_msg)
        self._b_print and print(s_msg)

    def release(self):
        del self._net
        del self._embed_mgr
        del self._np_embed_matrix
        del self._np_pad_vec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        pass