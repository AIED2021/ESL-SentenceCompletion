
import sys
import re
import pickle
## import pandas as pd 
import json
import sys
import os

"""
processed_data has multilines, each line is a sc question
an example of sc question
{
    "id": "01"
    "stem": "i'm so happy ___ do it.",
    "choice": "A.to B.on C.in D.of"
}
""" 
with open("*.pkl", "rb") as f:
    processed_data = pickle.load(f)
print(len(processed_data))
from pprint import pprint
pprint(processed_data[0])
# In[ ]:




if sys.argv[1] == "bart":
    from ModelBart import AutoFillBlank
elif sys.argv[1] == "bert":
    from ModelBert import AutoFillBlank
elif sys.argv[1] == "electra":
    from ModelElectra import AutoFillBlank
elif sys.argv[1] == "xlnet":
    from ModelXlnet import AutoFillBlank
elif sys.argv[1] == "roberta":
    from ModelRoberta import AutoFillBlank
else:
    raise ValueError
file_path = sys.argv[2]



def process_predict(processed_data, model_name):
    predict_file = file_path
    if os.path.exists(predict_file):
        predict_data = pickle.load(open(predict_file, "rb"))
        return predict_data
    model = AutoFillBlank()
    predict_data = []
    for i in range(len(processed_data)):
        json_line = processed_data[i]
        text_lst = json_line["content"]
        uid = json_line["id"]
        model_out = model.predict(json_line["choice_dict"])[1]
        json_line["pred"] = model_out["ans"]
        json_line["prob_dict"] = model_out["prob_dict"]
        json_line["prob_dict"] = model_out["prob_dict"]
        predict_data.append(json_line)
#         break
        if i % 100 == 0:
            print(str(i) + "|", end = "\t")
    with open(predict_file, "wb") as fw:
        pickle.dump(predict_data, fw)
    return predict_data

predict_data_origin = process_predict(processed_data, sys.argv[1])
print(len(predict_data_origin), len(processed_data))

def get_blank_word_sum(predict_data):
    word_num_dict = {}
    final_data = []
    for i in range(len(predict_data)):
        json_line = predict_data[i]
        select_dict = json_line["select_dict"]
        sum_words = 0
        for k, v in select_dict.items():
            blanks = v[1:].split(";")
            blank_words = 0
            for blank in blanks:
                blank_words += len(blank.strip().split(" "))
            sum_words = max(blank_words, sum_words)
        json_line["blank_word_sum"] = sum_words
        if sum_words not in word_num_dict:
            word_num_dict[sum_words] = 1
        else:
            word_num_dict[sum_words] += 1
        final_data.append(json_line)
        # if words_num == 3:
        #     print(json_line)
    print(word_num_dict)
    return final_data

predict_data = get_blank_word_sum(predict_data_origin)


def cal_seq_len(stem):
    words_lst = stem.split(" ")
    words_lst = [itm.strip() for itm in words_lst]
    pattern = r"[a-zA-Z]"
    length = 0
    for word in words_lst:
        if re.findall(pattern, word) != []:
            length += 1
    return length

def get_class_acc(blank_num_lst, word_num_lst, seq_len_lst, b2_word_num_lst, predict_data, threshold=0.0):
    seq_len_dict = {}
    
    all_num = 0; right_num = 0
    assert len(predict_data) == len(processed_data)
    i = 0
    for line in predict_data:
        blank_num, word_num, b2_word_num = line["blank_num"], line["word_num"], line["blank_word_sum"]
        if blank_num_lst and blank_num not in blank_num_lst:
            continue
        if word_num_lst and word_num not in word_num_lst:
            continue
        if b2_word_num_lst and b2_word_num not in b2_word_num_lst:
            continue
        # choice_dict = line["choice_dict"]["choice_dict"]
        # sent_lens = [len([itm for itm in v.split(" ") if itm.isalpha()]) for k, v in choice_dict.items()]
        # seq_len = max(sent_lens)
        seq_len = cal_seq_len(processed_data[i]["stem"])
        i += 1
        if seq_len not in seq_len_dict:
            seq_len_dict[seq_len] = 1
        else:
            seq_len_dict[seq_len] += 1

        if seq_len_lst and seq_len not in seq_len_lst:
            continue
        # print(line)
        prob_lsts = [x[1] for x in line["prob_dict"].items()]
        prob = max(prob_lsts)
        if prob >= threshold:
            if line["ans"] == line["pred"]:
                right_num += 1
            all_num += 1
        

    # for k, v in sorted(seq_len_dict.items(), key=lambda x: x[0]):
    #     print(k, v, sep="\t")
    
    if all_num == 0:
        return 0, 0
    return "\t".join([str(right_num / all_num), str(all_num)])


#[0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0]

print("threshold PR")
for threshold in [x / 100 for x in range(0, 100, 5)]:
    print("threshold", threshold, get_class_acc([], [], [], [], predict_data, threshold=threshold), end="\n", sep="\t")


print("classify by word num of blank and blank num")
print("one blank|one word", get_class_acc([1], [1], [], [], predict_data), end="\t")
print("one blank|multi-words", get_class_acc([1], list(range(2, 50)), [], [], predict_data), end="\t")
print("multi-blanks|one word", get_class_acc(list(range(2, 50)), [1], [], [], predict_data), end="\t")
print("multi-blanks|multi-words", get_class_acc(list(range(2, 50)), list(range(2, 50)), [], [], predict_data), end="\t")
print("\n")
print("classify by word num of blank")
print("one word", get_class_acc([], [1], [], [], predict_data), end="\t")
print("multi-words", get_class_acc([], list(range(2, 50)), [], [], predict_data), end="\t")
print("\n")
print("classify by num of blank")
print("one blank", get_class_acc([1], [], [], [], predict_data), end="\t")
print("multi-blanks", get_class_acc(list(range(2, 50)), [], [], [], predict_data), end="\t")
print("\n")
print("classify by word num of blank(one blank sc questions)")
print("1 word", get_class_acc([1], [1], [], [], predict_data), end="\t")
print("2 words", get_class_acc([1], [2], [], [],predict_data), end="\t")
print("3 words", get_class_acc([1], [3], [], [],predict_data), end="\t")
print("4+words", get_class_acc([1], list(range(4, 50)), [], [],predict_data), end="\t")
print("\n")

print("word num of two blanks")
print("2 words", get_class_acc([2], [], [], [2], predict_data), end="\t")
print("3 words", get_class_acc([2], [], [], [3], predict_data), end="\t")
print("4 words", get_class_acc([2], [], [], [4], predict_data), end="\t")
print("5 words", get_class_acc([2], [], [], list(range(5, 50)), predict_data), end="\t")
print("\n")

print("length of sc question's stem")
print("0-5", get_class_acc([], [], [0,1,2,3,4,5], [], predict_data), end="\t")
print("6-10", get_class_acc([], [], [6,7,8,9,10], [], predict_data), end="\t")
print("11-15", get_class_acc([], [], [11,12,13,14,15], [], predict_data), end="\t")
print("16-20", get_class_acc([], [], [16,17,18,19,20], [], predict_data), end="\t")
print("21+", get_class_acc([], [], list(range(21, 50)), [], predict_data), end="\t")
print("\n")

