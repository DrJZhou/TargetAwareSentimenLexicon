#!/usr/bin/env python
# coding: utf-8




import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import copy
import json
# Load pre-trained model tokenizer (vocabulary)
device = torch.device("cuda", 0)
domain = 'lap'
base_path = './bert_model/{}/lm_pretraining/'.format(domain)
# base_path = './bert_model/sentiment/'
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained(base_path)
model = BertForMaskedLM.from_pretrained(base_path).to(device)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()


def prediction(text, masked_index_start=3, masked_index_end=7):
#     print(text)
    tokenized_text = tokenizer.tokenize(text)
#     print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokenized_text_masked = copy.deepcopy(tokenized_text)
    for masked_index in range(masked_index_start, masked_index_end):
        tokenized_text_masked[masked_index] = '[MASK]'
#     print(tokenized_text, tokenized_text_masked)
    indexed_tokens_masked = tokenizer.convert_tokens_to_ids(tokenized_text_masked)
    with torch.no_grad():
        tokens_tensor = torch.tensor([indexed_tokens_masked]).to(device)
        predictions = model(tokens_tensor)
        del tokens_tensor
    import math
    ans = []
    for masked_index in range(masked_index_start, masked_index_end):
        predicted_rate = predictions[0, masked_index, indexed_tokens[masked_index]].item()
        ans.append([tokenized_text[masked_index], math.exp(-predicted_rate)])
    return ans


# In[ ]:

def get_aspects(domain="lap"):
    aspects = set()
    if domain == "lap":
        base_path = "./data_asc/14lap_train.json"
        data = json.load(open(base_path, 'r'))
        for tmp in data:
            opi2asp = tmp['Opi2Asp']
            text_tokens = tmp['text_token']
            for o_a in opi2asp:
                aspect = ""
                for x in text_tokens[o_a[3]: o_a[4] + 1]:
                    if x.startswith("##"):
                        aspect += x[2:]
                    else:
                        aspect += ' ' + x
                #         aspect = [x[2:] if x.startswith("##") else x for x in text_tokens[o_a[3]: o_a[4]+1]]
                aspect = aspect.strip()
                aspects.add(aspect)
    else:
        for year in ['14', '15', '16']:
            base_path = "./data_asc/{}res_train.json".format(year)
            data = json.load(open(base_path, 'r'))
            for tmp in data:
                opi2asp = tmp['Opi2Asp']
                text_tokens = tmp['text_token']
                for o_a in opi2asp:
                    aspect = ""
                    for x in text_tokens[o_a[3]: o_a[4] + 1]:
                        if x.startswith("##"):
                            aspect += x[2:]
                        else:
                            aspect += ' ' + x
                    #         aspect = [x[2:] if x.startswith("##") else x for x in text_tokens[o_a[3]: o_a[4]+1]]
                    aspect = aspect.strip()
                    aspects.add(aspect)
    return aspects


def mask_language(domain='res'):
    cut_words = 5
    aspects = get_aspects(domain=domain)

    import math
    ans = {}
    aspect_num = {}
    index = 0
    n_gram = 1
    fr = open("./sentiment/{}/train.tsv".format(domain), 'r')
    line = fr.readline()
    while line.strip() != "":
        if index%100 == 0 and index != 0:
            print(index)
            with open("ans_ml_{}_{}_final.json".format(domain, n_gram), 'w') as f:
                json.dump(ans, f)

            with open("aspect_ml_{}_number_final.json".format(domain), 'w') as f:
                json.dump(aspect_num, f)
        index += 1
        line = line.strip().split("\t")
        if len(line) != 2:
            print(line)
            line = fr.readline()
            continue
        text = line[0]
        label = int(line[1])
        tokenized_text = tokenizer.tokenize(text)
        
        tokenized_text = ['[CLS]'] + tokenized_text[: 480] + ['[SEP]']
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
#         print(len(tokenized_text), len(indexed_tokens))
#         text = " ".join(tokenized_text)
        text = ""
        for x in tokenized_text:
            if x.startswith("##"):
                text += x[2:]
            else:
                text += ' ' + x
        text = text.strip()
        for aspect in aspects:
            if " "+aspect+" " not in text:
                continue
            text_left = text[: text.find(aspect)]
            text_right = text[text.find(aspect)+len(aspect): ]
            text_left_len = len(tokenizer.tokenize(text_left))
            text_right_len = len(tokenizer.tokenize(text_right))
            if text_right_len == 0:
                continue
#             print(text, "\n", text_left, "\n", aspect, "\n", text_right)
            masked_index_start =  len(tokenizer.tokenize(text_left + aspect))
            if text_right_len < cut_words:
                masked_index_end = masked_index_start + text_right_len
            else:
                masked_index_end = masked_index_start + cut_words
#             print(text_left + aspect + text_right)
            try:    
                ans_predict = prediction(text_left + aspect + text_right, masked_index_start=masked_index_start, masked_index_end=masked_index_end)
            except:
                print(text_left, "\n", aspect, "\n", text_right)
                continue
            # print(ans_predict)
            if aspect not in ans:
                ans[aspect] = {}
            aspect_num[aspect] = aspect_num.get(aspect, 0) + 1
            for aspect_replace in list(aspects)[: 60]:
                if aspect_replace == aspect:
                    continue

                masked_index_start_replace =  len(tokenizer.tokenize(text_left + aspect_replace))
                if text_right_len < cut_words:
                    masked_index_end_replace = masked_index_start_replace + text_right_len
                else:
                    masked_index_end_replace = masked_index_start_replace + cut_words
                try:
                    ans_predict_replace = prediction(text_left + aspect_replace + text_right, masked_index_start=masked_index_start_replace, masked_index_end=masked_index_end_replace)
                except:
                    print(text_left, "\n", aspect, "\n", text_right)
                    continue
    #             print(ans_predict_replace, aspect_replace)

                for i in range(len(ans_predict)):
                    if i < len(ans_predict)-1 and ans_predict[i+1][0].startswith("##"):
                        word = ans_predict[i][0] + ans_predict[i+1][0].replace("##", "")
                        value = ((math.fabs(ans_predict_replace[i][1] - ans_predict[i][1])/ans_predict[i][1])+(math.fabs(ans_predict_replace[i+1][1] - ans_predict[i+1][1])/ans_predict[i+1][1]))/2.0
                        i += 1
                    else:
                        word = ans_predict[i][0]
                        value = (math.fabs(ans_predict_replace[i][1] - ans_predict[i][1])/ans_predict[i][1])
                    if word not in ans[aspect]:
                        ans[aspect][word] = []
                    ans[aspect][word].append(value)

            masked_index_end =  len(tokenizer.tokenize(text_left))
            if text_left_len < cut_words:
                masked_index_start = masked_index_end - text_left_len
            else:
                masked_index_start = masked_index_end - cut_words
            
            try:
                ans_predict = prediction(text_left + aspect + text_right, masked_index_start=masked_index_start, masked_index_end=masked_index_end)
            except:
                print(text_left, "\n", aspect, "\n", text_right)
                continue
            
            for aspect_replace in list(aspects)[: 60]:
                if aspect_replace == aspect:
                    continue

                masked_index_end_replace =  len(tokenizer.tokenize(text_left))
                if text_left_len < cut_words:
                    masked_index_start_replace = masked_index_end_replace - text_left_len
                else:
                    masked_index_start_replace = masked_index_end_replace - cut_words
                try:
                    ans_predict_replace = prediction(text_left + aspect_replace + text_right, masked_index_start=masked_index_start_replace, masked_index_end=masked_index_end_replace)
                except:
                    print(text_left, "\n", aspect, "\n", text_right)
                    continue
                    #             print(ans_predict_replace, aspect_replace, text_left, aspect_replace, )
    #             if len(ans_predict_replace) == 0:
    #                 continue
                for i in range(len(ans_predict)):
                    if i < len(ans_predict)-1 and ans_predict[i+1][0].startswith("##"):
                        word = ans_predict[i][0] + ans_predict[i+1][0].replace("##", "")
                        value = ((math.fabs(ans_predict_replace[i][1] - ans_predict[i][1])/ans_predict[i][1])+(math.fabs(ans_predict_replace[i+1][1] - ans_predict[i+1][1])/ans_predict[i+1][1]))/2.0
                        i += 1
                    else:
                        word = ans_predict[i][0]
                        value = (math.fabs(ans_predict_replace[i][1] - ans_predict[i][1])/ans_predict[i][1])
                    if word not in ans[aspect]:
                        ans[aspect][word] = []
                    ans[aspect][word].append(value)
            
        line = fr.readline()
    fr.close()

    with open("ans_ml_{}_{}_final.json".format(domain, n_gram), 'w') as f:
        json.dump(ans, f)

    with open("aspect_ml_{}_number_final.json".format(domain), 'w') as f:
        json.dump(aspect_num, f)


mask_language(domain=domain)


def get_top_k_lap(filename="ans_ml_lap_1.json", num_file="aspect_ml_number.json"):
    import json
    with open(filename, 'r') as f:
        ans = json.load(f)


    with open(num_file, 'r') as f:
        aspect_num = json.load(f)


    import numpy as np
    ranking = {}
    average_acc = 0.0
    count = 0
    for aspect in ans.keys():
        words = {}
        for word in ans[aspect].keys():
            words[word] = np.sum(ans[aspect][word])/len(ans[aspect][word])
    #     print(words)
        ranking[aspect] = sorted(words.items(), key=lambda item: item[1], reverse=True)
    #     print(aspect, ranking[aspect])
        if aspect_num.get(aspect, 0) < 15:
            continue
        terms = ""
        cnt = 0
        total_num = 0
        for i in range(len(ranking[aspect])):
            term_value = ranking[aspect][i]
    #         if term_value[1]< 2.5:
    #             continue
            if i >= 10:
                break
            total_num += 1
            terms += term_value[0] + ": " + str(term_value[1]) + " "
        print("{}/{}".format(cnt, total_num), aspect, " -> ",terms)

# get_top_k_lap()




