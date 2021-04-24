import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification, BertConfig
import copy
import json
import pickle

device = torch.device("cuda", 0)
domain = "lap"
base_path = './bert_model/{}/sentiment/'.format(domain)
num_labels = 5
config = BertConfig.from_pretrained(base_path, num_labels=num_labels)
tokenizer = BertTokenizer.from_pretrained(base_path)
model = BertForSequenceClassification.from_pretrained(base_path).to(device)
for name, param in model.named_parameters():
    if name != "bert.embeddings.word_embeddings.weight":
        print(name)
        param.requires_grad=False


def bert_token(text, label):
    tokenized_text = tokenizer.tokenize(text)
    #     print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    label_tensor = torch.tensor([label], dtype=torch.long)
    return tokens_tensor, label_tensor


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


def got_grad_change(text, label):
    ans = []
    input_ids, label = bert_token(text, label)
    input_tokens = tokenizer.tokenize(text)
    #     print(input_ids.size(), len(input_tokens))
    outputs = model(input_ids.to(device), labels=label.to(device))
    loss, logits = outputs[:2]
    #     print(loss, logits)
    loss.backward()
    #     print(tmp)
    grad = model.bert.embeddings.word_embeddings.weight.grad
    for i in range(0, input_ids.size(1)):
        word_grad = grad[input_ids][0][i]
        length = torch.sqrt(torch.sum(word_grad * word_grad)).item()
        #         print(input_tokens[i-1], length)
        ans.append([input_tokens[i], length])
    #     print(grad[input_ids][0][0])
    model.zero_grad()
    return ans


def bert_token_wo_label(text):
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    return tokens_tensor


def got_label(text):
    input_ids = bert_token_wo_label(text)
    with torch.no_grad():
        outputs = model(input_ids.to(device))
        logits = outputs[0]
    label = logits.argmax().item()
    pro = list(torch.softmax(logits, dim=-1)[0].cpu().numpy())
    return label, pro


def bert_token_wo_label_batch(text_list):
    text_input = []
    max_len = None
    for text in text_list:
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        if max_len is None:
            max_len = len(indexed_tokens) + 5
        indexed_tokens = indexed_tokens + [0]*(max_len-len(indexed_tokens))
        text_input.append(indexed_tokens)
    tokens_tensor = torch.tensor(text_input)
    return tokens_tensor


def got_label_batch(text_list):
    input_ids = bert_token_wo_label_batch(text_list)
    with torch.no_grad():
        outputs = model(input_ids.to(device))
        logits = outputs[0]
    label = logits.argmax(dim=-1).cpu().numpy()
    pro = torch.softmax(logits, dim=-1).cpu().numpy()
    return label, pro


def cal_score(domain="res", n_gram=1):
    cut_words = 10
    # aspects = get_aspects(domain=domain)
    # print(len(aspects))
    ans = []
    index = 0
    fr = open("./sentiment/{}/aspect_sentence_0_label.tsv".format(domain), 'r')
    line = fr.readline()
    while line.strip() != "":
        if index % 10000 == 0 and index != 0:
            print(index)
            with open("aspect_aware_gradient_{}_{}_final_all.pkl".format(domain, n_gram), 'wb') as f:
                pickle.dump(ans, f)
        index += 1
        line = line.strip().split("\t")
        if len(line) != 8:
            print(line)
            line = fr.readline()
            continue
        text = line[0]
        aspect = line[1]
        label = int(line[2])
        pro = [float(x) for x in line[3:]]
        aspect_index = text.find(" " + aspect + " ")
        if aspect_index == -1:
            line = fr.readline()
            continue
        ans_sample = {}
        ans_sample["orignal"] = {"text": text, "aspect": aspect, "label": label, "prob": pro}
        ans_predict = got_grad_change(text, label=label)
        ans_predict_word = []
        i = 0
        while i < len(ans_predict):
            subpiece_num = 0
            value_tmp = ans_predict[i][1]
            word_tmp = ans_predict[i][0]
            while i + subpiece_num + 1 < len(ans_predict) and ans_predict[i + subpiece_num + 1][0].startswith("##"):
                word_tmp += ans_predict[i + subpiece_num + 1][0].replace("##", "")
                value_tmp += ans_predict[i + subpiece_num + 1][1]
                subpiece_num += 1
            i += subpiece_num + 1
            ans_predict_word.append([word_tmp, value_tmp / (subpiece_num + 1)])

        ans_sample['gradient'] = ans_predict_word

        text_left = text[:aspect_index].strip()
        start = len(text_left.split(" "))
        end = start + len(aspect.strip().split(" "))
        text_token = text.split(" ")
        # text_list = []
        # word_list = []
        ans_replace_label = []
        for i in range(start - cut_words, start):
            if i < 0:
                continue
            text_replace = " ".join(text_token[:i]) + " " + " ".join(text_token[i + 1:])
            # word_list.append(text_token[i])
            # text_list.append(text_replace)
            label_repaced, pro_replaced = got_label(text_replace)
            ans_replace_label.append([text_token[i], label_repaced, pro_replaced])

        # if len(word_list) > 0:
        #     label_repaced, pro_replaced = got_label_batch(text_list)
        #     for i in range(len(label_repaced)):
        #         ans_replace_label.append([word_list[i], label_repaced[i], pro_replaced[i]])

        # text_list = []
        # word_list = []
        for i in range(end, end + cut_words):
            if i >= len(text_token):
                break
            text_replace = " ".join(text_token[:i]) + " " + " ".join(text_token[i + 1:])
            label_repaced, pro_replaced = got_label(text_replace)
            ans_replace_label.append([text_token[i], label_repaced, pro_replaced])
            # word_list.append(text_token[i])
            # text_list.append(" ".join(text_token[:i]) + " " + " ".join(text_token[i + 1:]))
        # if len(word_list) > 0:
        #     label_repaced, pro_replaced = got_label_batch(text_list)
        #     for i in range(len(label_repaced)):
        #         ans_replace_label.append([word_list[i], label_repaced[i], pro_replaced[i]])
        ans_sample['sentiment'] = ans_replace_label
        if index <= 2:
            print(ans_sample, flush=True)
        ans.append(ans_sample)
        line = fr.readline()
    fr.close()

    with open("ans_aware_gradient_{}_{}_final_all.pkl".format(domain, n_gram), 'wb') as f:
        pickle.dump(ans, f)


def cal_score_gradient(domain="res", n_gram=1):
    cut_words = 10
    # aspects = get_aspects(domain=domain)
    # print(len(aspects))
    ans = []
    index = 0
    fr = open("./sentiment/{}/aspect_sentence_0_label.tsv".format(domain), 'r')
    line = fr.readline()
    while line.strip() != "":
        if index % 10000 == 0 and index != 0:
            print(index)
            with open("aspect_aware_gradient_{}_{}_final_all_only_gradient.pkl".format(domain, n_gram), 'wb') as f:
                pickle.dump(ans, f)
        index += 1
        line = line.strip().split("\t")
        if len(line) != 8:
            print(line)
            line = fr.readline()
            continue
        text = line[0]
        aspect = line[1]
        label = int(line[2])
        if label == 2:
            line = fr.readline()
            continue

        if label > 2:
            label_revise = 0
        else:
            label_revise = 4
        pro = [float(x) for x in line[3:]]
        aspect_index = text.find(" " + aspect + " ")
        if aspect_index == -1:
            line = fr.readline()
            continue
        ans_sample = {}
        ans_sample["orignal"] = {"text": text, "aspect": aspect, "label": label, "prob": pro}
        ans_predict = got_grad_change(text, label=label)
        ans_predict_word = []
        i = 0
        while i < len(ans_predict):
            subpiece_num = 0
            value_tmp = ans_predict[i][1]
            word_tmp = ans_predict[i][0]
            while i + subpiece_num + 1 < len(ans_predict) and ans_predict[i + subpiece_num + 1][0].startswith("##"):
                word_tmp += ans_predict[i + subpiece_num + 1][0].replace("##", "")
                value_tmp += ans_predict[i + subpiece_num + 1][1]
                subpiece_num += 1
            i += subpiece_num + 1
            ans_predict_word.append([word_tmp, value_tmp / (subpiece_num + 1)])

        ans_sample['gradient'] = ans_predict_word

        ans_sample['sentiment'] = []
        if index <= 2:
            print(ans_sample, flush=True)
        ans.append(ans_sample)
        line = fr.readline()
    fr.close()

    with open("ans_aware_gradient_{}_{}_final_all_only_gradient.pkl".format(domain, n_gram), 'wb') as f:
        pickle.dump(ans, f)


if __name__ == '__main__':
    # cal_score(domain=domain)
    cal_score_gradient(domain=domain)