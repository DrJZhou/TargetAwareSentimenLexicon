import json

'''
get top k opinion words for each domain with different methods ("ml", "gradient")
'''


def get_top_k(domain="lap", method="ml"):
    if method == 'ml':
        filename = "./results/ans_ml_{}_1_final.json".format(domain)
        num_file = "./results/aspect_ml_{}_number_final.json".format(domain)
    else:
        filename = "./results/ans_gradient_{}_1_final.json".format(domain)
        num_file = "./results/aspect_number_{}_1_gradient_final.json".format(domain)

    with open(filename, 'r') as f:
        ans = json.load(f)

    with open(num_file, 'r') as f:
        aspect_num = json.load(f)
    stop_words_str = "re###able###ask###due###give###etc###tell###also###aaa###won###un###us###can###why###who###then###got###get###ll###but###lo###ki###can###wi###let###ve###his###could###still###about###this###them###so###or###if###would###only###both###been###when###our###as###be###by###he###him###she###her###they###their###your###after###with###there###what###for###at###we###you###is###!###,###,###.###;###:###are###these###those###other###were###on###its###is###was###has###will###my###how###do###does###a###an###am###me###gets###get###the###in###than###it###had###have###from###s###and###since###too###shows###that###to###of###at###itself###from###being###how###what###who###which###where###had###wants###b###c###d###e###f###g###h###i###j###k###l###m###n###o###p###q###r###s###t###u###v###w###x###y###z###-###_###'###\"###[CLS]###[SEP]".split(
        "###")
    stop_words = {}
    for word in stop_words_str:
        stop_words[word] = 1
    import numpy as np
    ranking = {}
    average_acc = 0.0
    count = 0
    opinion_word_num = 100
    fr_to = open("./results/{}_{}_words_list_{}.csv".format(domain, method, opinion_word_num), "w")
    for aspect in ans.keys():

        words = {}
        for word in ans[aspect].keys():
            words[word] = np.sum(ans[aspect][word]) / len(ans[aspect][word])
        # print(words)
        ranking[aspect] = sorted(words.items(), key=lambda item: item[1], reverse=True)
        #     print(aspect, ranking[aspect])
        if aspect_num.get(aspect, 0) < 15:
            continue
        fr_to.write(aspect)
        terms = ""
        cnt = 0
        total_num = 0
        for i in range(len(ranking[aspect])):
            term_value = ranking[aspect][i]
            #         if term_value[1]< 2.5:
            #             continue
            if term_value[0].startswith("##"):
                continue
            if term_value[0] in stop_words:
                # print(term_value)
                continue
            if total_num >= opinion_word_num:
                break

            total_num += 1
            terms += term_value[0] + ": " + str(term_value[1]) + " "
            fr_to.write("," + term_value[0])
        print(aspect, " -> ", terms)
        fr_to.write("\n")
    fr_to.close()


def read_csv(domain='lap'):
    fr_1 = open("./results/{}_{}_words_list_{}.csv".format(domain, 'ml', 100), "r")
    language_mask = {}
    for line in fr_1.readlines():
        data = line.strip().split(",")
        aspect = data[0]
        language_mask[aspect] = set()
        for opinion in data[1:]:
            language_mask[aspect].add(opinion)

    fr_2 = open("./results/{}_{}_words_list_{}_PMI.csv".format(domain, 'colinear', 100), "r")
    PMI_conlinear = {}
    for line in fr_2.readlines():
        data = line.strip().split(",")
        aspect = data[0]
        PMI_conlinear[aspect] = set()
        for opinion in data[1:]:
            PMI_conlinear[aspect].add(opinion)

    fr_to = open("./results/{}_ML_PMI.csv".format(domain), "w")
    for aspect in language_mask.keys():
        if aspect not in PMI_conlinear:
            continue
        jiaoji = language_mask[aspect] & PMI_conlinear[aspect]
        chaji_lm = language_mask[aspect] - jiaoji
        chaji_pmi = PMI_conlinear[aspect] - jiaoji
        print(jiaoji, chaji_lm, chaji_pmi)
        fr_to.write("intersection," + aspect + "," + ",".join(list(jiaoji)) + "\n")
        fr_to.write("MLM," + aspect + "," + ",".join(list(chaji_lm)) + "\n")
        fr_to.write("PMI," + aspect + "," + ",".join(list(chaji_pmi)) + "\n")
        fr_to.write("\n")
    fr_to.close()


import pickle
import math

'''
calculate the score by forward
'''


def load_pkl(domain='res'):
    ans = {}
    with open("./results/aspect_aware_gradient_{}_1_final_all.pkl".format(domain), 'rb') as f:
        data = pickle.load(f)
        # print(data)
        gradient_change_all = {}
        word_value_gradient = {}
        aspect_value_gradient = {}
        aspect_word_value_gradient = {}
        total_value_gradient = 0.0

        foward_change_all = {}
        word_value_forward = {}
        aspect_value_forward = {}
        aspect_word_value_forward = {}
        aspect_word_num_forward = {}
        total_value_forward = 0.0

        word_value_PMI = {}
        aspect_value_PMI = {}
        aspect_word_value_PMI = {}
        total_value_PMI = 0.0

        for i in range(len(data)):
            orignal = data[i]['orignal']
            gradient = data[i]['gradient']
            sentiment = data[i]['sentiment']
            label = orignal['label']
            prob = orignal['prob']
            text = orignal['text']
            aspect = orignal['aspect']
            aspect_index = text.find(' ' + aspect + ' ')
            start_token_index = len(text[:aspect_index].strip().split(" "))
            end_token_index = start_token_index + len(aspect.split(" "))
            text_token = text.split(" ")
            near_words = set(
                text_token[start_token_index - 5: start_token_index] + text_token[end_token_index: end_token_index + 5])
            prob_change = {}
            if aspect not in foward_change_all:
                foward_change_all[aspect] = []
                aspect_word_value_forward[aspect] = {}
                aspect_word_num_forward[aspect] = {}

            if aspect not in aspect_word_value_PMI:
                aspect_word_value_PMI[aspect] = {}

            for tmp in sentiment:
                word = tmp[0]
                if word not in near_words:
                    continue
                if word in ['[CLS]', '[SEP]']:
                    continue
                label_ = tmp[1]
                prob_ = tmp[2]
                prob_change_value = math.fabs(prob[label] - prob_[label])
                if word not in prob_change:
                    prob_change[word] = prob[label] - prob_[label]

                word_value_forward[word] = word_value_forward.get(word, 0.0) + prob_change_value
                aspect_word_value_forward[aspect][word] = aspect_word_value_forward[aspect].get(word,
                                                                                                0.0) + prob_change_value
                aspect_value_forward[aspect] = aspect_value_forward.get(aspect, 0.0) + prob_change_value
                total_value_forward += prob_change_value

                word_value_PMI[word] = word_value_PMI.get(word, 0.0) + 1
                aspect_word_value_PMI[aspect][word] = aspect_word_value_PMI[aspect].get(word, 0.0) + 1
                aspect_value_PMI[aspect] = aspect_value_PMI.get(aspect, 0.0) + 1
                total_value_PMI += 1

                foward_change_all[aspect].append([word, prob_change_value])

            prob_change = sorted(prob_change.items(), key=lambda item: item[1], reverse=True)
            # if prob_change[0][1] > 0.4:
            #     # print(text, aspect, label, prob_change)
            #     if aspect not in ans:
            #         ans[aspect] = {}
            #     for tmp in prob_change:
            #         if tmp[1] > 0.4:
            #             ans[aspect][tmp[0]] = ans[aspect].get(tmp[0], 0.0) + tmp[1]
            #             aspect_word_num_forward[aspect][tmp[0]] = aspect_word_num_forward[aspect].get(tmp[0], 0.0) + 1
            if aspect not in ans:
                ans[aspect] = {}
            for tmp in prob_change:
                ans[aspect][tmp[0]] = ans[aspect].get(tmp[0], 0.0) + tmp[1]
                aspect_word_num_forward[aspect][tmp[0]] = aspect_word_num_forward[aspect].get(tmp[0], 0.0) + 1.0

            if aspect not in gradient_change_all:
                gradient_change_all[aspect] = []
                aspect_word_value_gradient[aspect] = {}
            # gradient_change = {}
            for tmp in gradient:
                # print(tmp)
                word = tmp[0]
                if word not in near_words:
                    continue
                if word in ['[CLS]', '[SEP]']:
                    continue
                grad = tmp[1]
                word_value_gradient[word] = word_value_gradient.get(word, 0.0) + grad
                aspect_word_value_gradient[aspect][word] = aspect_word_value_gradient[aspect].get(word, 0.0) + grad
                aspect_value_gradient[aspect] = aspect_value_gradient.get(aspect, 0.0) + grad
                total_value_gradient += grad
                # if word not in gradient_change:
                #     gradient_change[word] = grad
                gradient_change_all[aspect].append([word, grad])
                # gradient_change = sorted(gradient_change.items(), key=lambda item: item[1], reverse=True)
                # print(text, aspect, gradient_change)
                # gradient_change_all[aspect] = gradient_change

    # Gradient_PMI = {}
    # ans_gradient_PMI = {}
    # for aspect in aspect_word_value_gradient.keys():
    #     Gradient_PMI[aspect] = {}
    #     max_value = 0
    #     min_value = 10000
    #     for word in aspect_word_value_gradient[aspect].keys():
    #         P_O_A = aspect_word_value_gradient[aspect][word] / aspect_value_gradient[aspect]
    #         P_O = word_value_gradient[word] / total_value_gradient
    #         PMI = P_O_A / (P_O + 0.001)
    #         Gradient_PMI[aspect][word] = PMI
    #         if PMI > max_value:
    #             max_value = PMI
    #         if PMI < min_value:
    #             min_value = PMI
    #     if max_value == min_value:
    #         continue
    #     for word in Gradient_PMI[aspect].keys():
    #         Gradient_PMI[aspect][word] = (Gradient_PMI[aspect][word] - min_value) / (max_value - min_value) / 2
    #     gradient_change = sorted(Gradient_PMI[aspect].items(), key=lambda item: item[1], reverse=True)
    #     ans_gradient_PMI[aspect] = gradient_change
    #     # print(aspect, gradient_change[:10])

    foward_PMI = {}
    ans_forward_PMI = {}
    for aspect in aspect_word_value_forward.keys():
        foward_PMI[aspect] = {}
        max_value = 0
        min_value = 10000
        for word in aspect_word_value_forward[aspect].keys():
            P_O_A = aspect_word_value_forward[aspect][word] / aspect_value_forward[aspect]
            P_O = word_value_forward[word] / total_value_forward
            PMI = P_O_A / (P_O + 0.001)
            foward_PMI[aspect][word] = PMI
            if PMI > max_value:
                max_value = PMI
            if PMI < min_value:
                min_value = PMI
        if max_value == min_value:
            continue
        for word in foward_PMI[aspect].keys():
            foward_PMI[aspect][word] = (foward_PMI[aspect][word] - min_value) / (max_value - min_value) / 2
        prob_change = sorted(foward_PMI[aspect].items(), key=lambda item: item[1], reverse=True)
        ans_forward_PMI[aspect] = prob_change
        print(aspect, prob_change[:10])

    PMI = {}
    ans_PMI = {}
    for aspect in aspect_word_value_PMI.keys():
        PMI[aspect] = {}
        max_value = 0
        min_value = 10000
        for word in aspect_word_value_PMI[aspect].keys():
            P_O_A = aspect_word_value_PMI[aspect][word] / aspect_value_PMI[aspect]
            P_O = word_value_PMI[word] / total_value_PMI
            PMI_value = P_O_A / (P_O + 0.001)
            PMI[aspect][word] = PMI_value
            if PMI_value > max_value:
                max_value = PMI_value
            if PMI_value < min_value:
                min_value = PMI_value
        if max_value == min_value:
            continue
        for word in PMI[aspect].keys():
            PMI[aspect][word] = (PMI[aspect][word] - min_value) / (max_value - min_value) / 2
        PMI_change = sorted(PMI[aspect].items(), key=lambda item: item[1], reverse=True)
        ans_PMI[aspect] = PMI_change
        # print(aspect, gradient_change[:10])

    ans_forward_avg = {}
    ans_forward_sum = {}
    for aspect in ans.keys():
        ans_forward_avg[aspect] = {}
        ans_forward_sum[aspect] = {}
        for word in ans[aspect].keys():
            ans_forward_avg[aspect][word] = ans[aspect][word] / aspect_word_num_forward[aspect][word]
            ans_forward_sum[aspect][word] = ans[aspect][word]
        aspect_word_list = sorted(ans_forward_avg[aspect].items(), key=lambda item: item[1], reverse=True)
        ans_forward_avg[aspect] = aspect_word_list
        aspect_word_list = sorted(ans_forward_sum[aspect].items(), key=lambda item: item[1], reverse=True)
        ans_forward_sum[aspect] = aspect_word_list

    ans_forward = {}
    for aspect in ans.keys():
        for word in ans[aspect].keys():
            ans[aspect][word] = ans[aspect][word] / aspect_word_num_forward[aspect][word] * (
                math.log(aspect_word_num_forward[aspect][word]) + 1)
        aspect_word_list = sorted(ans[aspect].items(), key=lambda item: item[1], reverse=True)
        ans_forward[aspect] = aspect_word_list
        # print(aspect, aspect_word_list)

    with open("./results/aspect_aware_gradient_{}_word_list_only_forward_1.pkl".format(domain), 'wb') as f:
        pickle.dump(
            {'forward': ans_forward, "forward_PMI": ans_forward_PMI, "PMI": ans_PMI, 'foward_avg': ans_forward_avg,
             "forward_sum": ans_forward_sum}, f)


'''
load cPickle file to csv for forward
'''


def load_word_list(domain='res'):
    stop_words_str = "`###+###]###[###@###%###=###/###`###$###&###*###(###)###?###re###able###ask###due###give###etc###tell###also###aaa###won###un###us###can###why###who###then###got###get###ll###but###lo###ki###can###wi###let###ve###his###could###still###about###this###them###so###or###if###would###only###both###been###when###our###as###be###by###he###him###she###her###they###their###your###after###with###there###what###for###at###we###you###is###!###,###,###.###;###:###are###these###those###other###were###on###its###is###was###has###will###my###how###do###does###a###an###am###me###gets###get###the###in###than###it###had###have###from###s###and###since###too###shows###that###to###of###at###itself###from###being###how###what###who###which###where###had###wants###b###c###d###e###f###g###h###i###j###k###l###m###n###o###p###q###r###s###t###u###v###w###x###y###z###-###_###'###\"###[CLS]###[SEP]".split(
        "###")
    stop_words = {}
    for word in stop_words_str:
        stop_words[word] = 1

    ans_file = open("./results/{}_forward_4.csv".format(domain), 'w', encoding='utf-8')
    with open("./results/aspect_aware_gradient_{}_word_list_only_forward_1.pkl".format(domain), 'rb') as f:
        data = pickle.load(f)
        # ans_gradient_PMI = data['gradient']
        ans_forward = data['forward']
        ans_forward_PMI = data['forward_PMI']
        ans_forward_avg = data['foward_avg']
        ans_forward_sum = data['forward_sum']
        ans_PMI = data['PMI']

        for aspect in ans_forward_PMI.keys():
            ans_file.write("{}\n".format(aspect))
            ans_file.write("Forward_PMI")
            for tmp in ans_forward_PMI[aspect]:
                if tmp[0] in stop_words:
                    continue
                ans_file.write(",{}".format(tmp[0]))
            # print(aspect, ans_forward_PMI[aspect][: 10])
            ans_file.write("\n")

            ans_file.write("Forward_avg")
            if aspect in ans_forward_avg:
                for tmp in ans_forward_avg[aspect]:
                    if tmp[0] in stop_words:
                        continue
                    ans_file.write(",{}".format(tmp[0]))
                ans_file.write("\n")

            ans_file.write("Forward_sum")
            if aspect in ans_forward_sum:
                for tmp in ans_forward_sum[aspect]:
                    if tmp[0] in stop_words:
                        continue
                    ans_file.write(",{}".format(tmp[0]))
                ans_file.write("\n")

            ans_file.write("Forward")
            if aspect in ans_forward:
                for tmp in ans_forward[aspect]:
                    if tmp[0] in stop_words:
                        continue
                    ans_file.write(",{}".format(tmp[0]))
                print(aspect, ans_forward[aspect][: 10])
            ans_file.write("\n")
            # ans_file.write("Backward_PMI")
            # if aspect in ans_gradient_PMI:
            #     for tmp in ans_gradient_PMI[aspect]:
            #         if tmp[0] in stop_words:
            #             continue
            #         ans_file.write(",{}".format(tmp[0]))
            #     print(aspect, ans_gradient_PMI[aspect][: 10])
            # ans_file.write("\n")
            ans_file.write("PMI")
            if aspect in ans_PMI:
                for tmp in ans_PMI[aspect]:
                    if tmp[0] in stop_words:
                        continue
                    ans_file.write(",{}".format(tmp[0]))
                print(aspect, ans_PMI[aspect][: 10])
            ans_file.write("\n\n")
    ans_file.close()


'''
calculate the score by gradient
'''


def load_pkl_gradient(domain='res'):
    with open("./results/aspect_aware_gradient_{}_1_final_all_only_gradient.pkl".format(domain), 'rb') as f:
        data = pickle.load(f)
        print(len(data))
        gradient_change_all = {}
        # word_value_gradient = {}
        # aspect_value_gradient = {}
        # aspect_word_value_gradient = {}
        # total_value_gradient = 0.0
        #
        # word_value_PMI = {}
        # aspect_value_PMI = {}
        # aspect_word_value_PMI = {}
        # total_value_PMI = 0.0

        for i in range(len(data)):
            orignal = data[i]['orignal']
            gradient = data[i]['gradient']
            label = orignal['label']
            prob = orignal['prob']
            text = orignal['text']
            aspect = orignal['aspect']
            aspect_index = text.find(' ' + aspect + ' ')
            start_token_index = len(text[:aspect_index].strip().split(" "))
            end_token_index = start_token_index + len(aspect.split(" "))
            text_token = text.split(" ")
            near_words = set(
                text_token[start_token_index - 5: start_token_index] + text_token[end_token_index: end_token_index + 5])

            # if aspect not in aspect_word_value_PMI:
            #     aspect_word_value_PMI[aspect] = {}

            if aspect not in gradient_change_all:
                gradient_change_all[aspect] = []
                # aspect_word_value_gradient[aspect] = {}
            gradient_change = {}
            avg_gradient = 0.0
            max_value = 0
            min_value = 10000
            for tmp in gradient:
                # print(tmp)
                word = tmp[0]
                if word not in near_words:
                    continue
                if word in ['[CLS]', '[SEP]']:
                    continue
                grad = tmp[1]
                if max_value < grad:
                    max_value = grad
                if min_value > grad:
                    min_value = grad
                if word not in gradient_change:
                    gradient_change[word] = grad
                    # gradient_change_all[aspect].append([word, grad])
            avg_gradient /= len(gradient)
            gradient_change = sorted(gradient_change.items(), key=lambda item: item[1], reverse=True)
            # print(text, aspect, gradient_change)
            if max_value == min_value:
                continue
            for tmp in gradient_change:
                # if tmp[1]>avg_gradient*2.5:
                word = tmp[0]
                grad = (tmp[1] - min_value) / (max_value - min_value)
                gradient_change_all[aspect].append([word, grad])

                # word_value_gradient[word] = word_value_gradient.get(word, 0.0) + grad
                # aspect_word_value_gradient[aspect][word] = aspect_word_value_gradient[aspect].get(word, 0.0) + grad
                # aspect_value_gradient[aspect] = aspect_value_gradient.get(aspect, 0.0) + grad
                # total_value_gradient += grad
                # avg_gradient += grad
    del data

    ans_gradient = {}
    gradient = {}
    for aspect in gradient_change_all.keys():
        gradient[aspect] = {}
        word_num = {}
        for tmp in gradient_change_all[aspect]:
            gradient[aspect][tmp[0]] = gradient[aspect].get(tmp[0], 0.0) + tmp[1]
            word_num[tmp[0]] = word_num.get(tmp[0], 0.0) + 1
            # print(aspect, gradient_change_all[aspect][:10])
        for word in gradient[aspect].keys():
            gradient[aspect][word] = gradient[aspect][word] / word_num[word] * (math.log(word_num[word]) + 1)
        gradient_change = sorted(gradient[aspect].items(), key=lambda item: item[1], reverse=True)
        print(aspect, gradient_change[:10])
        ans_gradient[aspect] = gradient_change
    del gradient_change_all
    del gradient
    # Gradient_PMI = {}
    # ans_gradient_PMI = {}
    # for aspect in aspect_word_value_gradient.keys():
    #     Gradient_PMI[aspect] = {}
    #     max_value = 0
    #     min_value = 10000
    #     for word in aspect_word_value_gradient[aspect].keys():
    #         P_O_A = aspect_word_value_gradient[aspect][word] / aspect_value_gradient[aspect]
    #         P_O = word_value_gradient[word] / total_value_gradient
    #         PMI = P_O_A / (P_O + 0.001)
    #         Gradient_PMI[aspect][word] = PMI
    #         if PMI > max_value:
    #             max_value = PMI
    #         if PMI < min_value:
    #             min_value = PMI
    #     if max_value == min_value:
    #         continue
    #     for word in Gradient_PMI[aspect].keys():
    #         Gradient_PMI[aspect][word] = (Gradient_PMI[aspect][word] - min_value) / (max_value - min_value) / 2
    #     gradient_change = sorted(Gradient_PMI[aspect].items(), key=lambda item: item[1], reverse=True)
    #     ans_gradient_PMI[aspect] = gradient_change
    #     # print(aspect, gradient_change[:10])
    # del aspect_word_value_gradient
    # del aspect_value_gradient

    # PMI = {}
    # ans_PMI = {}
    # for aspect in aspect_word_value_PMI.keys():
    #     PMI[aspect] = {}
    #     max_value = 0
    #     min_value = 10000
    #     for word in aspect_word_value_PMI[aspect].keys():
    #         P_O_A = aspect_word_value_PMI[aspect][word] / aspect_value_PMI[aspect]
    #         P_O = word_value_PMI[word] / total_value_PMI
    #         PMI_value = P_O_A / (P_O + 0.001)
    #         PMI[aspect][word] = PMI_value
    #         if PMI_value > max_value:
    #             max_value = PMI_value
    #         if PMI_value < min_value:
    #             min_value = PMI_value
    #     if max_value == min_value:
    #         continue
    #     for word in PMI[aspect].keys():
    #         PMI[aspect][word] = (PMI[aspect][word] - min_value) / (max_value - min_value) / 2
    #     PMI_change = sorted(PMI[aspect].items(), key=lambda item: item[1], reverse=True)
    #     ans_PMI[aspect] = PMI_change
    #     # print(aspect, gradient_change[:10])
    # del aspect_word_value_PMI
    # del aspect_value_PMI
    with open("./results/aspect_aware_gradient_{}_word_list_only_gradient.pkl".format(domain), 'wb') as f:
        pickle.dump(
            {'gradient': ans_gradient}, f)


def normalization(x):
    ans_tmp = {}
    ans = {}
    for aspect in x.keys():
        max_value = 0
        min_value = 1000.0
        ans_tmp[aspect] = {}
        words = x[aspect]
        for i in range(len(words)):
            if words[i][1] > max_value:
                max_value = words[i][1]
            if words[i][1] < min_value:
                min_value = words[i][1]
        for i in range(len(words)):
            ans_tmp[aspect][words[i][0]] = (words[i][1] - min_value) / (max_value - min_value) / 2
        ans[aspect] = sorted(ans_tmp[aspect].items(), key=lambda item: item[1], reverse=True)
    return ans


'''
from cPickle to csv file for both forward and gradient
'''


def load_word_list_forward_gradient(domain='res'):
    stop_words_str = "told###wont###isnt###without###dont###`###+###]###[###@###%###=###/###`###$###&###*###(###)###?###re###able###ask###due###give###etc###tell###also###aaa###won###un###us###can###why###who###then###got###get###ll###but###lo###ki###can###wi###let###ve###his###could###still###about###this###them###so###or###if###would###only###both###been###when###our###as###be###by###he###him###she###her###they###their###your###after###with###there###what###for###at###we###you###is###!###,###,###.###;###:###are###these###those###other###were###on###its###is###was###has###will###my###how###do###does###a###an###am###me###gets###get###the###in###than###it###had###have###from###s###and###since###too###shows###that###to###of###at###itself###from###being###how###what###who###which###where###had###wants###b###c###d###e###f###g###h###i###j###k###l###m###n###o###p###q###r###s###t###u###v###w###x###y###z###-###_###'###\"###[CLS]###[SEP]".split(
        "###")
    stop_words = {}
    for word in stop_words_str:
        stop_words[word] = 1

    ans_file = open("./results/{}_forward_backward_4.csv".format(domain), 'w', encoding='utf-8')
    ans_final = {}
    with open("./results/aspect_aware_gradient_{}_word_list_only_forward.pkl".format(domain), 'rb') as f_forward, open(
            "./results/aspect_aware_gradient_{}_word_list_only_gradient.pkl".format(domain), 'rb') as f_gradient:
        data_forward = pickle.load(f_forward)
        data_gradient = pickle.load(f_gradient)
        ans_gradient = data_gradient['gradient']
        ans_gradient = normalization(ans_gradient)
        # ans_gradient_PMI = data_gradient['gradient_PMI']
        ans_forward = data_forward['forward']
        ans_forward = normalization(ans_forward)
        ans_forward_PMI = data_forward['forward_PMI']
        ans_PMI = data_forward['PMI']

        for aspect in ans_forward.keys():
            ans_file.write("{}\n".format(aspect))
            ans_file.write("Forward_PMI")
            if aspect in ans_forward_PMI:
                word_num = 0
                for tmp in ans_forward_PMI[aspect]:
                    if tmp[0] in stop_words:
                        continue
                    ans_file.write(",{}".format(tmp[0]))
                    word_num += 1
                    if word_num >= 500:
                        break
                print(aspect, ans_forward_PMI[aspect][: 10])
            ans_file.write("\n")
            ans_file.write("Forward")
            if aspect in ans_forward:
                word_num = 0
                for tmp in ans_forward[aspect]:
                    if tmp[0] in stop_words:
                        continue
                    ans_file.write(",{}".format(tmp[0]))
                    word_num += 1
                    if word_num >= 500:
                        break
                print(aspect, ans_forward[aspect][: 10])
            ans_file.write("\n")
            ans_file.write("Backward")
            if aspect in ans_gradient:
                for tmp in ans_gradient[aspect]:
                    if tmp[0] in stop_words:
                        continue
                    ans_file.write(",{}".format(tmp[0]))
                print(aspect, ans_gradient[aspect][: 10])
            ans_file.write("\n")

            ans_file.write("Forward+Backward")
            if aspect in ans_forward and aspect in ans_gradient:
                word_value_gradient = {}
                words = set()
                for tmp in ans_gradient[aspect]:
                    if tmp[0] in stop_words:
                        continue
                    word_value_gradient[tmp[0]] = tmp[1]
                    words.add(tmp[0])

                word_value_forward = {}
                for tmp in ans_forward[aspect]:
                    if tmp[0] in stop_words:
                        continue
                    word_value_forward[tmp[0]] = tmp[1]
                    words.add(tmp[0])
                ans_tmp = {}
                for word in words:
                    ans_tmp[word] = word_value_forward.get(word, 0.0) + word_value_gradient.get(word, 0.0)
                ans_tmp = sorted(ans_tmp.items(), key=lambda item: item[1], reverse=True)
                if aspect not in ans_final:
                    ans_final[aspect] = {}
                for tmp in ans_tmp:
                    ans_final[aspect][tmp[0]] = tmp[1]
                    ans_file.write(",{}".format(tmp[0]))
                print(aspect, ans_tmp[:10])
            ans_file.write("\n")

            # ans_file.write("Backward_PMI")
            # if aspect in ans_gradient_PMI:
            #     for tmp in ans_gradient_PMI[aspect]:
            #         if tmp[0] in stop_words:
            #             continue
            #         ans_file.write(",{}".format(tmp[0]))
            #     print(aspect, ans_gradient_PMI[aspect][: 10])
            # ans_file.write("\n")
            #
            # ans_file.write("Forward_PMI+Backward_PMI")
            # if aspect in ans_forward_PMI and aspect in ans_gradient_PMI:
            #     word_value_gradient = {}
            #     words = set()
            #     for tmp in ans_gradient_PMI[aspect]:
            #         if tmp[0] in stop_words:
            #             continue
            #         word_value_gradient[tmp[0]] = tmp[1]
            #         words.add(tmp[0])
            #
            #     word_value_forward = {}
            #     for tmp in ans_forward_PMI[aspect]:
            #         if tmp[0] in stop_words:
            #             continue
            #         word_value_forward[tmp[0]] = tmp[1]
            #         words.add(tmp[0])
            #     ans_tmp = {}
            #     for word in words:
            #         ans_tmp[word] = word_value_forward.get(word, 0.0) + word_value_gradient.get(word, 0.0)
            #     ans_tmp = sorted(ans_tmp.items(), key=lambda item: item[1], reverse=True)
            #     for tmp in ans_tmp:
            #         ans_file.write(",{}".format(tmp[0]))
            #     print(aspect, ans_tmp[:10])
            # ans_file.write("\n")

            ans_file.write("PMI")
            if aspect in ans_PMI:
                for tmp in ans_PMI[aspect]:
                    if tmp[0] in stop_words:
                        continue
                    ans_file.write(",{}".format(tmp[0]))
                print(aspect, ans_PMI[aspect][: 10])
            ans_file.write("\n\n")
    ans_file.close()
    with open("./results/{}_forward_backward_4_with_score.pkl".format(domain), 'wb') as f:
        pickle.dump(ans_final, f)


def aspect_num(domain='res'):
    with open("./results/aspect_aware_gradient_{}_1_final_all_only_gradient.pkl".format(domain), 'rb') as f:
        data = pickle.load(f)
        ans = {}
        for i in range(len(data)):
            orignal = data[i]['orignal']
            aspect = orignal['aspect']
            ans[aspect] = ans.get(aspect, 0) + 1

    with open("./results/{}_aspect_num.pkl".format(domain), 'wb') as f:
        pickle.dump(ans, f)


if __name__ == '__main__':
    # domain = 'lap'
    # get_top_k(domain=domain, method='ml')
    # domain = 'res'
    # get_top_k(domain=domain, method='ml')
    # read_csv(domain='lap')
    # read_csv(domain='res')

    # load_pkl(domain='res')
    # load_pkl(domain='lap')
    # load_pkl_gradient(domain='res')
    # load_pkl_gradient(domain='lap')

    load_word_list(domain='res')
    load_word_list(domain='lap')

    # load_word_list_forward_gradient(domain='res')
    # load_word_list_forward_gradient(domain='lap')

    # aspect_num(domain='res')
    # aspect_num(domain='lap')
