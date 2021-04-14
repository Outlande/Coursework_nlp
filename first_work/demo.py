import os
import jieba
import glob
import re
import numpy as np
import math

def read_data():
    datasets_root = "./datasets"
    catalog = "inf.txt"
    with open(os.path.join(datasets_root, catalog), "r") as f:
        all_files = f.readline().split(",")
        print(all_files)

    all_texts = []
    for name in all_files:
        with open(os.path.join(datasets_root, name+".txt"), "r") as f:
            file_read = f.readlines()
            all_text = ""
            for line in file_read:
                line = re.sub('\s','', line)
                line = re.sub('[\u0000-\u4DFF]',",", line)
                line = re.sub('[\u9FA6-\uFFFF]',",", line)
                all_text += line
            all_texts.append(all_text)

    return all_texts

def main():
    all_texts = read_data()
    all_terms = []
    term_count = dict()
    term_count_two = dict()
    term_count_three = dict()

    count = 1
    for text in all_texts:
        for text_line in text.split(','):
            seg_list = jieba.cut(text_line, cut_all=False) # 使用精确模式
            all_terms.append(list(seg_list))
        print("finished to calculate the ", count, " text")
        count += 1

    count_words = 0
    count_terms = 0
    for all_term in all_terms:
        term_two = []
        term_three = []
        count_number = 0
        for term in all_term:
            count_words += len(term)
            count_terms += 1
            count_number += 1
            if count_number > 3:
                del term_two[0]
                term_two.append(term)
                del term_three[0]
                term_three.append(term)

                if not term in term_count:
                    term_count[term] = 0

                if not term_two[0]+','+term_two[1] in term_count_two:
                    term_count_two[term_two[0]+','+term_two[1]] = 0

                if not term_three[0]+','+term_three[1]+','+term_three[2] in term_count_three:
                    term_count_three[term_three[0]+','+term_three[1]+','+term_three[2]] = 0

                term_count[term] += 1
                term_count_two[term_two[0]+','+term_two[1]] += 1
                term_count_three[term_three[0]+','+term_three[1]+','+term_three[2]] += 1
            elif count_number == 3:
                del term_two[0]
                term_two.append(term)
                term_three.append(term)
                if not term in term_count:
                    term_count[term] = 0
                if not term_two[0]+','+term_two[1] in term_count_two:
                    term_count_two[term_two[0]+','+term_two[1]] = 0
                if not term_three[0]+','+term_three[1]+','+term_three[2] in term_count_three:
                    term_count_three[term_three[0]+','+term_three[1]+','+term_three[2]] = 0

                term_count[term] += 1
                term_count_two[term_two[0]+','+term_two[1]] += 1
                term_count_three[term_three[0]+','+term_three[1]+','+term_three[2]] += 1
            elif count_number == 2:
                term_two.append(term)
                term_three.append(term)

                if not term in term_count:
                    term_count[term] = 0
                if not term_two[0]+','+term_two[1] in term_count_two:
                    term_count_two[term_two[0]+','+term_two[1]] = 0

                term_count[term] += 1
                term_count_two[term_two[0]+','+term_two[1]] += 1
            else:
                if not term in term_count:
                    term_count[term] = 0
                term_two.append(term)
                term_three.append(term)
                term_count[term] += 1

    print("汉字 数量: ", count_words)
    print("词语 数量: ", count_terms)
    print("词语的平均字数:", count_words/count_terms)

    print("一元组 类型数: ", len(term_count))
    print("二元组 类型数: ", len(term_count_two))
    print("三元组 类型数: ", len(term_count_three))

    print("一元组 总数: ", sum(term_count.values()))
    print("二元组 总数: ", sum(term_count_two.values()))
    print("三元组 总数: ", sum(term_count_three.values()))

    indice_one = np.argsort(-np.array(list(term_count.values())))
    indice_two = np.argsort(-np.array(list(term_count_two.values())))
    indice_three = np.argsort(-np.array(list(term_count_three.values())))


    print("一元组频数 top 10")
    for i in range(10):
        print(list(term_count.keys())[indice_one[i]], " has ", list(term_count.values())[indice_one[i]])

    print("二元组频数 top 10")
    for i in range(10):
        print(list(term_count_two.keys())[indice_two[i]], " has ", list(term_count_two.values())[indice_two[i]])

    print("三元组频数 top 10")
    for i in range(10):
        print(list(term_count_three.keys())[indice_three[i]], " has ", list(term_count_three.values())[indice_three[i]])

    sum_number_one = sum(term_count.values())
    entropy_one = 0
    for item in term_count:
        entropy_one -= term_count[item]/sum_number_one * math.log(term_count[item]/sum_number_one)/math.log(2)
    print("Entropy of 一元组: ", entropy_one)

    sum_number_two = sum(term_count_two.values())
    entropy_two = 0
    for item in term_count_two:
        entropy_two -= term_count_two[item]/sum_number_two * math.log(term_count_two[item]/term_count[item.split(',')[0]])/math.log(2)
    print("Entropy of 二元组: ", entropy_two)

    sum_number_three = sum(term_count_three.values())
    entropy_three = 0
    for item in term_count_three:
        entropy_three -= term_count_three[item]/sum_number_three * math.log(term_count_three[item] /(term_count_two[item.split(',')[0]+','+item.split(',')[1]]))/math.log(2)
    print("Entropy of 三元组: ", entropy_three)



if __name__ == "__main__":
    # execute only if run as a script
    main()
