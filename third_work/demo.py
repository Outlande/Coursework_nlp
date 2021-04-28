import os
import jieba
import glob
import re
import numpy as np
import math
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def read_data():
    datasets_root = "../datasets"
    catalog = "inf.txt"
    with open(os.path.join(datasets_root, catalog), "r") as f:
        all_files = f.readline().split(",")
        print(all_files)

    train_files_dict = dict()
    test_files_dict = dict()
    # test 200行
    # train 50000行
    test_num = 10
    test_length = 20
    for name in all_files:
        with open(os.path.join(datasets_root, name+".txt"), "r") as f:
            file_read = f.readlines()
            train_num = len(file_read)-test_num

            choice_index = np.random.choice(len(file_read), test_num+train_num, replace=False)

            train_text = ""
            for train in choice_index[0:train_num]:
                line = file_read[train]
                line = re.sub('\s','', line)
                line = re.sub('[\u0000-\u4DFF]','', line)
                line = re.sub('[\u9FA6-\uFFFF]','', line)
                if len(line) == 0:
                    continue
                train_text += line

            train_files_dict[name] = train_text

            for test in choice_index[train_num:test_num+train_num]:
                if test + test_length >= len(file_read):
                    continue
                test_line = ""
                for i in range(test, test+test_length):
                    line = file_read[i]
                    line = re.sub('\s','', line)
                    line = re.sub('[\u0000-\u4DFF]','', line)
                    line = re.sub('[\u9FA6-\uFFFF]','', line)
                    test_line += line
                if not name in test_files_dict.keys():
                    test_files_dict[name] = [test_line]
                else:
                    test_files_dict[name].append(test_line)
    return train_files_dict, test_files_dict

def main():
    train_texts_dict, test_texts_dict = read_data()

    train_terms_list = []
    train_terms_dict = dict()
    name_list = []
    for name in train_texts_dict.keys():
        text = train_texts_dict[name]
        seg_list = list(jieba.cut(text, cut_all=False)) # 使用精确模式

        terms_string = ""
        for term in seg_list:
            terms_string += term+" "
        train_terms_dict[name] = terms_string
        train_terms_list.append(terms_string)
        name_list.append(name)
        print("finished to calculate the train "+name+" text")

    test_terms_dict = dict()
    for name in test_texts_dict.keys():
        text_list = test_texts_dict[name]
        for text in text_list:
            seg_list = list(jieba.cut(text, cut_all=False)) # 使用精确模式
            terms_string = ""
            for term in seg_list:
                terms_string += term+" "
            if not name in test_terms_dict.keys():
                test_terms_dict[name] = [terms_string]
            else:
                test_terms_dict[name].append(terms_string)
        print("finished to calculate the test "+name+" text")

    # calculate terms vector
    cnt_vector = CountVectorizer(max_features=500)
    cnt_tf_train = cnt_vector.fit_transform(train_terms_list)

    lda = LatentDirichletAllocation(n_components=35,  # 主题个数
                                    max_iter=3000,    # EM算法的最大迭代次数
                                    # learning_method='online',
                                    # learning_offset=50.,  # 仅仅在算法使用online时有意义，取值要大于1。用来减小前面训练样本批次对最终模型的影响
                                    random_state=0)
    target = lda.fit_transform(cnt_tf_train)
    print("target: ", target.shape)

    files_feature = dict()
    correct_number = 0
    wrong_number = 0
    for name in test_terms_dict.keys():
        terms_list = test_terms_dict[name]
        for terms in terms_list:
            cnt_tf_file = cnt_vector.transform([terms])

            res = lda.transform(cnt_tf_file)
            min_index = ((target - res.repeat(target.shape[0], axis=0))**2).sum(axis=1).argmin()

            if name == name_list[min_index]:
                correct_number += 1
            else:
                wrong_number += 1

    print("accuracy: ", correct_number/(correct_number+wrong_number))





if __name__ == "__main__":
    # execute only if run as a script
    main()
