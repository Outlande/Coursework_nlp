import os
import jieba
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def read_data():
    datasets_root = "../datasets"
    catalog = "inf.txt"
    with open(os.path.join(datasets_root, catalog), "r") as f:
        all_files = f.readline().split(",")
        print(all_files)

    all_texts = dict()
    for name in all_files:
        with open(os.path.join(datasets_root, name+".txt"), "r") as f:
            file_read = f.readlines()
            all_text = ""
            for line in file_read:
                line = re.sub('\s','', line)
                line = re.sub('[\u0000-\u4DFF]',',', line)
                line = re.sub('[\u9FA6-\uFFFF]',',', line)
                all_text += line
            all_texts[name] = all_text

    return all_texts

def train():
    all_texts = read_data()

    all_terms = dict()
    for name in list(all_texts.keys()):
        text = all_texts[name]
        text_terms = list()
        for text_line in text.split(','):
            seg_list = list(jieba.cut(text_line, cut_all=False)) # 使用精确模式
            if len(seg_list) == 0:
                continue
            text_terms.append(seg_list)

        all_terms[name] = text_terms
        print("finished to calculate the ", name, " text sentence")

    text_models = dict()
    for name in list(all_terms.keys()):
        print("Start to build ", name, " model")

        # sg=0 cbow, sg=1 skip-gram
        # size is the dim of feature
        model = Word2Vec(sentences=all_terms[name], sg=0, vector_size=50, min_count=10, window=10, epochs=100)

        print("Finish to build ", name, " model")
        text_models[name] = model
        model.save('models/'+name+'.model')

def test_people():
    text_name  = '射雕英雄传'
    model = Word2Vec.load('models/'+text_name+'.model')

    test_names = ['郭靖', '黄蓉', '杨康', '黄药师', '洪七公', '周伯通']

    print('\n', '\n', '\n', text_name+":", '\n', '\n')
    for name in test_names:
        peoples = model.wv.most_similar(positive=name, topn=20)
        print(name+': ')
        for people in peoples:
            print(people)
        print('\n')


    text_name  = '天龙八部'
    model = Word2Vec.load('models/'+text_name+'.model')

    test_names = ['萧峰', '乔峰', '段誉', '虚竹', '慕容复', '王语嫣']

    print('\n', '\n', '\n', text_name+":", '\n', '\n')
    for name in test_names:
        peoples = model.wv.most_similar(positive=name, topn=20)
        print(name+': ')
        for people in peoples:
            print(people)
        print('\n')


    text_name  = '倚天屠龙记'
    model = Word2Vec.load('models/'+text_name+'.model')

    test_names = ['张无忌','赵敏','周芷若', '小昭', '谢逊', '宋青书']

    print('\n', '\n', '\n', text_name+":", '\n', '\n')
    for name in test_names:
        peoples = model.wv.most_similar(positive=name, topn=20)
        print(name+': ')
        for people in peoples:
            print(people)
        print('\n')


def test_graph():
    text_name  = '射雕英雄传'
    model = Word2Vec.load('models/'+text_name+'.model')

    test_names = ['郭靖', '黄蓉', '杨康', '黄药师', '洪七公', '周伯通', '欧阳锋', '一灯大师', '王重阳', '穆念慈', '裘千仞', '梅超风', '柯镇恶', '华筝', '铁木真']

    results = np.zeros((0, 50))

    for name in test_names:
        feature = model.wv[name]
        feature = (feature/np.linalg.norm(feature))[np.newaxis, :]
        results = np.concatenate((results,feature), axis=0)

    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(results)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], marker = 'o', color = 'green', s = 40)

    for idx in range(len(test_names)):
        plt.annotate(test_names[idx], xy = (principalComponents[idx, 0], principalComponents[idx, 1]),\
                     xytext = (principalComponents[idx, 0]+0.01, principalComponents[idx, 1]+0.01))
    plt.show()



    text_name  = '天龙八部'
    model = Word2Vec.load('models/'+text_name+'.model')

    test_names = ['萧峰', '乔峰', '段誉', '虚竹', '慕容复', '王语嫣', '阿朱', '阿紫', '段正淳', '鸠摩智', '天山童姥', '李秋水', '游坦之', '慕容博', '萧远山']

    results = np.zeros((0, 50))

    for name in test_names:
        feature = model.wv[name]
        feature = (feature/np.linalg.norm(feature))[np.newaxis, :]
        results = np.concatenate((results,feature), axis=0)

    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(results)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], marker = 'o', color = 'green', s = 40)

    for idx in range(len(test_names)):
        plt.annotate(test_names[idx], xy = (principalComponents[idx, 0], principalComponents[idx, 1]),\
                     xytext = (principalComponents[idx, 0]+0.01, principalComponents[idx, 1]+0.01))
    plt.show()



    text_name  = '倚天屠龙记'
    model = Word2Vec.load('models/'+text_name+'.model')

    test_names = ['赵敏', '周芷若', '殷素素', '纪晓芙', '张翠山', '灭绝师太', '殷离', '俞莲舟', '宋青书', '殷梨亭', '谢逊', '张无忌', '范遥', '黛绮丝', '小昭']

    results = np.zeros((0, 50))

    for name in test_names:
        feature = model.wv[name]
        feature = (feature/np.linalg.norm(feature))[np.newaxis, :]
        results = np.concatenate((results,feature), axis=0)

    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(results)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], marker = 'o', color = 'green', s = 40)

    for idx in range(len(test_names)):
        plt.annotate(test_names[idx], xy = (principalComponents[idx, 0], principalComponents[idx, 1]),\
                     xytext = (principalComponents[idx, 0]+0.01, principalComponents[idx, 1]+0.01))
    plt.show()


def test_content():
    text_name  = '射雕英雄传'
    model = Word2Vec.load('models/'+text_name+'.model')

    test_names = ['降龙十八掌', '打狗棒法', '蛤蟆功', '灵蛇', '一阳指', '九阴','双手','空明拳']

    results = np.zeros((0, 50))

    for name in test_names:
        feature = model.wv[name]
        feature = (feature/np.linalg.norm(feature))[np.newaxis, :]
        results = np.concatenate((results,feature), axis=0)

    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(results)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], marker = 'o', color = 'green', s = 40)

    for idx in range(len(test_names)):
        plt.annotate(test_names[idx], xy = (principalComponents[idx, 0], principalComponents[idx, 1]),\
                     xytext = (principalComponents[idx, 0]+0.01, principalComponents[idx, 1]+0.01))
    plt.show()



    text_name  = '天龙八部'
    model = Word2Vec.load('models/'+text_name+'.model')

    test_names = ['少林', '天龙', '逍遥', '天山', '星宿', '姑苏', '丐帮']
    results = np.zeros((0, 50))

    for name in test_names:
        feature = model.wv[name]
        feature = (feature/np.linalg.norm(feature))[np.newaxis, :]
        results = np.concatenate((results,feature), axis=0)

    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(results)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], marker = 'o', color = 'green', s = 40)

    for idx in range(len(test_names)):
        plt.annotate(test_names[idx], xy = (principalComponents[idx, 0], principalComponents[idx, 1]),\
                     xytext = (principalComponents[idx, 0]+0.01, principalComponents[idx, 1]+0.01))
    plt.show()



    text_name  = '倚天屠龙记'
    model = Word2Vec.load('models/'+text_name+'.model')

    test_names =  ['少林','武当','峨嵋', '华山', '昆仑', '崆峒', '明教', '波斯', '天鹰', '西域', '金刚', '巨鲸帮',  '海沙', '神拳门']

    results = np.zeros((0, 50))

    for name in test_names:
        feature = model.wv[name]
        feature = (feature/np.linalg.norm(feature))[np.newaxis, :]
        results = np.concatenate((results,feature), axis=0)

    pca = PCA(n_components = 2)
    principalComponents = pca.fit_transform(results)
    plt.scatter(principalComponents[:, 0], principalComponents[:, 1], marker = 'o', color = 'green', s = 40)

    for idx in range(len(test_names)):
        plt.annotate(test_names[idx], xy = (principalComponents[idx, 0], principalComponents[idx, 1]),\
                     xytext = (principalComponents[idx, 0]+0.01, principalComponents[idx, 1]+0.01))
    plt.show()


def test_social():
    text_names = ['碧血剑', '飞狐外传', '连城诀', '鹿鼎记', \
                '射雕英雄传', '神雕侠侣', '书剑恩仇录', '天龙八部', '侠客行', '笑傲江湖', \
                '雪山飞狐', '倚天屠龙记', '鸳鸯刀']

    for text_name in text_names:
        model = Word2Vec.load('models/'+text_name+'.model')

        social_name = '江湖'
        print('\n', '\n', '\n', text_name+":", '\n', '\n')
        peoples = model.wv.most_similar(positive=social_name, topn=30)
        print(social_name+': ')
        for people in peoples:
            print(people)
        print('\n')

if __name__ == "__main__":
    # execute only if run as a script
    train()
    print("4.1: ")
    test_people()
    print("4.2: ")
    test_graph()
    print("4.3: ")
    test_content()
    print("4.4: ")
    test_social()