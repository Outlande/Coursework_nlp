import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import jieba
from tqdm import tqdm, trange

class seq_net(nn.Module):
    def __init__(self, onehot_num):
        super(seq_net, self).__init__()
        onehot_size = onehot_num
        embedding_size = 256
        n_layer = 2
        # input [seq_length, batch, embedding_size] and output [seq_length, batch, embedding_size]
        self.lstm = nn.LSTM(embedding_size, embedding_size, n_layer, batch_first=True)
        self.encode = nn.Linear(onehot_size, embedding_size)
        # self.decode =torch.nn.Sequential(
        #     nn.Linear(embedding_size, 1024),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.ReLU(),
        #     nn.Linear(1024, 2048),
        #     torch.nn.Dropout(0.5),
        #     torch.nn.ReLU(),
        #     nn.Linear(2048, onehot_size),
        #     torch.nn.Softmax()
        # )
        self.decode =torch.nn.Sequential(
            nn.Linear(embedding_size, onehot_size),
            torch.nn.Sigmoid()
        )


    def forward(self, x):
        # input [seq_length, onehot_size]
        em = self.encode(x).unsqueeze(dim=1)
        out, _ = self.lstm(em)
        res = self.decode(out[:,0,:])
        return res


def read_data():
    datasets_root = "../datasets"
    catalog = "test.txt"
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
                line = re.sub('！','。', line)
                line = re.sub('？','。', line)
                # u3002是句号
                line = re.sub('[\u0000-\u3001]','', line)
                line = re.sub('[\u3003-\u4DFF]','', line)
                line = re.sub('[\u9FA6-\uFFFF]','', line)
                all_text += line
            all_texts[name] = all_text

    return all_texts


def train():
    print("start read data")
    all_texts = read_data()
    all_terms_list = dict()
    all_terms_dict = dict()
    all_terms_number_dict = dict()
    for name in list(all_texts.keys()):
        text = all_texts[name]
        text_terms = list()
        text_terms_dict = dict()
        text_terms_name_dict = dict()
        encode_num = 0
        for text_line in text.split('。'):
            seg_list = list(jieba.cut(text_line, cut_all=False)) # 使用精确模式
            if len(seg_list) < 5:
                continue
            seg_list.append("END")
            # add list
            text_terms.append(seg_list)

            # add term dict
            for term in seg_list:
                if not term in text_terms_dict:
                    text_terms_dict[term] = encode_num
                    text_terms_name_dict[term] = 1
                    encode_num = encode_num + 1
                else:
                    text_terms_name_dict[term] = text_terms_name_dict[term]+1
        all_terms_list[name] = text_terms
        all_terms_dict[name] = text_terms_dict
        all_terms_number_dict[name] = text_terms_name_dict
    print("end read data")


    # get_onehot_embedding
    print("start calculate embedding one hot")
    onehot_embedding_text = dict()
    for text in list(all_terms_dict.keys()):
        onehot_num = len(all_terms_dict[text])
        text_term_dict = all_terms_dict[text]
        onehot_em = dict()
        for term in list(text_term_dict.keys()):
            a = torch.zeros(onehot_num)
            a[text_term_dict[term]] = 1
            onehot_em[term] = a
        onehot_embedding_text[text] = onehot_em
    print("finish calculate embedding one hot")

    print("end get sequence emnedding")
    epochs = 1
    end_num = 10
    for name in list(all_terms_list.keys()):
        print("start train ", name)
        onehot_embedding = onehot_embedding_text[name]
        embed_size = len(onehot_embedding)
        sequences = all_terms_list[name]

        model = seq_net(embed_size).cuda()
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=0.0001)
        for epoch_id in range(epochs):
            for idx in trange(0, len(sequences)//end_num - 1):
                seq = []
                for k in range(end_num):
                    seq += sequences[idx+k]

                target = []
                for k in range(end_num):
                    target += sequences[idx+end_num+k]

                input_seq = torch.zeros(len(seq), embed_size)
                for idy in range(len(seq)):
                    input_seq[idy] = onehot_embedding[seq[idy]]

                # optimizer.zero_grad()
                # out_res = model(input_seq[:-1].cuda())
                # loss = F.binary_cross_entropy(out_res, input_seq[1:].cuda())
                # loss.backward()
                # optimizer.step()
                # if idx % 50 == 0:
                #     print("loss: ", loss.item(), " in epoch ", epoch_id, " res: ", out_res.max(dim=1).indices, input_seq[1:].max(dim=1).indices)

        state = {"model" : model.state_dict(), "terms: ": all_terms_dict[name]}
        torch.save(state, name+str(epoch_id)+".pth")


# def test():








if __name__ == "__main__":
    # execute only if run as a script
    train()
    # test()