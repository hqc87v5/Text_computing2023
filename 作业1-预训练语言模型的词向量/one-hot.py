#基于one-hot的文本表示进行分类
#1.构造one-hot词表

def make_vocab(train_path,vocab_path,freq_thres):#词表构建
    vocab={}
    with open(train_path,encoding='utf-8') as fr:
        for line in fr:
            line=line.strip().split(' ') #通过空格来进行分词
            for word in line:
                if word not in vocab:
                    vocab[word]=1
                vocab[word]+=1
    vocab_sort=sorted(vocab.items(),key=lambda x:x[1],reverse=True) #按词频由大到小顺序排列
    vocab_sort=[x for x in vocab_sort if x[1]>freq_thres]
    #freq_thres：至少出现多少次，才会放进词表里

    with open(vocab_path,'w',encoding='utf-8') as fw:
        #读入词表
        for word,freq in vocab_sort:
            fw.write("{}\t{}\n".format(word,freq))

    return vocab_sort

if __name__=='__main__':
    vocab=make_vocab('yelp_data.txt','onehot_vocab.txt',20)
    #print(vocab)


#2.构造one-hot词向量和句向量

import numpy as np
word2id={}
with open ('onehot_vocab.txt','r',encoding='utf-8') as f_vocab: #读入词表
    count=0 #key是词，value是索引（词表中的位置）
    for line in f_vocab:
        word=line.split('\t')[0] #单词 \t 频次
        word2id[word] = count #给每个词编上号
        count+=1
#print(word2id)
vocab_len=len(word2id)

#2.1 emb_train.0
f_w=open('emb_train.0','w',encoding='utf-8')
with open('Yelp数据集/sentiment.train.0','r',encoding='utf-8') as f_r:
    for line in f_r:
        words =line.split()
        words_emb=[]
        for w in words:#遍历句子中每个词
            if w in word2id.keys(): #判断词是否在词表中
                w_emb=np.zeros([vocab_len],dtype=int) #设置全零向量
                w_emb[word2id[w]]=1 #出现的单词，将会在emb的对应位置编号上记1
                words_emb.append(w_emb) #数组里是：每一句话的所有单词的emb向量
            else:
                continue
        words_emb=np.array(words_emb) #变成一个矩阵
        sent_emb=np.sum(words_emb,axis=0) #句向量：按第0维加和，也就是把每条句子对应位置加和
        try:
            f_w.write(' '.join([str(i) for i in list(sent_emb)])+'\n')
        except:
            continue
f_w.close()
f_r.close()

#2.2 emb_train.1
f_w=open('emb_train.1','w',encoding='utf-8')
#都一直，就是情感标签不一样
with open('Yelp数据集/sentiment.train.1','r',encoding='utf-8') as f_r:
    for line in f_r:
        words =line.split()
        words_emb=[]
        for w in words:#遍历句子中每个词
            if w in word2id.keys(): #判断词是否在词表中
                w_emb=np.zeros([vocab_len],dtype=int) #设置全零向量
                w_emb[word2id[w]]=1
                words_emb.append(w_emb)
            else:
                continue
        words_emb=np.array(words_emb)
        sent_emb=np.sum(words_emb,axis=0)
        try:
            f_w.write(' '.join([str(i) for i in list(sent_emb)])+'\n')
        except:
            continue
f_w.close()
f_r.close()

#2.3 emb_test.0
f_w=open('emb_test.0','w',encoding='utf-8')
with open('Yelp数据集/sentiment.test.0','r',encoding='utf-8') as f_r:
    for line in f_r:
        words =line.split()
        words_emb=[]
        for w in words:#遍历句子中每个词
            if w in word2id.keys(): #判断词是否在词表中
                w_emb=np.zeros([vocab_len],dtype=int) #设置全零向量
                w_emb[word2id[w]]=1
                words_emb.append(w_emb)
            else:
                continue
        words_emb=np.array(words_emb)
        sent_emb=np.sum(words_emb,axis=0)
        try:
            f_w.write(' '.join([str(i) for i in list(sent_emb)])+'\n')
        except:
            continue
f_w.close()
f_r.close()

#2.4 emb_test.1
f_w=open('emb_test.1','w',encoding='utf-8')
with open('Yelp数据集/sentiment.test.1','r',encoding='utf-8') as f_r:
    for line in f_r:
        words =line.split()
        words_emb=[]
        for w in words:#遍历句子中每个词
            if w in word2id.keys(): #判断词是否在词表中
                w_emb=np.zeros([vocab_len],dtype=int) #设置全零向量
                w_emb[word2id[w]]=1
                words_emb.append(w_emb)
            else:
                continue
        words_emb=np.array(words_emb)
        sent_emb=np.sum(words_emb,axis=0)
        try:
            f_w.write(' '.join([str(i) for i in list(sent_emb)])+'\n')
        except:
            continue
f_w.close()
f_r.close()


#3.构建训练集和测试集
X_Train=[] #句向量
Y_Train=[] #标签
count=0

with open('emb_train.0','r',encoding='utf-8') as f_train0:
    for line in f_train0:
        if count==50000: #正负例各要5000
            break
        count+=1

        X_Train.append([float(i) for i in line.strip().split()])
        Y_Train.append(0)
count=0

with open('emb_train.1','r',encoding='utf-8') as f_train1:
    for line in f_train1:
        if count==50000:
            break
        count+=1

        X_Train.append([float(i) for i in line.strip().split()])
        Y_Train.append(1)

X_Test=[] #句向量
Y_Test=[] #标签

count=0
with open('emb_test.0','r',encoding='utf-8') as f_train0:
    for line in f_train0:
        if count==5000:
            break
        count+=1

        X_Test.append([float(i) for i in line.strip().split()])
        Y_Test.append(0)

count=0
with open('emb_test.1','r',encoding='utf-8') as f_train1:
    for line in f_train1:
        if count==5000:
            break
        count+=1

        X_Test.append([float(i) for i in line.strip().split()])
        Y_Test.append(1)

#4.利用one-hot的句向量进行逻辑回归

from sklearn.linear_model import LogisticRegression
#实例化了一个类
classifier = LogisticRegression(random_state=0)
classifier.fit(X_Train,Y_Train)

#5.测试结果
Y_Pred=classifier.predict(X_Test) #预测结果
print('train samples counts:',len(X_Train))
print('test samples counts:',len(X_Test))
c=0
#把对应位置的结果zip起来
for pred,truth in zip(Y_Pred,Y_Test): #每一位进行比对
    if pred==truth:
        c+=1 #判断正确的样本个数
print('Accuracy:',c/len(Y_Test))
