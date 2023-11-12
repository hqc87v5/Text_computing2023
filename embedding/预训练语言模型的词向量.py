#利用预训练词向量得到句向量
#1.读入词向量
import numpy as np
count=0
word_embedding={}
# path='word_embedding_yelp.txt'
path='../glove/glove.6B.300d.txt'
with open(path,'r',encoding='utf-8') as f_word_emb:
    for line in f_word_emb:
        line=line.split()
        word_embedding[line[0]]=0
        word_embedding[line[0]]=[float(i) for i in line[1:]]
#2.读入句子，词向量转换为句向量
#2.1 emb_300d_train.0
f_w=open('emb_300d_train.0','w',encoding='utf-8')
with open('../yelp/sentiment.train.0','r',encoding='utf-8') as f_r:
    for line in f_r:
        words=line.split()
        words_emb=[]
        for w in words:
            try:
                #word_embedding 是已有的词向量转化的字典
                words_emb.append(word_embedding[w])#一句话中的所有单词emb
            except:
                continue
        words_emb=np.array(words_emb)
        sent_emb=np.mean(words_emb,axis=0)
        try:
            f_w.write(' '.join([str(i) for i in list(sent_emb)])+'\n')
        except:
            continue
f_w.close()

#2.2 emb_300d_train.1
f_w=open('emb_300d_train.1','w',encoding='utf-8')
with open('../yelp/sentiment.train.1','r',encoding='utf-8') as f_r:
    for line in f_r:
        words=line.split()
        words_emb=[]
        for w in words:
            try:
                words_emb.append(word_embedding[w])
            except:
                continue
        words_emb=np.array(words_emb)
        sent_emb=np.mean(words_emb,axis=0)
        try:
            f_w.write(' '.join([str(i) for i in list(sent_emb)])+'\n')
        except:
            continue
f_w.close()

#2.3 emb_300d_test.0
f_w=open('emb_300d_test.0','w',encoding='utf-8')
with open('../yelp/sentiment.test.0','r',encoding='utf-8') as f_r:
    for line in f_r:
        words=line.split()
        words_emb=[]
        for w in words:
            try:
                words_emb.append(word_embedding[w])
            except:
                continue
        words_emb=np.array(words_emb)
        sent_emb=np.mean(words_emb,axis=0)
        try:
            f_w.write(' '.join([str(i) for i in list(sent_emb)])+'\n')
        except:
            continue
f_w.close()

#2.4 emb_300d_test.1
f_w=open('emb_300d_test.1','w',encoding='utf-8')
with open('../yelp/sentiment.test.1','r',encoding='utf-8') as f_r:
    for line in f_r:
        words=line.split()
        words_emb=[]
        for w in words:
            try:
                words_emb.append(word_embedding[w])
            except:
                continue
        words_emb=np.array(words_emb)
        sent_emb=np.mean(words_emb,axis=0)
        try:
            f_w.write(' '.join([str(i) for i in list(sent_emb)])+'\n')
        except:
            continue
f_w.close()

#3.利用句向量进行逻辑回归（同one-hot方式一样）
X_Train=[]
Y_Train=[]
count=0
with open('emb_300d_train.0','r',encoding='utf-8') as f_train0:
    for line in f_train0:

        if count==20000:
            break
        count+=1
        X_Train.append([float(i) for i in line.strip().split()])
        Y_Train.append(0)
count=0
with open('emb_300d_train.1','r',encoding='utf-8') as f_train1:
    for line in f_train1:

        if count==20000:
            break
        count+=1
        X_Train.append([float(i) for i in line.strip().split()])
        Y_Train.append(1)

X_Test=[] #句向量
Y_Test=[] #标签

count=0
with open('emb_300d_test.0','r',encoding='utf-8') as f_test0:
    for line in f_test0:
        if count==2000:
            break
        count+=1

        X_Test.append([float(i) for i in line.strip().split()])
        Y_Test.append(0)

count=0
with open('emb_300d_test.1','r',encoding='utf-8') as f_test1:
    for line in f_test1:
        if count==2000:
            break
        count+=1

        X_Test.append([float(i) for i in line.strip().split()])
        Y_Test.append(1)

#4.逻辑回归
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_Train,Y_Train)


#5.测试结果
Y_Pred=classifier.predict(X_Test)
print('train samples counts:',len(X_Train))
print('test samples counts:',len(X_Test))
c=0
for pred,truth in zip(Y_Pred,Y_Test):
    if pred==truth:
        c+=1
print('Accuracy:',c/len(Y_Test))
