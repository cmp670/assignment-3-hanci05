
# coding: utf-8

import random
import math
import nltk
import dynet as dy
import numpy as np
from scipy.sparse import dok_matrix
#%% reading data

file_name = 'trumpspeeches.txt'
data = open(file_name, 'r',encoding='utf8').read().lower()

tokens = nltk.sent_tokenize(data)
word_list = set()


bigram_count = 0



for i in range(len(tokens)):
    tokens[i] = nltk.word_tokenize(tokens[i])
    
    for word in tokens[i]:
        word_list.add(word)
    
    bigram_count=bigram_count+ len(tokens[i]) + 1
    
w_count = len(word_list) + 2
word2index = dict()









#%% mode, weigths and biases
class hidden:

    def __init__(self, d_input, d_output, d_hidden, learning_rate=0.1):
        
        self.model = dy.Model()
        self.sgd = dy.MomentumSGDTrainer(self.model , learning_rate=learning_rate)
        self.w1 = self.model .add_parameters((d_hidden, d_input))
        self.w2 = self.model .add_parameters((d_output, d_hidden))
        self.bias = self.model .add_parameters((d_hidden, ))
   
   
    
    
    def calc1(self, x):
        x = dy.inputVector(x)
        h = dy.rectify(self.w1 * x + self.bias)
        logits = self.w2 * h
    
 
        temp = np.exp(logits.npvalue())
        p_list = temp / np.sum(temp)
        return p_list
    
    
    
    
    def cal2(tempWord,wordVector):
        curVector = []
        for i in range(0,len(wordVector)):
            if wordVector[i] == tempWord:
                curVector.append(1)
                
            else:
                curVector.append(0)
        return curVector
    
   
    
    
    
    def calc3(self, x_train, y_train):
        
        loss_list = []
        for inp, out in zip(x_train, y_train):
            inp = dy.inputVector(inp)
            h = dy.rectify(self.w1 * inp + self.bias)
            logits = self.w2 * h

            loss = dy.pickneglogsoftmax(logits, out)
            loss_list.append(loss)
        
        
        

   
    
#%%

x_train = dok_matrix((bigram_count, w_count), dtype=int)
y_train = [-1 for _ in range(bigram_count)]

index_lst = list(range(bigram_count))
random.shuffle(index_lst)

def add(w):
    
    if w not in word2index:
        word2index[w] = len(word2index)
    
    return word2index[w]
c = 0
for i in tokens:
    pre_word = 0 
    for word in i:
        c_w = add(word)
        index = index_lst[c]
        c=c+1
        x_train[index, pre_word] = 1
        y_train[index] =  c_w
    
        pre_word =  c_w
        
    index = index_lst[c]
    c=c+1
    x_train[index, pre_word] = 1
    y_train[index] =  c_w


x_train = x_train.tocsr()
model  = hidden(d_input=w_count,d_output=w_count, d_hidden=2048,  learning_rate=0.1)




r = math.ceil(bigram_count *0.01)

for i in range(1000):
    
    start = (i%r) * 10
    
    if i == r - 1:
        finish = bigram_count
    else:
        finish = ((i%r) + 1) * 10
    
    inp = np.array(x_train[start:finish].todense())
    model .calc3(inp, y_train[start:finish])

word_index = [None for _ in range(w_count)]
for word in word2index:
    word_index[word2index[word]] = word




#%% Generate sentences

def gen_sentences(max_length=15):
    pre_word = 0
    word_lst = [] 
    c_w = None
    while( c_w != 1):
        x_test = np.zeros(w_count)
        x_test[pre_word] = 1

       
        prob_lst = model .calc1(x_test)

      
        temp = random.random()

        t = 0
        if  c_w == 1:
            break
        i=0
        while(i <=w_count):
            t += prob_lst[i]
            if temp < t:
                c_w = i
                break
            i=i+1   
        

        word = word_index[ c_w]
        word_lst.append(word)

        if len(word_lst) == max_length:
            return word_lst
        
        pre_word =  c_w
    
    return word_lst


i=1
while(i<=5):
    word_lst = gen_sentences()
    print(i,".sentence-> ",*word_lst,"\n")
    text_file= open("word_level_sentence.txt","w+")  
    text_file.write("%s\n" %word_lst)
    text_file.close()
    i=i+1

