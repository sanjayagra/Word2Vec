
# coding: utf-8

# In[1]:

import os
import gensim
import datetime
import numpy as np
import pandas as pd
import gc
from csv import DictReader


# In[7]:

event_vectors_input=pd.read_csv("/axp/rim/imsadsml/warehouse/sagra39/Orchestra/python_code_march/oet_doe_input_event_vectors_new.csv")


# In[10]:

event_vectors_input.tail(2)


# In[11]:

event_vectors_input['list_var']= event_vectors_input['list_var_new_1'].map(lambda x: x.replace("[","").replace("]","").replace("'","").split(","))
event_vectors_input['list_var']=event_vectors_input['list_var'].map(lambda x:[i.strip() for i in x])
event_vectors_input=event_vectors_input[['cust_xref_id','datetime','list_var']]


# In[30]:

sentences = []
prevCM = ""
sentence = []

for index,row in event_vectors_input.iterrows():
    cm = row['cust_xref_id']
    offer=  row['list_var'] 
    
    if cm != prevCM and prevCM != "":
        
        sentences.append(sentence)
        sentence = []
        sentence +=offer
        prevCM = cm

    else:
        sentence+=offer
        prevCM = cm
sentences.append(sentence)        
print len(sentences)


# In[6]:

size = 32
model = (gensim.models.Word2Vec(sentences = sentences,  size=size, alpha=0.025, window=6,
                                min_count=2, sample=0, seed=1, 
                                workers=15, min_alpha=0.0001, sg=1, hs=1, negative=0, 
                                cbow_mean=0, iter=5))
model.save('/axp/rim/imsadsml/warehouse/sagra39/Orchestra/python_code_march/word2vec_model_32_new.word2vec')


# In[7]:

model = gensim.models.Word2Vec.load('/axp/rim/imsadsml/warehouse/sagra39/Orchestra/python_code_march/word2vec_model_32_new.word2vec')


# In[9]:

len(model.wv.vocab)


# In[10]:

f = open('/axp/rim/imsadsml/warehouse/sagra39/Orchestra/python_code_march/offer_event_vectors.csv', "w")
for word in model.wv.vocab.keys():
    f.write(word)
    for v in model[word]:
        f.write(",")        
        f.write(str(v))
    f.write("\n")
f.flush()
f.close()

