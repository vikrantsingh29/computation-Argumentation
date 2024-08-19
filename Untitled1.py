#!/usr/bin/env python
# coding: utf-8

# In[1]:


print("hello")


# In[2]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pandas as pd
import numpy as np

from gensim.models import Word2Vec

from keras.preprocessing.sequence import pad_sequences
from keras.layers.core import Dropout, Dense
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.core import Activation
from sklearn.metrics import confusion_matrix, classification_report
import string


# In[3]:


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# In[3]:


df = pd.read_json("C:/Users/vikrant.singh/Desktop/ca21-assignment-2/train-data-prepared.json")
df2 = pd.read_json("C:/Users/vikrant.singh/Desktop/ca21-assignment-2/val-data-prepared.json")

df['noun'] = 0
df['verb'] = 0
df['conjunction'] = 0
df['adjective'] = 0
df['adverb'] = 0
df['preposition'] = 0
df['determiner'] = 0
df['that'] = 0
df['modal'] = 0
df['to'] = 0

df2['noun'] = 0
df2['verb'] = 0
df2['conjunction'] = 0
df2['adjective'] = 0
df2['adverb'] = 0
df2['preposition'] = 0
df2['determiner'] = 0
df2['that'] = 0
df2['modal'] = 0
df2['to'] = 0

y_trainn = df['label']
y_testt = df2['label']


# In[4]:



# counts verbs
def addFeatureVectures(dataframe):
    length_dataFrame = len(dataframe.index)

    for i in range(0, length_dataFrame):

        #   initializing the variables
        counter_verb = 0
        counter_noun = 0
        counter_preposition = 0
        counter_adjective = 0
        counter_adverb = 0
        counter_determiner = 0
        counter_conjuction = 0
        counter_that = 0
        counter_modal = 0
        counter_to = 0

        #   tokenizing and getting POS Tags
        text = nltk.word_tokenize(dataframe['text'][i])
        words_value = nltk.pos_tag(text)
        tagLength = len(words_value)

        for tags in range(0, tagLength):
            word_tag = words_value[tags][1]
            text = (words_value[tags][0]).lower()

            if (word_tag == 'VB'
                    or word_tag == 'VBP'
                    or word_tag == 'VBD'
                    or word_tag == 'VBZ'
                    or word_tag == 'VBG'
                    or word_tag == 'VBN'):
                counter_verb = counter_verb + 1
                
            if (word_tag == "TO"):
                counter_to = counter_to + 1

            if (word_tag == "NN"
                    or word_tag == "NNP"
                    or word_tag == "NNS"
                    or word_tag == "PRP"):
                counter_noun = counter_noun + 1

            if (word_tag == "IN"):
                counter_preposition = counter_preposition + 1

            if (word_tag == "DT"):
                counter_determiner = counter_determiner + 1

            if (word_tag == "JJ"
                    or word_tag == "JJR"
                    or word_tag == "JJS"):
                counter_adjective = counter_adjective + 1

            if (word_tag == "CC"):
                counter_conjuction = counter_conjuction + 1

            if (text == 'that'):
                counter_that = counter_that + 1

            if (word_tag == "RB"
                    or word_tag == "RBR"
                    or word_tag == "RBS"):
                counter_adverb = counter_adverb + 1

            if (word_tag == "MD"):
                counter_modal = counter_modal + 1

        dataframe.at[i, 'verb'] = counter_verb
        dataframe.at[i, 'noun'] = counter_noun
        dataframe.at[i, 'preposition'] = counter_preposition
        dataframe.at[i, 'that'] = counter_that
        dataframe.at[i, 'conjuction'] = counter_conjuction
        dataframe.at[i, 'adverb'] = counter_adverb
        dataframe.at[i, 'adjective'] = counter_adjective
        dataframe.at[i, 'determiner'] = counter_determiner
        dataframe.at[i, 'modal'] = counter_modal

    return dataframe


df = addFeatureVectures(df)
df2 = addFeatureVectures(df2)


# In[5]:


X = df.copy()
X_trainn = X.drop('label', axis=1)

X2 = df2.copy()
X2_trainn = X2.drop('label', axis=1)


# In[6]:


max_length = max([len(s.split()) for s in df['text']])
max_length2 = max([len(s.split()) for s in df2['text']])


# In[7]:


nltk.download('stopwords')


# In[8]:



corpus = list()
corpus2 = list()

comments = df['text'].values.tolist()
comments2 = df['text'].values.tolist()

for line in comments:
    tokens = word_tokenize(line)
    tokens = [w.lower() for w in tokens]
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    corpus.append(words)

# for line in comments2:
#     tokens = word_tokenize(line)
#     tokens = [w.lower() for w in tokens]
#     table = str.maketrans('', '', string.punctuation)
#     stripped = [w.translate(table) for w in tokens]
#     words = [word for word in stripped if word.isalpha()]
#     stop_words = set(stopwords.words('english'))
#     words = [w for w in words if not w in stop_words]
#     corpus2.append(words)


# In[9]:


model = Word2Vec(corpus,
                 window=5,
                 workers=4,
                 min_count=1,
                 epochs=100)

# model2 = Word2Vec(corpus2,
#                  window=5,
#                  workers=4,
#                  min_count=1,
#                  epochs=100)


# In[10]:


X_train, y_train = np.array(df['text']), np.array(df['label'])
X_test, y_test = np.array(df2['text']), np.array(df2['label'])


# In[11]:


tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(X_train)
vocab_length = len(tokenizer.word_index) + 1
X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_length)


# In[12]:


tokenizer2 = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer2.fit_on_texts(X_test)
vocab_length2 = len(tokenizer2.word_index) + 1
X_test_padded = pad_sequences(tokenizer2.texts_to_sequences(X_test), maxlen=max_length)


# In[13]:


print("X_train.shape:", X_train_padded.shape)
print("X_test.shape :", X_test_padded.shape)


# In[14]:


embedding_matrix = np.zeros((vocab_length, 100))

# embedding_matrix2 = np.zeros((vocab_length2, 100))


# In[15]:


for word, token in tokenizer.word_index.items():
    if model.wv.__contains__(word):
        embedding_matrix[token] = model.wv.__getitem__(word)


# In[16]:


# for word, token in tokenizer2.word_index.items():
#     if model2.wv.__contains__(word):
#         embedding_matrix2[token] = model2.wv.__getitem__(word)        


# In[17]:


X2_train = df[['verb',
               'adjective', 'adverb', 'preposition'
               ,'that']].values
X2_test = df2[['verb',
               'adjective', 'adverb', 'preposition'
               ,'that']].values


# In[18]:


input_1 = Input(shape=(max_length,))


# In[19]:


embedding_layer = Embedding(input_dim=vocab_length,
                            output_dim=100,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)(input_1)


# In[20]:


biLSTM = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3))(embedding_layer)
input_2 = Input(shape=(5,))
concat_layer = Concatenate()([biLSTM, input_2])
dropout = Dropout(0.3)(concat_layer)
dense = Dense(1)(dropout)
active = Activation('sigmoid')(dense)


# In[21]:



model_Classifier = Model([input_1, input_2], active)
model_Classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[22]:


history = model_Classifier.fit(x=[X_train_padded, X2_train], 
                               y=y_train, 
                               batch_size=100, 
                               epochs=10, 
                               verbose=1)


# In[85]:


y_pred = model_Classifier.predict(x=[X_test_padded, X2_test])
import json
# Converting prediction to reflect the sentiment predicted.

y_pred = np.where(y_pred >= 0.5, 1, 0)
df3 = pd.DataFrame()
df3['id'] = df2['id']

df3.set_index('id')
df3['prediction'] = y_pred
x = df3.to_json(orient = 'values')
parsed = json.loads(x)
my_json = json.dumps(parsed, indent=2) 
b = y_pred.flatten()


# for i in range (0, len(b)):
#     b[i] = int(b[i])
    
dict = {}
for i in range(0 , len(b)):
    dict[df['id'][i]] = int(b[i])

# dict = dict.to_json() 
jsonObj = json.dumps(dict)
with open ('prediction-value.txt' , 'w') as f:
        f.write(jsonObj)
    

    
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




