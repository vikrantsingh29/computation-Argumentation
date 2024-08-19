#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import gensim
from gensim.models import Word2Vec
from gensim.models import Doc2Vec
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
import datetime
# from better_profanity import profanity
# from profanity_check import predict, predict_prob
import string
# import joblib


df = pd.read_json("static/train-data-prepared.json")
df2 = pd.read_json("static/val-data-prepared.json")
badWords_df = pd.read_csv('static/bad-words.txt')

with open('static/bad-words.txt', 'r') as myfile:
     data = myfile.read()

badWordDictionary = data.split()



# In[2]:


df['bodyText'] = ""
df['upVotesFirstComment'] = 0
df['upVotesSecondComment'] = 0
df['badWordsFirstComment'] = 0
df['badWordsSecondComment'] = 0
df['lateNightFirstComment'] = 0
df['lateNightSecondComment'] = 0
df['bodyFirstSIARating'] =0
df['bodySecondSIARating'] = 0

df2['bodyText'] = ""
df2['upVotesFirstComment'] = 0
df2['upVotesSecondComment'] = 0
df2['badWordsFirstComment'] = 0
df2['badWordsSecondComment'] = 0
df2['lateNightFirstComment'] = 0
df2['lateNightSecondComment'] = 0
df2['bodyFirstSIARating'] =0
df2['bodySecondSIARating'] = 0

# In[3]:


nltk.download('tagsets')
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# In[4]:

def sentimentAnalysisNLTK(mainBody):
    temp = sia.polarity_scores(mainBody)
    return temp['compound']

def getBodyContent(df):
    dataFrameLength = len(df.index)
    for i in range (0,dataFrameLength):
        precedingPosts = df.iloc[i]['preceding_posts']
        
        #getting the body from both the arguments and concat that
        bodyFirst = precedingPosts[0]['body']
        bodySecond = precedingPosts[1]['body']
        mainBody = bodyFirst + bodySecond
        df.at[i, 'bodyText'] = mainBody

        df.loc[i, 'bodyFirstSIARating'] = sentimentAnalysisNLTK(bodyFirst)
        df.loc[i, 'bodySecondSIARating'] = sentimentAnalysisNLTK(bodySecond)

        #getting up votes of each argument
        upFirst = precedingPosts[0]['ups']
        df.at[i, 'upVotesFirstComment'] = upFirst
        upSecond = precedingPosts[1]['ups']
        df.at[i, 'upVotesSecondComment'] = upSecond
        
        
        #getting the number of bad words used in the sentance for the first Comment
        badWordsCount = 0
        wordlist = list()
        wordlist = bodyFirst.split()
        listLength = len(wordlist)
        
        for j in range(0,listLength):
            if wordlist[j] in badWordDictionary:
                badWordsCount = badWordsCount + 1
            
        df.at[i, 'badWordsFirstComment'] = badWordsCount        
           
        badWordsCount = 0
        wordlist.clear()
        
        #getting the number of bad words used in the sentance for the second Comment
        wordlist = bodySecond.split()
        listLength = len(wordlist)
        
        for j in range(0,listLength):
            if wordlist[j] in badWordDictionary:
                badWordsCount = badWordsCount + 1
            
        df.at[i, 'badWordsSecondComment'] = badWordsCount        
        
        badWordsCount = 0
        wordlist.clear()
        
                
        #getting the time of the first comment made
        firstArgumentTime = precedingPosts[0]['created'] 
        st = datetime.datetime.fromtimestamp(firstArgumentTime)
        if(st.hour > 21):
            df.at[i, 'lateNightFirstComment'] = 1
            
            
        #getting the time of the second comment made
        secondArgumentTime = precedingPosts[1]['created'] 
        st = datetime.datetime.fromtimestamp(secondArgumentTime)
        if(st.hour > 21):
            df.at[i, 'lateNightSecondComment'] = 1
            
    return df
            
                               
df = getBodyContent(df)
df2 = getBodyContent(df2)


# In[5]:


df['noun'] = 0
df['verb'] = 0
df['verbPhrase'] = 0
df['interjection'] = 0
df['adverbComparitive'] = 0
df['adverb'] = 0
df['preposition'] = 0
df['otherVerbs'] = 0
df['symbols'] = 0
df['adjectives'] = 0
df['conjuction'] = 0

df2['noun'] = 0
df2['verb'] = 0
df2['verbPhrase'] = 0
df2['interjection'] = 0
df2['adverbComparitive'] = 0
df2['adverb'] = 0
df2['preposition'] = 0
df2['otherVerbs'] = 0
df2['symbols'] = 0
df2['adjectives'] = 0
df2['conjuction'] = 0


# In[6]:



# counts verbs
def addFeatureVectures(dataframe):
    length_dataFrame = len(dataframe.index)

    for i in range(0, length_dataFrame):

        #   initializing and resetting the variables
        counter_verb = 0
        counter_verbPhrase = 0
        counter_otherVerbs = 0
        counter_noun = 0
        counter_preposition = 0
        counter_adjective = 0
        counter_adverb = 0
        counter_adverbComparitive = 0  
        counter_conjuction = 0        
        counter_interjection = 0      
        counter_symbols = 0

        #   tokenizing and getting POS Tags
        text = nltk.word_tokenize(df['bodyText'][i])
        words_value = nltk.pos_tag(text)
        tagLength = len(words_value)

        for tags in range(0, tagLength):
            word_tag = words_value[tags][1]
            text = (words_value[tags][0]).lower()

            if (word_tag == 'VB'):
                counter_verb = counter_verb + 1
                
            if (word_tag == 'VBP'):
                counter_verbPhrase = counter_verbPhrase + 1
                
            if(word_tag == 'VBD'
               or word_tag == 'VBZ'
               or word_tag == 'VBG'
               or word_tag == 'VBN'):
                counter_otherVerbs = counter_otherVerbs + 1
                
            if (word_tag == "NN"
                    or word_tag == "NNP"
                    or word_tag == "NNS"
                    or word_tag == "PRP"):
                counter_noun = counter_noun + 1

            if (word_tag == "IN"):
                counter_preposition = counter_preposition + 1

            if (word_tag == "JJ"
                    or word_tag == "JJR"
                    or word_tag == "JJS"):
                counter_adjective = counter_adjective + 1
                
            if (word_tag == "RB"
                    or word_tag == "RBS"):
                counter_adverb = counter_adverb + 1
                
            if (word_tag == "RBR"):
                counter_adverbComparitive = counter_adverbComparitive + 1

            if (word_tag == "CC"):
                counter_conjuction = counter_conjuction + 1
                
            if (word_tag == "UH"):
                counter_interjection = counter_interjection + 1
                
            if (word_tag == "SYM"):
                counter_symbols = counter_symbols + 1

        
        dataframe.at[i,'noun'] = counter_noun
        dataframe.at[i,'verb'] = counter_verb 
        dataframe.at[i,'verbPhrase'] = counter_verbPhrase
        dataframe.at[i,'interjection'] = counter_interjection 
        dataframe.at[i,'adverbComparitive'] = counter_adverbComparitive
        dataframe.at[i,'adverb'] = counter_adverb
        dataframe.at[i,'preposition'] = counter_preposition
        dataframe.at[i,'otherVerbs'] = counter_otherVerbs
        dataframe.at[i,'symbols'] = counter_symbols
        dataframe.at[i,'adjectives'] = counter_adjective
        dataframe.at[i,'conjuction'] = counter_conjuction      

    return dataframe


df = addFeatureVectures(df)
df2 = addFeatureVectures(df2)


# In[7]:


#word2Vec Embeddings

corpus = list()
arguments =df['bodyText'].values.tolist()

#for test data
corpus2 = list()
argumentsTest = df2['bodyText'].values.tolist()


# In[8]:


for line in arguments:
    tokens = word_tokenize(line) # tokenization
    tokens = [w.lower() for w in tokens] # making all the tokens to lower case
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    words = [word for word in stripped if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    corpus.append(words)
          


# In[9]:


for line in argumentsTest:
    tokens2 = word_tokenize(line) # tokenization
    tokens2 = [w.lower() for w in tokens2] # making all the tokens to lower case
    table2 = str.maketrans('', '', string.punctuation)
    stripped2 = [w.translate(table2) for w in tokens2]
    words2 = [word for word in stripped2 if word.isalpha()]
    stop_words2 = set(stopwords.words('english'))
    words2 = [w for w in words2 if not w in stop_words2]
    corpus2.append(words2)


# In[10]:


#word2vec model

word2vec_model = Word2Vec(corpus,
                 window=5,
                 workers=4,
                 min_count=1)


# In[11]:


tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(corpus)
vocab_length = len(tokenizer.word_index) + 1

max_length = max([len(s.split()) for s in df['bodyText']])

X_train_padded = pad_sequences(tokenizer.texts_to_sequences(corpus), maxlen=max_length)
y_train = df['label']


# In[12]:





# In[13]:


embedding_matrix = np.zeros((vocab_length, 100)) #using 100 featureVectors

#making the wordEmbeddings
for word, token in tokenizer.word_index.items():
    if word2vec_model.wv.__contains__(word):
        embedding_matrix[token] = word2vec_model.wv.__getitem__(word)
        


# In[14]:


#word2VecEmbeddingLayer
input_1 = Input(shape=(max_length,))

X_Train_Layer2 = df[['upVotesFirstComment','upVotesSecondComment','badWordsFirstComment',
              'badWordsSecondComment','lateNightFirstComment','lateNightSecondComment','noun','verb','verbPhrase',
              'interjection','adverbComparitive','adverb','preposition','otherVerbs',
              'symbols','adjectives','conjuction','bodyFirstSIARating', 'bodySecondSIARating']].values


word2Vec_embedding_layer = Embedding(input_dim=vocab_length,
                            output_dim=100,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)(input_1)


# In[15]:


biLSTM = LSTM(128)(word2Vec_embedding_layer)
input_2 = Input(shape=(19,))
concat_layer = Concatenate()([biLSTM, input_2])
dropout = Dropout(0.3)(concat_layer)
dense = Dense(1)(dropout)
active = Activation('sigmoid')(dense)


# In[16]:


model_Classifier = Model([input_1, input_2], active)
model_Classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_Classifier.summary()


# In[17]:


history = model_Classifier.fit(x=[X_train_padded, X_Train_Layer2], 
                               y = y_train,  
                               epochs=5)


# In[37]:


tokenizer = Tokenizer(filters="", lower=False, oov_token="<oov>")
tokenizer.fit_on_texts(corpus2)
vocab_length2 = len(tokenizer.word_index) + 1
X_test_padded = pad_sequences(tokenizer.texts_to_sequences(corpus2), maxlen=max_length)
y_test = df2['label']

X_Test_Layer2 = df2[['upVotesFirstComment','upVotesSecondComment','badWordsFirstComment',
              'badWordsSecondComment','lateNightFirstComment','lateNightSecondComment','noun','verb','verbPhrase',
              'interjection','adverbComparitive','adverb','preposition','otherVerbs',
              'symbols','adjectives','conjuction']].values

y_pred = model_Classifier.predict(x=[X_test_padded,X_Test_Layer2,])
y_pred = np.where(y_pred >= 0.5, 1, 0)


# In[38]:


print(classification_report(y_test, y_pred))


# In[39]:


import json
b = y_pred.flatten() 
dict = {}
for i in range(0 , len(b)):
    dict[df2['id'][i]] = int(b[i])

jsonObj = json.dumps(dict)
with open ('prediction-value.txt' , 'w') as f:
        f.write(jsonObj)


# In[ ]:


# #doc2Vec
# from nltk.tokenize import word_tokenize
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument

# # Create the tagged document needed for Doc2Vec
# # def create_tagged_document(_d):
# tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(df['bodyText'])]

# # train_data = list(create_tagged_document(df['bodyText']))

# print(tagged_data[:1])


# In[ ]:


# # Init the Doc2Vec model
# doc2Vec_model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

# # Build the Volabulary
# doc2Vec_model.build_vocab(tagged_data)


# # Train the Doc2Vec model
# doc2Vec_model.train(tagged_data, total_examples=doc2Vec_model.corpus_count, epochs=doc2Vec_model.epochs)


# In[ ]:


# print(doc2Vec_model.infer_vector(df.iloc[51]['bodyText'].split()))


# In[ ]:




