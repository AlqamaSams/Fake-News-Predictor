#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re #for seaarching text in a document
from nltk.corpus import stopwords #means body or important content of the text
from nltk.stem.porter import PorterStemmer #return root word or main word 
from sklearn.feature_extraction.text import TfidfVectorizer #convert text to number
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


fn= pd.read_csv(r'C:\Users\Zeeshan Sams\Downloads\ML PROJECTS\fakenews.csv')


# In[3]:


fn.head()


# In[4]:


fn.isnull().sum()


# In[5]:


fn.shape


# In[6]:


fn.fillna('') #here we will fill the null values


# In[7]:


fn.info()


# In[8]:


import nltk
nltk.download('stopwords')
print(stopwords.words('english'))


# we will pre-process the data

# In[9]:


fn['label'].value_counts()


# as we can see the data is balanced which will give the better accuracy

# In[10]:


fn['content']= fn['author']+' '+fn['title']


# In[11]:


print(fn['content'])


# now we will compress the data to only main words with the help of stemming

# In[14]:


port_stem= PorterStemmer()


# In[15]:


stemmed_content = re.sub('[^a-zA-Z]',' ','Alqama scored 90 marks')
print(stemmed_content)


# In[16]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',str(content))
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[17]:


fn['content'] = fn['content'].apply(lambda x:stemming(x))


# In[18]:


print(fn['content'])


# In[20]:


x= fn['content'].values
y= fn['label'].values
print(x)
print(y)


# converting the data to numerical values

# In[21]:


vectorizer= TfidfVectorizer()


# In[24]:


vectorizer.fit(x)
x=vectorizer.transform(x)
print(x)


# In[25]:


#train test split


# In[27]:


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.3, stratify=y, random_state=2)
print(x.shape, x_train.shape, x_test.shape)


# In[28]:


#model training


# In[29]:


model= LogisticRegression()


# In[30]:


model= model.fit(x_train, y_train)


# In[31]:


#predict model


# In[32]:


x_train_prediction= model.predict(x_train)
training_data_accuracy= accuracy_score(x_train_prediction, y_train)
print(training_data_accuracy)


# In[33]:


x_test_prediction= model.predict(x_test)
test_data_accuracy= accuracy_score(x_test_prediction, y_test)
print(test_data_accuracy)


# for checking fake news data

# In[44]:


x_new_news= x_test[2]
prediction= model.predict(x_new_news)
print(prediction)
if prediction[0]==0:
    print('the news is real')
else:
    print('the news is fake')


# In[45]:


print(y_test[2])


# In[ ]:




