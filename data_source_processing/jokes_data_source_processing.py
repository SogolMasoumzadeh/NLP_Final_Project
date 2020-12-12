import pandas as pd
import copy
pd.set_option('max_colwidth', -1)
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import copy

df = pd.read_json ('drive//My Drive/reddit_jokes.json')
df['text'] = df[['title', 'body']].apply(lambda x: ' '.join(x), axis=1)
df = df.loc[df['score'] >= 12]
df['source'] = 'reddit_jokes'
df['text'] = df['text'].str.replace('\n'," ")
df['text'] = df['text'].str.replace('"',' ')
df_jokes1 = df[['text','source']].drop_duplicates() 

df1 = pd.read_csv('drive//My Drive/shortjokes.csv') 
df1['source'] = 'short_jokes'
df1['Joke'] = df1['Joke'].str.replace("\r"," ")
df1['Joke'] = df1['Joke'].str.replace('\n'," ")
df1['Joke'] = df1['Joke'].str.replace('"',' ')
df_jokes2 = df1[['Joke','source']].drop_duplicates()  
df_jokes2.rename(columns={'Joke':'text'}, inplace=True)

df3 = pd.read_json ('drive//My Drive/wocka.json')
df3['source'] = 'wocka'
df3['body'] = df3['body'].str.replace("\r"," ")
df3['body'] = df3['body'].str.replace("\n"," ")
df3['body'] = df3['body'].str.replace('"',' ')
df3.rename(columns={'body':'text'}, inplace=True)
df_jokes4 = df3[['text','source']].drop_duplicates()  

stack = pd.concat([df_jokes1, df_jokes2, df_jokes4], axis=0)
#stack = df_jokes4
stack.reset_index(drop=True, inplace=True)
stack = stack.drop_duplicates(subset=['text'], keep='last')
stack.reset_index(drop=True, inplace=True)
print(stack.shape)
stack.head(2)

stack['count_len_sentence'] = stack['text'].apply(lambda x: len(x))
stack = stack.loc[ (stack['count_len_sentence'] > 5) & (stack['count_len_sentence'] <=200) ]
stack['count_words'] = stack['text'].str.split().apply(len)
stack = stack.loc[ (stack['count_words'] >= 6) & (stack['count_words'] <=45) ]

def preprocessing(df):
  from collections import defaultdict
  from nltk.corpus import wordnet as wn
  df['text'] = df['text'].str.replace('\n'," ")
  df['text'] = df['text'].str.replace('\r'," ")
  print('initial shape df: ' +  str(df.shape))
  #as per jokes profile, sentence length between 5 and 200 characters and word count between 6 and 45
  df['count_length_sent'] = df['text'].apply(lambda x: len(x))
  df = df.loc[ (df['count_length_sent'] > 5) & (df['count_length_sent'] <=200) ]
  df['count_words'] = df['text'].str.split().apply(len)
  df = df.loc[ (df['count_words'] >= 6) & (df['count_words'] <=40) ]

  print('sentence length between 5 and 200 characters and word count between 6 and 45. The current shape : ' + str(df.shape))

  #start nltk tokenize
  df['text_token'] = df['text'].apply(nltk.tokenize.WordPunctTokenizer().tokenize)

  #check if the sentence contains a question
  target_words = ['?']
        # Convert target words to lower case just to be safe.
        #  target_words = [word.lower() for word in target_words]

  df['question_flag'] = df['text_token'].apply(lambda words: any(target_word in words for target_word in target_words))

#  print('how many questions?')
#  print(df.groupby('question_flag').count() )

  #within sentence: lemmatize
  tag_map = defaultdict(lambda : wn.NOUN)
  tag_map['J'] = wn.ADJ
  tag_map['V'] = wn.VERB
  tag_map['R'] = wn.ADV 

  df['text_lemma'] = df['text_token'].apply(lambda words: [nltk.stem.WordNetLemmatizer().lemmatize(token, tag_map[tag[0]]) for token, tag in nltk.pos_tag(words)] )

  #within sentence: exclude stop words
  stop_words = set(nltk.corpus.stopwords.words('english')) 
  df['text_lemma_nostop'] = df['text_lemma'].apply(lambda words: [word for word in words if not word in stop_words])

  import string
  punc = string.punctuation
  punc

  #within sentence: exclude punctuation
  df['text_lemma_final'] = df['text_lemma_nostop'].apply(lambda x: [word for word in x if word not in punc])

  #exclude rows where there is a word with char length > 20 holacomoestasbien!!
  df['max_len_word'] = df['text_lemma_final'].apply(lambda x: max(len(w) for w in x) if len(x) > 0 else 0 )
  df = df.loc[ (df['max_len_word'] <= 20) ]

  df['text_final'] = df['text_lemma_final'].apply(lambda x: ' '.join(x))
  df.drop(['text_lemma','text_lemma_nostop','text_lemma_final'], axis=1, inplace=True)
  df = df.drop_duplicates(subset=['text_final'], keep='last')

  print('final shape: '+  str(df.shape))
  return df


jokes.drop(['text_token', 'text_lemma', 'text_lemma_nostop'], axis=1, inplace=True)

jokes.to_csv('jokes_processed_20201110.csv')

jokes=pd.read_csv('drive//My Drive/jokes_processed_20201110.csv', index_col=0, converters={"text_lemma_final": literal_eval})
jokes.reset_index(drop=True, inplace=True)





