import os
import json, re
import gzip
import pandas as pd
from urllib.request import urlopen

######################################################
##### process amazon QA
######################################################

!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Health_and_Personal_Care.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Appliances.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Arts_Crafts_and_Sewing.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Automotive.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Baby.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Beauty.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Cell_Phones_and_Accessories.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Clothing_Shoes_and_Jewelry.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Electronics.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Grocery_and_Gourmet_Food.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Health_and_Personal_Care.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Home_and_Kitchen.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Industrial_and_Scientific.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Musical_Instruments.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Office_Products.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Patio_Lawn_and_Garden.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Pet_Supplies.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Software.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Sports_and_Outdoors.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Tools_and_Home_Improvement.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Toys_and_Games.json.gz
!wget http://jmcauley.ucsd.edu/data/amazon/qa/qa_Video_Games.json.gz


def parse(path):
    g = gzip.open(path, 'rb')
    i=0
    for l in g:
        #if i < 10:
        #    print(l )
            yield eval(l)
        #    i = i+1
        #else:
        #    break


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        #print(i)
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')
	

def load_json(list_files):
  data = []
  i=0
  for file in list_files:
    df_temp = getDF(file)
    df_temp['text'] = df_temp[['question', 'answer']].apply(lambda x: ' '.join(x), axis=1)
    df_temp['source'] = file
    df_temp.drop(['questionType', 'asin', 'answerTime', 'unixTime','question', 'answer', 'answerType'], axis=1, inplace=True)
    for index, row in df_temp.iterrows():
        #print(row['c1'], row['c2']) 
        sent = ''
        tokens = nltk.sent_tokenize(row['text'])
        for t in tokens:
            #print (t, len(t), "\n") 
            if len(sent) <= 40:
                sent = sent + ' ' + t
            else:
                break
        if 20<len(sent)<220:
            i = i + 1
            row['text'] = str(sent).strip()
            #data.append(str(sent).strip())
            if i%10000 == 0:
                print(i)
    data.append(df_temp)
    print('done for ' + file)
    #i=i+1

  frame = pd.concat(data, axis=0, ignore_index=True)
  #df = pd.DataFrame(data)
       
  return frame	


list_files = ['qa_Appliances.json.gz','qa_Arts_Crafts_and_Sewing.json.gz', 'qa_Automotive.json.gz' ,'qa_Baby.json.gz', 'qa_Beauty.json.gz',
              'qa_Cell_Phones_and_Accessories.json.gz','qa_Clothing_Shoes_and_Jewelry.json.gz','qa_Electronics.json.gz','qa_Grocery_and_Gourmet_Food.json.gz',
              'qa_Health_and_Personal_Care.json.gz','qa_Home_and_Kitchen.json.gz','qa_Industrial_and_Scientific.json.gz','qa_Musical_Instruments.json.gz',
              'qa_Office_Products.json.gz','qa_Patio_Lawn_and_Garden.json.gz','qa_Pet_Supplies.json.gz','qa_Software.json.gz','qa_Sports_and_Outdoors.json.gz',
              'qa_Tools_and_Home_Improvement.json.gz','qa_Toys_and_Games.json.gz','qa_Video_Games.json.gz']
df_amazon_qa = load_json(list_files)


df_amazon_qa = df_amazon_qa.drop_duplicates(subset=['text'], keep='first')

n = 115000  #chunk row size
list_df = [df_amazon_qa[i:i+n] for i in range(0,df_amazon_qa.shape[0],n)]

print(len(list_df))
for id, k in enumerate(list_df):
  file = 'df_amaz_qa_'+str(id)+'.csv'
  #print(list_df[id].head(1))
  #print(type(list_df[id]))
  print(list_df[id].shape)
  list_df[id].to_csv(file)
  #!cp $file "drive/My Drive/"
  print(file, id)
  
##################################################################### 
##### process amazon review
#####################################################################

!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/All_Beauty.json.gz
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Books.json.gz
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Digital_Music.json.gz
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Home_and_Kitchen.json.gz
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Industrial_and_Scientific.json.gz
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Magazine_Subscriptions.json.gz
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Movies_and_TV.json.gz
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Sports_and_Outdoors.json.gz
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Toys_and_Games.json.gz
!wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Video_Games.json.gz



def getDF2(path):
    data = []
  #keywords = set(keywords)
    i=0
    with gzip.open(path) as f:
        for l in f:
            line = json.loads(l.strip())
        #print(type(line) )
            try:
                sentence = line['reviewText'].strip().replace("\n",".").replace("\r",".")
                if i<6000000:
                    sent = ''
                    tokens = nltk.sent_tokenize(sentence)
                    for t in tokens:
                        #print (t, len(t), "\n") 
                        if len(sent) <= 30:
                            sent = sent + ' ' + t
                    #print(type(sent))
                    #if len(sent) < 150:
                        #data.append(sent.strip())
              #for keyword in keywords:
              #if re.findall(keyword, sentence):
              #if keywords.intersectmusic (sentence.lower().split() ) and i <10:
                    #print(sent)
                    if 20<len(sent)<125:
                        i = i + 1
                        data.append(str(sent).strip())
                        if i%100000 == 0:
                            print(i)
                #data.append(sentence) 
                #i = i+1
                else:
                    break
            except:
                pass
    

    return pd.DataFrame.from_dict(data)
  
#print(len(df))

def load_json2(list_files):
    data = []
    i=0
    for file in list_files:
        print('done for ' + file)
        df_temp = getDF2(file)
        data.append(df_temp)
        i=i+1
        print(i)

    frame = pd.concat(data, axis=0, ignore_index=True)
    #df = pd.DataFrame(data)
    print(i)     
    return frame

list_files_amazon_reviews = ['Books.json.gz','All_Beauty.json.gz' ,'Digital_Music.json.gz','Home_and_Kitchen.json.gz','Industrial_and_Scientific.json.gz','Magazine_Subscriptions.json.gz','Movies_and_TV.json.gz','Sports_and_Outdoors.json.gz','Toys_and_Games.json.gz','Video_Games.json.gz']
df_amaz_rev = load_json2(list_files_amazon_reviews)	
 
n = 110000  #chunk row size
list_df = [df_amaz_rev[i:i+n] for i in range(0,df_amaz_rev.shape[0],n)]

print(len(list_df))
for id, k in enumerate(list_df):
  file = 'df_amaz_rev00_'+str(id)+'.csv'
  #print(list_df[id].head(1))
  #print(type(list_df[id]))
  print(list_df[id].shape)
  list_df[id].to_csv(file)
  #!cp $file "drive/My Drive/"
  print(file, id)
  
  
#########################################################
### process yahoo news
#########################################################
import json
data = []
with open('drive//My Drive/train_news_yahoo.data', encoding='utf8')as p:
    i=0
    for line in p:
      #if i <=5000:
        sample = json.loads(line)
        paragraph = ' '.join( [ parag for parag in sample['paras']] )
        #print(paragraph)
        tokens = nltk.sent_tokenize(paragraph)
        sent = ''
        for t in tokens:
          #print (t, len(t), "\n") 
          if len(sent) < 35:
            sent = sent + ' ' + t
        #print('sentence: ', sent)
        #comments = ' '.join( [ comment['cmt'] for comment in sample['cmts']] )
        #text_line = paragraph.join(comments)
        #for id, x in enumerate( sample['cmts'] ): 
        #  print('comment: ', sample['cmts'][id]['cmt'])
        sent = sent.replace(' .',".")
        sent = sent.replace(' ,',",")
        sent = sent.replace("  n't","n't")
        sent = sent.replace(" 's","'s")  
        sent = sent.replace(" ’s","’s")
        sent = sent.replace(" ’m","’m") 
        sent = sent.replace(" 'm","'m")
        sent = sent.replace(" n't","n't")   
        sent = sent.replace("is n’t","isn’t")
        sent = sent.replace(" 've","'ve") 
        sent = sent.replace("s '","s'")  
        sent = sent.replace("‘ s","‘s")
        sent = sent.replace(" ?","?") 
        sent = sent.replace(" 'll","'ll")  
        sent = sent.replace(" 'd","'d")
        sent = sent.replace("did n't","didn't")
        if len(sent) < 350:
          data.append(sent)
        #data.append(comments)
        for id, x in enumerate( sample['cmts'] ):
          comm = str(sample['cmts'][id]['cmt'])
          comm = comm.replace(' .',".")
          comm = comm.replace(' ,',",")
          comm = comm.replace(" 's","'s")  
          comm = comm.replace(" ’s","’s")
          comm = comm.replace(" ’m","’m") 
          comm = comm.replace(" 'm","'m")
          comm = comm.replace(" n't","n't")   
          comm = comm.replace("is n’t","isn’t")
          comm = comm.replace(" 've","'ve") 
          comm = comm.replace("s '","s'")  
          comm = comm.replace("‘ s","‘s")
          comm = comm.replace(" ?","?") 
          comm = comm.replace(" 'll","'ll")  
          comm = comm.replace(" 'd","'d")
          comm = comm.replace("did n't","didn't")
          if 10 < len(comm) <= 150 :
            data.append(comm) 
            #print(id, 'comment: ', sample['cmts'][id]['cmt'])        
        #data.append(sample)

        #i=i+1
      #else:
        #break
      
df_news_y = pd.DataFrame.from_dict(data)		
df_news_y.rename(columns={0:'text'}, inplace=True)
n = 120000  #chunk row size
list_df = [df_news_y[i:i+n] for i in range(0,df_news_y.shape[0],n)]
print(len(list_df))
for id, k in enumerate(list_df):
  file = 'data_yahoo_news'+str(id)+'.csv'
  #print(list_df[id].head(1))
  #print(type(list_df[id]))
  print(list_df[id].shape)
  list_df[id].to_csv(file)
  !cp $file "drive/My Drive/"
  print(file, id)





  



