from .engine import LinearClassifier
import matplotlib.pyplot as plt
import seaborn as sns
	
def filter_non_humor():
	filter_non_humor_amaz_yahoo()
	filter_non_humor_comm_reddit()



def filter_non_humor_amaz_yahoo():
	import os
	list = []
	directory = ['df_amaz_rev00', 'df_amaz_rev01','df_amaz_rev02','df_amaz_rev03','df_amaz_rev04', 'df_amaz_qa', 'df_yahoo_news']

	for folder in directory:
		for file in os.listdir(folder):
			if file.endswith(".csv"):
				try:
					df=pd.read_csv(folder+'/'+file, index_col=0)
					df.rename(columns={'0':'text'}, inplace=True)
					df.reset_index(drop=True, inplace=True)
					df['text'] = df['text'].apply(lambda x: str(x))
					print(file, df.shape)
					list.append(df)
				except:
					pass
	print(len(list))
	df_nojokes = pd.concat(list)
	print(df_nojokes.shape)

	#df_nojokes.to_csv("no_joke_20201202.csv")	
	df_nojokes=pd.read_csv('no_joke_20201202.csv', index_col=0, dtype='str')
	df_nojokes.drop(['source'], axis=1, inplace=True)
	df_nojokes = df_nojokes.drop_duplicates(subset=['text'], keep='first')
	df_nojokes.reset_index(drop=True, inplace=True)

	df_nojokes.sample(1000000).to_csv("amazon_yahoo.csv")

	JOKES_PATH = "jokes_processed_20201110.csv"
	NO_JOKES_PATH = "amazon_yahoo.csv"
	
	c = LinearClassifier()
	c.run()
	
	df_nojokes['text'] = df_nojokes['text'].apply(str)
	pred_jokes, prob_jokes = c.classifier_prediction(df_nojokes['text']) 
	df_nojokes['prob'] = pd.DataFrame(prob_jokes)
	df_nojokes['joke_pred'] = pd.DataFrame(pred_jokes)
	df_nojokes = df_nojokes[['text','joke_pred', 'prob']]
	df_nojokes = df_nojokes[df_nojokes['joke_pred']==1]
	print(df_nojokes.shape)

	sns.set(rc={'figure.figsize':(8.7,5.6)})
	plt.hist(df_nojokes['prob'], bins=20)
	plt.show()

	#df_nojokes[df_nojokes['prob']>=0.75].shape   #241515
	df_nojokes[df_nojokes['prob']>=0.75].to_csv("no_jokes_amaz_yahoo_20201204.csv")

def filter_non_humor_comm_reddit():
	lines = []
	with open("Answers_R.txt",  encoding='utf8') as file_in:
		for line in file_in:
			lines.append(line.replace('\n',""))
	with open("Questions_R.txt",  encoding='utf8') as file_in:
		for line in file_in:
			lines.append(line.replace('\n',""))
		
	df_AnswerQuestR =pd.DataFrame(lines)
	df_AnswerQuestR.rename(columns={0:'text'}, inplace=True)	

	df_comments_neg=pd.read_csv('df_comments_negative.csv', index_col=0, dtype='str')
	df_comments_pos=pd.read_csv('df_comments_positive.csv', index_col=0, dtype='str')
	
	df_4Mreddit=pd.read_csv('df_4Mreddit.csv', index_col=0, dtype='str')
	
	stack = pd.concat([df_AnswerQuestR, df_comments_neg, df_comments_pos, df_4Mreddit], axis=0)
	
	stack = stack.drop_duplicates(subset=['text'], keep='first')
	#stack.shape 15297276
	
	stack['text'] = stack['text'].str.replace('\n'," ")
	stack['text'] = stack['text'].str.replace('\r'," ")
	print('initial shape df: ' +  str(stack.shape))
	#as per jokes profile, sentence length between 5 and 200 characters and word count between 6 and 44
	stack['count_length_sent'] = stack['text'].apply(str).apply(lambda x: len(x))
	stack = stack.loc[ (stack['count_length_sent'] > 5) & (stack['count_length_sent'] <=215) ]
	stack['count_words'] = stack['text'].str.split().apply(len)
	stack = stack.loc[ (stack['count_words'] >= 6) & ( stack['count_words'] <=45) ]
	print('after shape df: ' +  str(stack.shape))
	#initial shape df: (15297276, 1)
	#after shape df: (11744316, 3)

	stack.sample(1000000).to_csv("reddit_comments.csv")
	JOKES_PATH = "jokes_processed_20201110.csv"
	NO_JOKES_PATH = "reddit_comments.csv"
	
	c = LinearClassifier()
	c.run()
	
	pred_jokes, prob_jokes = c.classifier_prediction(stack['text']) 
	stack['prob'] = pd.DataFrame(prob_jokes)
	stack['joke_pred'] = pd.DataFrame(pred_jokes)
	stack = stack[['text', 'joke_pred', 'prob']]
	#print(stack.shape)
	#print(stack.head(2))
	stack = stack[stack['joke_pred']==1]
	print(stack.shape)

	sns.set(rc={'figure.figsize':(8.7,5.6)})
	plt.hist(stack['prob'], bins=20)
	plt.show()

	#stack[stack['prob']>=0.65].shape #(242394, 3)
	stack[stack['prob']>=0.65].to_csv("no_jokes_reddit_20201204.csv")
	

filter_non_humor()