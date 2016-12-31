# Stanford Sentiment Analysis


## Modules
	* softmax (vectorized implementation)
	* word2vec (Skipgram, CBOW)
	* stochastic gradient descent

	
## Sentiment Analysis
When the word vectors are trained, we are going to perform a simple sentiment analysis. For each sentence in the Stanford Sentiment Treebank dataset, we are going to use the average of all the word vectors in that sentence as its feature, and try to predict the sentiment level of the said sentence. The sentiment level of the phrases are represented as real values in the original dataset, here we'll just use five classes:

"very negative", "negative", "neutral", "positive", "very positive"
<br>
	

# Acknowledgement
> CS224d: Deep Learning for Natural Language Processing
> CS224d: Assignment #1