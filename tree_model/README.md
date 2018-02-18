## Library Dependencies
* Python 2.7
* Scipy Stack (`numpy`, `scipy` and `pandas`)
* [scikit-learn](http://scikit-learn.org/stable/)
* [XGBoost](http://xgboost.readthedocs.io/en/latest/)
* [gensim (for word2vec)](https://radimrehurek.com/gensim/)
* [NLTK (python NLP library)](http://www.nltk.org)

## Procedure
**1. Install all the dependencies**
**2. Download the word2vec model trained on Google News corpus. The file GoogleNews-vectors-negative300.bin has to be present in /tree_model.**
**3. Run generateFeatures.py to produce all the feature files (train_stances_processed.csv, train_bodies_processed.csv, test_bodies_processed.csv, test_stances_unlabeled.csv). The following files will be generated:**
```
train.sim.word2vec.pkl
train.sim.tfidf.pkl
train.sim.svd.pkl
train.headline.word2vec.pkl
train.headline.tfidf.pkl
train.headline.svd.pkl
train.headline.ner.pkl
train.body.word2vec.pkl
train.body.tfidf.pkl
train.body.svd.pkl
train.basic.pkl
test.sim.word2vec.pkl
test.sim.tfidf.pkl
test.sim.svd.pkl
test.headline.word2vec.pkl
test.headline.tfidf.pkl
test.headline.svd.pkl
test.headline.ner.pkl
test.body.word2vec.pkl
test.body.tfidf.pkl
test.body.svd.pkl
test.basic.pkl
data.pkl
**4. Comment out line 121 in TfidfFeatureGenerator.py, then uncomment line 122 in the same file. Raw TF-IDF vectors are needed by SvdFeatureGenerator.py during feature generation, but only the similarities are needed for training.**

**5. Run xgb_train_cvBodyId_twoclass.py to train and make predictions on the test set. The output file is tree_pred_cor2.csv**

**6. Then use tree_pred_cor2.csv as input for 2ed stage RNN model to classify out agree, disagree and discuss stances for each query and article pair
Note about input data format:**
```
train_stances_processed.csv : 
	col1: Headline
	col2: Body ID
	col3: Stance
train_bodies_processed.csv :
	col1: Body ID
	col2: articleBody
test_bodies_processed.csv :
	col1: Body ID
	col2: articleBody
test_stances_unlabeled.csv :
	col1: Headline
	col2: Body ID
```

##Reference:
https://github.com/Cisco-Talos/fnc-1/tree/master/tree_model
