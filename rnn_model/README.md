## Library Dependencies
* Python 3.5
* Scipy Stack (`numpy`, `scipy`)
* [Tensorflow 1.3](https://github.com/tensorflow/tensorflow/releases)
* [NLTK (python NLP library)](http://www.nltk.org)

## Procedure
**1. Install all the dependencies.**

**2. Download the GloVe word embedding from https://nlp.stanford.edu/projects/glove, this should be present in ./glove.6B, the result reported is using the following one: Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download): glove.840B.300d.zip**

**3. Run preprocessing code to generate the input to the model using the dataset:**
```
python3 preprocessing.py
```

**4. Run training loop to do the training:**
```
python3 train.py
```

**5. Change the "train" flag to False (line 16 in train.py) and run prediction to store the prediction:**
```
python3 train.py
```

**6. Run post processing code to generate the result in csv file:**
```
cd ./evalution
python3 nDCG.py
```

**7. (Optional) You could get the fnc winner's results by the following step:**
```
cd ./evaluation
```
**Open nDCG.py. Comment out line 13, uncomment line 12 (Since the file format used in fnc winner's implementation is different). Comment out line 163-164, uncomment line 167-169.**
```
python2 nDCG.py (Note this is python2 instead of python3)
```