import numpy as np
import csv
import json
from nltk import sent_tokenize, word_tokenize
from scipy.spatial.distance import cosine

# read GloVe embedding file
def read_embedding(fn):
    words = {}
    stance_embeddings = {"agree":[], "disagree":[]}
    
    with open(fn, "r", encoding="utf-8") as fh:
        for line in fh:
            vector = line.split()
            try:
                words[vector[0]] = np.array([float(i) for i in vector[1:]])
                if vector[0] in ["agree", "yes", "right"]:
                    stance_embeddings["agree"].append(
                            np.array([float(i) for i in vector[1:]]))
                if vector[0] in ["disagree", "no", "wrong"]:
                    stance_embeddings["disagree"].append(
                            np.array([float(i) for i in vector[1:]]))
            except:
                continue
    
    # create stence embedding vector     
    stance_embeddings["agree"] = np.mean(stance_embeddings["agree"], axis=0)
    stance_embeddings["disagree"] = np.mean(stance_embeddings["disagree"], axis=0)
    return words, stance_embeddings


# create vacab from training/test set
def create_vocab(train_stances, train_bodies, test_stances, test_bodies):
    def read(fn, pos):
        words = set()
        with open(fn, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                for sent in sent_tokenize(row[pos]):
                    for token in word_tokenize(sent):
                        words.add(token)
        return words
            
    return list(read(train_stances, 0) | read(train_bodies, 1) |
                read(test_stances, 0) | read(test_bodies, 1))


# map vocab to embedding, preserve the mapping, and needed embeddings
def preserve_vocab_embedding(vocab, embedding, size=100):
    results = np.zeros((len(vocab), size))
    missed = 0
    mapping = {}
    for i in range(len(vocab)):
        w = vocab[i]
        w_no_punc = "".join(j for j in w if j.isalpha())
        mapping[w] = i
        if w in embedding:
            results[i,:] = embedding[w]
        elif w.lower() in embedding:
            results[i,:] = embedding[w.lower()]
        elif w.capitalize() in embedding:
            results[i,:] = embedding[w.capitalize()]
        elif w.upper() in embedding:
            results[i,:] = embedding[w.upper()]
        elif w_no_punc in embedding:
            results[i,:] = embedding[w_no_punc]
        elif w_no_punc.lower() in embedding:
            results[i,:] = embedding[w_no_punc.lower()]
        elif w_no_punc.capitalize() in embedding:
            results[i,:] = embedding[w_no_punc.capitalize()]
        elif w_no_punc.upper() in embedding:
            results[i,:] = embedding[w_no_punc.upper()]
        else:
            missed += 1
            
    print("{} / {} vocab words exist in GloVe".format(len(vocab)-missed, len(vocab)))
    
    np.save("./data/vocab_embedding", results)
    with open("./data/vocab_mapping.json", "w") as fh:
        json.dump(mapping, fh)
        
    return mapping, results
        

# transform words in dataset to ids in embeddings
def dataset_to_ids(mapping, train_stances, train_bodies, test_stances, test_bodies):
    def transform(fn, mapping, pos, is_body=False, triplicate_disagree=False, keep=False):
        results = []
        with open(fn + ".csv", "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                if is_body:
                    body_ids = []
                    for sent in sent_tokenize(row[pos]):
                        sent_ids = []
                        for token in word_tokenize(sent):
                            sent_ids.append(mapping[token])
                        body_ids.append(sent_ids)
                    results.append(row[:pos] + [body_ids])
                else:
                    headline_ids = []
                    for sent in sent_tokenize(row[pos]):
                        for token in word_tokenize(sent):
                            headline_ids.append(mapping[token])
                    results.append(row[:pos] + [headline_ids])
                    if pos < len(row)-1:
                        results[-1].extend(row[pos+1:])
                        
                    # transform label, triplicate rare class
                    if results[-1][-1] == "agree":
                        results[-1][-1] = [1, 0]
                    elif results[-1][-1] == "disagree":
                        results[-1][-1] = [0, 1]
                        if triplicate_disagree:
                            results.append(results[-1])
                            results.append(results[-1])
                    elif results[-1][-1] == "discuss":
                        results[-1][-1] = [0, 0]
                    elif results[-1][-1] == "unrelated" and not keep:
                        results.pop()
                    else:
                        results[-1][-1] = [-1, -1]
        if not keep:           
            with open(fn + ".json", "w") as fh:
                json.dump(results, fh)
        else:
            with open(fn + "_keep.json", "w") as fh:
                json.dump(results, fh)
                
    transform(train_stances, mapping, 0, triplicate_disagree=True)
    transform(train_bodies, mapping, 1, is_body=True)
    transform(test_stances, mapping, 0)
    transform(test_stances, mapping, 0, keep=True)
    transform(test_bodies, mapping, 1, is_body=True)


# given headline, document pair, return top k similar sentence
def most_similar_sentences(headline, body, k, embeddings, stance_embeddings):
    headline_embeddings = np.mean(np.take(embeddings, headline, axis=0), axis=0)
    s1 = sorted(body, key=lambda sent: cosine(
            headline_embeddings, np.mean(np.take(embeddings, sent, axis=0), axis=0)))
    res = [word for sent in s1[:k] for word in sent]
    
    if stance_embeddings != None:    
        s2 = max(body, key=lambda sent: cosine(
                stance_embeddings["agree"],
                np.mean(np.take(embeddings, sent, axis=0), axis=0)))
        s3 = max(body, key=lambda sent: cosine(
                stance_embeddings["disagree"],
                np.mean(np.take(embeddings, sent, axis=0), axis=0)))
        res = res + [word for word in s2] + [word for word in s3]
    return res


# transform body in dataset to top k similar sentence, preserve transformed dataset
def body_to_most_similar_sentences(train_stances_fn, train_bodies_fn, test_stances_fn,
                                   test_bodies_fn, k=3, stance_embeddings=None):
    with open(train_stances_fn + ".json", "r") as fh:
        train_stances = json.load(fh)
    with open(train_bodies_fn + ".json", "r") as fh:
        train_bodies = {i[0]:i[1] for i in json.load(fh)}
    with open(test_stances_fn + ".json", "r") as fh:
        test_stances = json.load(fh)
    with open(test_stances_fn + "_keep.json", "r") as fh:
        test_stances_keep = json.load(fh)
    with open(test_bodies_fn + ".json", "r") as fh:
        test_bodies = {i[0]:i[1] for i in json.load(fh)}
    
    embeddings = np.load("./data/vocab_embedding.npy")
    for i in train_stances:
        i[1] = most_similar_sentences(i[0], train_bodies[i[1]], k,
         embeddings, stance_embeddings)
    for i in test_stances:
        i[1] = most_similar_sentences(i[0], test_bodies[i[1]], k,
         embeddings, stance_embeddings)
    for i in test_stances_keep:
        i[1] = most_similar_sentences(i[0], test_bodies[i[1]], k,
         embeddings, stance_embeddings)
    
    with open(train_stances_fn + "_top{}.json".format(k), "w") as fh:
        json.dump(train_stances, fh)
    with open(test_stances_fn + "_top{}.json".format(k), "w") as fh:
        json.dump(test_stances, fh)
    with open(test_stances_fn + "_top{}_keep.json".format(k), "w") as fh:
        json.dump(test_stances_keep, fh)
        
def gen_seed():
    import random
    random.seed(100)
    x = [random.randint(1,1000) for _ in range(10)]
    print(x)
    
    with open("seed.json", "w") as f:
        json.dump(x, f)
    
if __name__ == "__main__":
    params = {
            "embedding_size": 300,          # choose from 50, 100, 200, 300
            "num_similar_sentence": 3,      # 999 for whole article
            "include_stance_sentence": False     # include most similar sentence to stance
            }
    # gen_seed()
    vocab = create_vocab("./data/train_stances.csv", "./data/train_bodies.csv",
                         "./data/test_stances_labeled.csv", "./data/test_bodies.csv")
    embedding, stance_embeddings = read_embedding(
            "./glove.6B/glove.840B.{}d.txt".format(params["embedding_size"]))
    mapping, _ = preserve_vocab_embedding(vocab, embedding, params["embedding_size"])
    dataset_to_ids(mapping, "./data/train_stances", "./data/train_bodies",
                   "./data/test_stances_labeled", "./data/test_bodies")
    body_to_most_similar_sentences("./data/train_stances", "./data/train_bodies",
                                   "./data/test_stances_labeled", "./data/test_bodies",
                                   params["num_similar_sentence"], stance_embeddings
                                   if params["include_stance_sentence"] else None)