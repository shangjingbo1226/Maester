import csv
import numpy as np
import json

def combine_files(f_tree, f_deep, f_actual, f_out, f_tree_bin=None, two_stage=False):
    def read(fn):
        res = []
        with open(fn, "r") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                res.append(row)
        return res

    pred_tree, pred_deep, actual = read(f_tree), read(f_deep), read(f_actual)
    if f_tree_bin:
        pred_tree_bin = read(f_tree_bin)
    labels = ["agree", "disagree", "discuss", "unrelated"]
    with open(f_out, "w") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Headline", "Body ID", "Score", "Stance Predicted", "Stance Label"])
        for i in range(len(pred_tree)):
            if two_stage:
                pred1 = np.array([float(j) for j in pred_tree[i][-4:]])
                pred2 = np.array([float(j) for j in pred_deep[i][-4:]])
                if (not f_tree_bin and np.argmax(pred1) == 3) or (f_tree_bin and pred_tree_bin[i][-1] == "unrelated"):
                    writer.writerow(actual[i][:2] + [0, "unrelated", actual[i][-1]])
                else:                    
                    pred = pred1[:-1] + pred2[:-1]
                    pos = np.argmax(pred)
                    writer.writerow(actual[i][:2] + [pred[pos], labels[pos], actual[i][-1]])
            else:
                pred = np.array([float(j) for j in pred_tree[i][-4:]]) + np.array([float(j) for j in pred_deep[i][-4:]])
                pos = np.argmax(pred)
                writer.writerow(actual[i][:2] + [pred[pos], labels[pos], actual[i][-1]])
  
def rnn_score(f_relate, f_rnn, f_actual, num_sent, epochs=10, th=0.3):
    def read(fn):
        res = []
        with open(fn, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            next(reader)
            for row in reader:
                res.append(row)
        return res

    relate, actual = read(f_relate), read(f_actual)
    rnn = []
    for i in range(epochs):
        with open(f_rnn + "_{}.json".format(i+1), "r") as fh:
            rnn.append(json.load(fh))
    
    scores = []
    for e in range(epochs):
        acc = 0.
        cnt = 0.
        with open("./res/res_{}_{}.csv".format(num_sent, e+1), "w", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["Headline", "Body ID", "Score", "Stance Predicted", "Stance Label"])
            for i in range(len(actual)):
                if relate[i][-1] == "unrelated":
                    writer.writerow(actual[i][:2] + [0, "unrelated", actual[i][-1]])
                else:
                    p1 = float(rnn[e][i][0])
                    p2 = float(rnn[e][i][1])
                    if p1 > th or p2 > th:
                        if p1 < p2:
                            writer.writerow(actual[i][:2] + [p2, "disagree", actual[i][-1]])
                            if "disagree" == actual[i][-1]:
                                acc += 1
                        else:
                            writer.writerow(actual[i][:2] + [p1, "agree", actual[i][-1]])
                            if "agree" == actual[i][-1]:
                                acc += 1
                    else:
                        writer.writerow(actual[i][:2] + [1 - np.mean([p1, p2]), "discuss", actual[i][-1]])
                        if "discuss" == actual[i][-1]:
                            acc += 1
                    cnt += 1
                    
        scores.append(acc / cnt)
    
    return scores
