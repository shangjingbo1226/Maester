import csv
import math
from process import rnn_score, combine_files

gain = {"agree":{"disagree":0, "unrelated":0, "discuss":0, "agree":1},
        "disagree":{"agree":0, "unrelated":0, "discuss":0, "disagree":1},
        "discuss":{"agree":1, "disagree":1, "discuss":1, "unrelated":0}}

# read data from csv
def read_data(file, keep=False):
    data = []
    # with open(file, "r") as csvfile:
    with open(file, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append([row["Headline"], row["Body ID"], float(row["Score"]),
                         row["Stance Predicted"], row["Stance Label"]])

    if keep:
        valid_headlines = {}
        for i in data:
            headline = i[0]
            if i[-1] in ["agree", "disagree"]:
                if headline not in valid_headlines:
                    valid_headlines[headline] = set([i[-1]])
                else:
                    valid_headlines[headline].add(i[-1])
        data = [i for i in data if i[0] in valid_headlines and len(valid_headlines[i[0]]) == 2]
        print("num of valid_queries: {}".format(len([i for i in valid_headlines if len(valid_headlines[i]) == 2])))
    
    query_document_pairs = {}
    for i in data:
        headline = i[0]
        if headline not in query_document_pairs:
            query_document_pairs[headline] = [i]
        else:
            query_document_pairs[headline].append(i)
            
    for headline in query_document_pairs:
        tmp = set()
        for i in query_document_pairs[headline]:
            tmp.add(i[-1])
        if keep:
            assert "agree" in tmp
            assert "disagree" in tmp, query_document_pairs[headline]
        
    return query_document_pairs, data

def prepare_data(data):    
    labels = ["agree", "disagree", "discuss", "unrelated"]
    predicted_data = {label:[] for label in labels}    
    # prepare data for DCG
    for i in data:
        predicted_data[i[3]].append(i)        
    # get the ranked list
    for i in predicted_data:
        predicted_data[i].sort(key=lambda x:x[2], reverse=True)        
    return predicted_data

def calculate_DCG(label, data, gain, k):
    if len(data) == 0:
        return 0
    res = gain[label][data[0][4]]
    for i in range(1, min(len(data), k)):
        res += gain[label][data[i][4]] / math.log(i+1)
    return res

def calculate_nDCG(label, predicted_data, real_data, gain, k):
    # sort the documents w.r.t gain
    real_data.sort(key=lambda x:gain[label][x[4]], reverse=True)
    try:
        return calculate_DCG(label, predicted_data, gain, k) / calculate_DCG(label, real_data, gain, k)
    except:
        return -1

def score(data):
    score = 0.
    acc = 0.
    max_score = 0.
    relate_acc = 0.
    stance_acc = 0.
    num_stance = 0.
    for i in data:
        if i[-1] == i[-2]:
            acc += 1
            if i[-1] == "unrelated":
                score += 0.25
                relate_acc += 1
            else:
                score += 0.75
        if i[-1] != "unrelated" and i[-2] != "unrelated":
            score += 0.25
            relate_acc += 1
            
        if i[-1] == "unrelated":
            max_score += 0.25
        else:
            max_score += 1
                
        if i[-2] != "unrelated":
            num_stance += 1
            if i[-1] == i[-2]:
                stance_acc += 1
    print("accuracy: {}, relatedness acc: {}, stance accuracy: {}, score: {}, relative score: {}".format(acc / len(data), relate_acc / len(data), stance_acc / num_stance, score, score / max_score))
    return stance_acc / num_stance, score / max_score

def get_score(num_sent, valid_only=False, epochs=10):
    ndcg_agree = []
    ndcg_disagree = []
    ndcg_discuss = []
    weighted_score = []
    stance_score = []
    for i in range(epochs):
        data, raw = read_data("./res/res_{}_{}.csv".format(num_sent, i+1), valid_only)
        sscore, wscore = score(raw)
        stance_score.append(sscore)
        weighted_score.append(wscore)
    
        for k in [3, 5]:
            res = {i:[] for i in gain}
            for i in data:
                predicted_data = prepare_data(data[i])
                for label in gain:
                    res[label].append(calculate_nDCG(label, predicted_data[label], data[i], gain, k))           
            for i in res:
                tmp = [j for j in res[i] if j != -1]
                print("nDCG@{:5}\tfor class {}:\t".format(k, i), sum(tmp)/len(tmp))
                if k == 3:
                    if i == "agree":
                        ndcg_agree.append(sum(tmp)/len(tmp))
                    if i == "disagree":
                        ndcg_disagree.append(sum(tmp)/len(tmp))
                if k == 5 and i == "discuss":
                    ndcg_discuss.append(sum(tmp)/len(tmp))
    
    return [ndcg_agree, ndcg_disagree, ndcg_discuss, weighted_score, stance_score]

def generate_result(f, valid_only):
    num_sent = [1, 2, 3, 4, 5]
    seed = ["1_150", "2_471", "3_466"]
    th = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    with open("res/{}.csv".format(f), "w", newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(["num_epoch"] + [i+1 for i in range(20)])
        for n in num_sent:
            for s in seed:
                for t in th:
                    try:
                        rnn_score("tree_two_class.csv", "./../prediction/top{}_seed_{}/prediction_{}".format(n, s, n), "competition_test_stances.csv", n, 20, t)
                        ndcg_agree, ndcg_disagree, ndcg_discuss, weighted_score, stance_score = get_score(n, valid_only, 20)
                        writer.writerow(["num_sent: {}, seed: {}, th: {}".format(n, s, t)])
                        writer.writerow(["weighted accuracy"] + weighted_score)
                        writer.writerow(["stance accuracy"] + stance_score)
                        writer.writerow(["ndcg agree@3"] + ndcg_agree)
                        writer.writerow(["ndcg disagree@3"] + ndcg_disagree)
                        writer.writerow(["ndcg discuss@5"] + ndcg_discuss)
                        writer.writerow([])
                    except:
                        print("no such files, n: {}, s: {}".format(n, s))

# generate result of our model
generate_result("exact_results", False)
generate_result("exact_results_opinion", True)

# generate result of fnc winner
#combine_files("./tree_pred_prob_cor2.csv", "./deepoutput.csv", "competition_test_stances.csv", "./res/res_0_1.csv", "tree_two_class.csv", False)
#ndcg_agree, ndcg_disagree, ndcg_discuss, weighted_score, stance_score = get_score(0, False, 1)
#ndcg_agree, ndcg_disagree, ndcg_discuss, weighted_score, stance_score = get_score(0, True, 1)