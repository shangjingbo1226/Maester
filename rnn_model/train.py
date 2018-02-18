import tensorflow as tf
import numpy as np
import json
from model import model

def main(_):
    params = {
            "num_sent":3,
            "dropout_input": 0.8,
            "dropout_fully_connected": 0.5,
            "hidden_unit_lstm": 150,
            "hidden_unit_fully_connected": 1024,
            "num_epoch": 20,
            "batch_size": 30,
            "threshold": 0.5,
            "train": True,
            "seed": 150
    }
    
    # load data from files
    with open("./data/train_stances_top{}.json".format(params["num_sent"]), "r") as fh:
        train = json.load(fh)
    with open("./data/test_stances_labeled_top{}.json".format(params["num_sent"]), "r") as fh:
        test = json.load(fh)
    embedding = np.load("./data/vocab_embedding.npy")
    
    # training
    tf.reset_default_graph()
    tf.set_random_seed(params["seed"])
    clf = model(params, embedding)
    
    with tf.Session() as sess:
        if params["train"]:
            sess.run(tf.global_variables_initializer())
            clf.run(sess, (train, test), params["num_epoch"],
                    params["batch_size"], params["threshold"])
        else:
            with open("./data/test_stances_labeled_top{}_keep.json".format(params["num_sent"]), "r") as fh:
                dataset = json.load(fh)
            for e in range(params["num_epoch"]):
                tf.train.Saver().restore(sess, "./model/model_{}_epoch.chk".format(e+1))
                prob = clf.evaluate(sess, dataset, params["batch_size"], params["threshold"])
                print("Made {} predictions".format(len(prob)))
                with open("./prediction/prediction_{}_{}.json".format(params["num_sent"], e+1), "w") as fh:
                    json.dump([list([float(i) for i in p]) for p in prob], fh)

if __name__ == "__main__":
    tf.app.run()