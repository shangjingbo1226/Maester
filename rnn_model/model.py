import tensorflow as tf
import numpy as np
import time
import random
from src.attention_wrapper import BahdanauAttention, AttentionWrapper

class model(object):
    def __init__(self, params, embedding):
        random.seed(params["seed"])
        
        # hyperprameters
        self.keep_prob_in = params["dropout_input"]
        self.keep_prob_fc = params["dropout_fully_connected"]
        self.hidden_unit_lstm = params["hidden_unit_lstm"]
        self.hidden_unit_fc = params["hidden_unit_fully_connected"]
        
        # setup system
        self.setup_placeholder()
        self.setup_embedding(embedding)
        self.setup_model()
        self.setup_optimizer()
    
    def encoder(self):
        # get headline/document representations using LSTMs
        with tf.variable_scope("encoded_headline"):
            headline_lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_unit_lstm)
            self.encoded_headline, _ = tf.nn.dynamic_rnn(
                    headline_lstm, self.headline_embedding, self.H, dtype=tf.float32)
        
        with tf.variable_scope("encoded_document"):
            document_lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_unit_lstm)
            self.encoded_document, _ = tf.nn.dynamic_rnn(
                    document_lstm, self.document_embedding, self.D, dtype=tf.float32)
            
    def decoder(self):
        # use attention to get weight of headline conditioned on document
        with tf.variable_scope("attention"):
            attention_mechanism = BahdanauAttention(
                    self.encoded_headline.get_shape()[-1], self.encoded_headline, memory_sequence_length=self.H)
            lstm = tf.contrib.rnn.BasicLSTMCell(self.hidden_unit_lstm)
            attention = AttentionWrapper(
                    lstm, attention_mechanism, output_attention=False,
                    attention_input_fn = lambda x,y: tf.concat([x,y], axis=-1))
            weight, _ = tf.nn.dynamic_rnn(attention, self.encoded_document, dtype=tf.float32)
        
        self.weight = tf.reduce_mean(weight, axis=1)
        
    def fully_connected(self):
        # use fully connected layer to predict the final answer
        with tf.variable_scope("fully_connected"):
            h1 = tf.contrib.layers.fully_connected(self.weight, self.hidden_unit_fc, activation_fn=None)
            h2 = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.batch_norm(h1)), self.dropout_fc)
            self.logits = tf.contrib.layers.fully_connected(h2, 2, activation_fn=tf.sigmoid)
    
    def setup_placeholder(self):
        # placeholders for ids of headline and document
        self.headline = tf.placeholder(tf.int32, shape=[None,None], name="headline")
        self.document = tf.placeholder(tf.int32, shape=[None,None], name="document")
        
        # placeholders for length of headline and document
        self.H = tf.placeholder(tf.int32, shape=[None], name="H")
        self.D = tf.placeholder(tf.int32, shape=[None], name="D")
        
        # placeholders for stance (multilabel) and dropout
        self.stance = tf.placeholder(tf.int32, shape=[None,2], name="stance")
        self.dropout_in = tf.placeholder(tf.float32, shape=[], name="dropout_in")
        self.dropout_fc = tf.placeholder(tf.float32, shape=[], name="dropout_fc")
    
    def setup_embedding(self, embedding):
        _embedding = tf.Variable(embedding, name="embedding",
                                 dtype=tf.float32, trainable=False)
        headline_embedding = tf.nn.embedding_lookup(
                _embedding, self.headline, name="headline_embedding")
        document_embedding = tf.nn.embedding_lookup(
                _embedding, self.document, name="document_embedding")
        self.headline_embedding = tf.nn.dropout(headline_embedding, self.dropout_in)
        self.document_embedding = tf.nn.dropout(document_embedding, self.dropout_in)
    
    def setup_model(self):
        self.encoder()
        self.decoder()
        self.fully_connected()
    
    def setup_optimizer(self):
        # for multi-label classification, it's better to use mean square error with sigmoid
        self.loss = tf.losses.mean_squared_error(self.stance, self.logits)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
    
    def feed(self, mini_batch, train=True):
        headline, document, stance, H, D = mini_batch
        input_feed = {
                self.headline : headline,
                self.document : document,
                self.H : H,
                self.D : D,
                self.stance : stance,
                self.dropout_in : self.keep_prob_in if train else 1,
                self.dropout_fc : self.keep_prob_fc if train else 1
        }
        return input_feed
    
    def train(self, session, mini_batch):
        _, loss = session.run([self.optimizer, self.loss], self.feed(mini_batch))
        return loss
    
    def test(self, session, mini_batch):
        logits, loss = session.run([self.logits, self.loss], self.feed(mini_batch, False))
        return logits, loss
    
    def evaluate(self, session, dataset, batch_size, th):
        prob = []
        acc = 0.
        counter = {"agree":0, "disagree":0, "discuss":0}
        
        for j in range(0, len(dataset), batch_size):
            data = self.create_minibatch(dataset, batch_size, j)
            logits, loss = self.test(session, data)
            if len(dataset) - j < batch_size:
                logits = logits[:len(dataset)-j]
            for i in logits:
                prob.append(i)
            
            stances = data[2]
            for i in range(len(logits)):
                if logits[i, 0] > th or logits[i, 1] > th:
                    s = np.argmax(logits[i])
                    if stances[i, s] == 1:
                        acc += 1
                    if s == 0:
                        counter["agree"] += 1
                    elif s == 1:
                        counter["disagree"] += 1
                else:
                    counter["discuss"] += 1
                    if np.sum(stances[i]) == 0:
                        acc += 1
                    
        print("Accuracy: {:.5f} %".format(acc / len(dataset)))
        print("Counter:", counter, "\n")
        return prob
    
    def create_minibatch(self, dataset, batch_size, start):
        minibatch = list(dataset[start : start+batch_size])
        if len(minibatch) < batch_size:
            minibatch.extend(dataset[-(batch_size-len(minibatch)):])
        headline_length = max(len(h) for h,d,s in minibatch)
        document_length = max(len(d) for h,d,s in minibatch)
        headlines, documents, stances = [], [], []
        for h,d,s in minibatch:
            headlines.append(h + [0 for _ in range(headline_length-len(h))])
            documents.append(d + [0 for _ in range(document_length-len(d))])
            stances.append(s)
        return (np.array(headlines), np.array(documents), np.array(stances),
                np.array([headline_length for _ in range(batch_size)]),
                np.array([document_length for _ in range(batch_size)]))
    
    def run(self, session, dataset, num_epoch, batch_size, threshold):
        train, test = dataset
        start = time.time()
        
        # main training loop
        for i in range(num_epoch):
            random.shuffle(train)
            
            for j in range(0, len(train), batch_size):
                mini_batch = self.create_minibatch(train, batch_size, j)
                loss = self.train(session, mini_batch)
                
                # periodically monitor loss
                if j % (50 * batch_size) == 0:
                    print("epoch {} iteration {}, elapsed {:.2f} min, train loss: {:.5f}".format(
                            i, j / batch_size, (time.time()-start)/60, loss))
            
            if i % 5 == 0:
                self.evaluate(session, test, batch_size, threshold)
            tf.train.Saver().save(session, "./model/model_{}_epoch.chk".format(i+1))
