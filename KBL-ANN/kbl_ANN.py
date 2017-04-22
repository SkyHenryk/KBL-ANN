import tensorflow as tf
import numpy as np

class KblANN():

    learning_rate = 1e-3
    training_time = 10001
    test_size = 100
    keep_prob_size = 0.8

    X_size = 242
    W_size = 300
    Y_size = 2

    model_folder_name = "model"
    log_folder_name = "logs"

    def __init__(self):

        tf.set_random_seed(777)  # reproducibility
        self.Minus_Y = self.Y_size * -1
        self.Minus_test = self.test_size * -1

    def start(self):

        self.loadInputData()
        self.designNN()
        self.runSession()

    def loadInputData(self):
        xy = np.loadtxt('data/kbo_data.csv', delimiter=',', dtype=np.float32)
        x_data = self.MinMaxScaler(xy[:, :self.Minus_Y])
        train_x_data = x_data[:self.Minus_test, :]
        train_y_data = xy[:self.Minus_test, self.Minus_Y:]
        test_x_data = x_data[self.Minus_test:, :]
        test_y_data = xy[self.Minus_test:, self.Minus_Y:]

        self.modelName = "kbl_ANN_L{learning_rate}_W{W_size}_T{training_time}".format(
            learning_rate=str(self.learning_rate), W_size=str(self.W_size), training_time=str(self.training_time))

        self.X = tf.placeholder(tf.float32, [None, self.X_size])
        self.Y = tf.placeholder(tf.float32, [None, self.Y_size])
        self.keep_prob = tf.placeholder(tf.float32)

        self.train_dict = {self.X: train_x_data, self.Y: train_y_data, self.keep_prob: self.keep_prob_size}
        self.test_dict = {self.X: test_x_data, self.Y: test_y_data, self.keep_prob: 1}


    def designNN(self):
        with tf.name_scope("layer1") as scope:
            W1 = tf.get_variable("W1", shape=[self.X_size, self.W_size], initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.random_normal([self.W_size]))
            L1 = tf.nn.relu(tf.matmul(self.X, W1) + b1)
            L1 = tf.nn.dropout(L1, keep_prob=self.keep_prob)

            self.variable_summaries(W1, "W1")
            self.variable_summaries(b1, "b1")
            self.variable_summaries(L1, "L1")

        with tf.name_scope("layer2") as scope:
            W2 = tf.get_variable("W2", shape=[self.W_size, self.W_size], initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.random_normal([self.W_size]))
            L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
            L2 = tf.nn.dropout(L2, keep_prob=self.keep_prob)
            self.variable_summaries(W2, "W2")
            self.variable_summaries(b2, "b2")
            self.variable_summaries(L2, "L2")

        with tf.name_scope("layer3") as scope:
            W3 = tf.get_variable("W3", shape=[self.W_size, self.W_size], initializer=tf.contrib.layers.xavier_initializer())
            b3 = tf.Variable(tf.random_normal([self.W_size]))
            L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
            L3 = tf.nn.dropout(L3, keep_prob=self.keep_prob)
            self.variable_summaries(W3, "W3")
            self.variable_summaries(b3, "b3")
            self.variable_summaries(L3, "L3")

        with tf.name_scope("layer4") as scope:
            W4 = tf.get_variable("W4", shape=[self.W_size, self.W_size], initializer=tf.contrib.layers.xavier_initializer())
            b4 = tf.Variable(tf.random_normal([self.W_size]))
            L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
            L4 = tf.nn.dropout(L4, keep_prob=self.keep_prob)
            self.variable_summaries(W4, "W4")
            self.variable_summaries(b4, "b4")
            self.variable_summaries(L4, "L4")

        with tf.name_scope("layer5") as scope:
            W5 = tf.get_variable("W5", shape=[self.W_size, self.W_size], initializer=tf.contrib.layers.xavier_initializer())
            b5 = tf.Variable(tf.random_normal([self.W_size]))
            L5 = tf.nn.relu(tf.matmul(L4, W5) + b5)
            L5 = tf.nn.dropout(L5, keep_prob=self.keep_prob)
            self.variable_summaries(W5, "W5")
            self.variable_summaries(b5, "b5")
            self.variable_summaries(L5, "L5")

        with tf.name_scope("layer6") as scope:
            W6 = tf.get_variable("W6", shape=[self.W_size, self.Y_size], initializer=tf.contrib.layers.xavier_initializer())
            b6 = tf.Variable(tf.random_normal([self.Y_size]))
            self.hypothesis = tf.matmul(L5, W6) + b6
            self.variable_summaries(W6, "W6")
            self.variable_summaries(b6, "b6")
            self.variable_summaries(self.hypothesis, "hypothesis")

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.hypothesis, labels=self.Y))
        tf.summary.scalar('cost', self.cost)
        self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def runSession(self):

        with tf.Session() as sess:
            saver = tf.train.Saver()
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(f"{self.log_folder_name}/{self.modelName}")
            writer.add_graph(sess.graph)  # Show the graph

            sess.run(tf.global_variables_initializer())

            prediction = tf.arg_max(self.hypothesis, 1)
            is_correct = tf.equal(prediction, tf.arg_max(self.Y, 1))
            accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
            tf.summary.scalar('accuracy', accuracy)

            for step in range(self.training_time):

                c, t, h, summary = sess.run([self.cost, self.train, self.hypothesis, merged_summary], feed_dict=self.train_dict)
                if step % 100 == 0:
                    writer.add_summary(summary, global_step=step)
                    print(step, c)

            print('Learning Finished!')
            print("Prediction: ", sess.run(prediction, feed_dict=self.test_dict))
            print("Accuracy: ", sess.run(accuracy, feed_dict=self.test_dict))

            saver.save(sess, f"{self.model_folder_name}/{self.modelName}.ckpt")


    def MinMaxScaler(self, data):
        numerator = data - np.min(data, 0)
        denominator = np.max(data, 0) - np.min(data, 0)
        return numerator / (denominator + 1e-7)

    def variable_summaries(self, var,name):
      with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

if __name__ == "__main__":
    KblANN().start()
# tensorboard --logdir=./logs/kbl_ANN_L0.0001_W1000_T10001