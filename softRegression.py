#softmax回归
import tensorflow as tf
import input_data

if __name__ == "__main__":
    #导入数据
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    #画图
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    #模型
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    #损失函数lost()
    y_ = tf.placeholder("float", [None, 10])
    #交叉熵损失函数
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    #训练
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    #提交tensorflow执行
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    #开始让tensorflow训练
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


    #构建评估模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    #提交tensorflow执行
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}), sess.run(W),
          sess.run(b))