import numpy as np
import tensorflow as tf
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
#tf.logging.set_verbosity(tf.logging.INFO)

def getImages(file_name):
    fp = open(file_name,'rb')
    images = pickle.load(fp)
    fp.close()
    return images

def getLabels(file_name):
    fp = open(file_name,'rb')
    labels = pickle.load(fp)
    fp.close()
    return labels

def cnn_model_fn(features, labels, mode):
    #Input Layer
    input_layer = tf.reshape(features["x"], [-1, 128, 128, 3])

    #Convolutional Layer 1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)

    #Convolutional Layer 2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    #Convolutional Layer 3
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2)

    #Convolutional layer 4
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[5,5],
        padding="same",
        activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=2)
    # Dense Layer 1
    layer_shape = pool4.get_shape()
    num_features = layer_shape[1:4].num_elements()
    pool4_flat = tf.reshape(pool4, [-1, num_features])
    dense1 = tf.layers.dense(inputs=pool4_flat, units=1024, activation=tf.nn.relu)

    # Dense Layer2
    dense2 = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense2, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    
    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=36)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax")
        }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=36)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Operation (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    #Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    train_data = np.asarray(getImages("./train_data/train_images"),dtype=np.float32)
    train_labels = np.asarray(getLabels("./train_data/train_labels"))
    eval_data = np.asarray(getImages("./test_data/test_images"),dtype=np.float32)
    eval_labels = np.asarray(getLabels("./test_data/test_labels"))

    # Create Estimator
    classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=500)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    classifier.train(
        input_fn=train_input_fn,
        steps=10000,
        hooks=[logging_hook])

    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)
    print(eval_results.keys())

if __name__ =="__main__":
    tf.app.run()
