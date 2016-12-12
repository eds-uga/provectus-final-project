import tensorflow as tf
from test_tensorflow import build_estimator
import sys

COLUMNS = ["index","label","I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13", "C1",
                   "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14",
                   "C15", "C16", "C17", "C18", "C19",
"C20", "C21", "C22", "C23", "C24", "C25", "C26"]

test_data="/home/ubuntu/Data/train01.csv"
LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = {"C1","C2", "C3", "C4", "C5", "C6", "C7",
                       "C8", "C9", "C10", "C11", "C12", "C13", "C14",
                   "C15", "C16", "C17", "C18", "C19",
                   "C20", "C21", "C22", "C23", "C24", "C25", "C26"}

CONTINUOUS_COLUMNS = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13"]


def input_fn_eval(batch_size,file_name):
    """
     Input function to predict the test features
    :param batch_size:
    :param file_name:
    :return: features and label dict
    """
    examples_op = tf.contrib.learn.read_batch_examples(
        file_name,
        batch_size=batch_size,
        reader=tf.TextLineReader,
        randomize_input=False,
        read_batch_size=1,
        num_threads=5,
        num_epochs=1,
        parse_fn=lambda x: tf.decode_csv(x, [tf.constant([''], dtype=tf.string)] * len(COLUMNS),field_delim=","))
    examples_dict = {}

    for i, header in enumerate(COLUMNS):
        examples_dict[header] = examples_op[:,i]


    feature_cols = {k: tf.string_to_number(examples_dict[k], out_type=tf.float32)
                    for k in CONTINUOUS_COLUMNS}

    feature_cols.update({k: dense_to_sparse(examples_dict[k])
                         for k in CATEGORICAL_COLUMNS})


    return feature_cols

def dense_to_sparse(dense_tensor):
    indices = tf.to_int64(tf.transpose([tf.range(tf.shape(dense_tensor)[0]), tf.zeros_like(dense_tensor, dtype=tf.int32)]))
    values = dense_tensor
    shape = tf.to_int64([tf.shape(dense_tensor)[0], tf.constant(1)])

    return tf.SparseTensor(
        indices=indices,
        values=values,
        shape=shape
    )

def input_fn(batch_size,file_name):
    """
    Input function creates feautre and label dict for cross-validation
    :param batch_size:
    :param file_name:
    :return: feature dict
    """
    examples_op = tf.contrib.learn.read_batch_examples(
        file_name,
        batch_size=batch_size,
        reader=tf.TextLineReader,
	num_threads=5,
        num_epochs=1,
        randomize_input=False,
        parse_fn=lambda x: tf.decode_csv(x, [tf.constant([''], dtype=tf.string)] * len(COLUMNS),field_delim=","))

    examples_dict = {}

    for i, header in enumerate(COLUMNS):
        examples_dict[header] = examples_op[:,i]


    feature_cols = {k: tf.string_to_number(examples_dict[k], out_type=tf.float32)
                    for k in CONTINUOUS_COLUMNS}

    feature_cols.update({k: dense_to_sparse(examples_dict[k])
                         for k in CATEGORICAL_COLUMNS})

    label = tf.string_to_number(examples_dict[LABEL_COLUMN], out_type=tf.int32)

    return feature_cols, label

def evaluate_and_prdict(model_dir):
    """
    Method evaluate validation dataset and predict target class for test dataset
    :param model_dir:
    :return:
    """
    m=build_estimator(model_dir=model_dir)
    results = m.evaluate(input_fn=lambda: input_fn(5000,test_data), steps=2000)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    y = m.predict(input_fn=lambda :input_fn_eval(5000,test_data),as_iterable=True)
    file_test= open("prediction_final.txt", "w")
    for x in y:
        file_test.write('%s' % x+"\n")
    with tf.Session() as sess:
        init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess = tf.Session(config=tf.ConfigProto())
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        coord.request_stop()
        coord.join(threads)
        sess.close()


def main():
    evaluate_and_prdict(sys.argv[1])

if __name__ == "__main__":
    main()
