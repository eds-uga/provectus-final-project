import tempfile
import sys
import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir", "/home/dharamendra/model", "Base directory for output models.")
flags.DEFINE_string("model_type", "wide_n_deep",
                    "Valid model types: {'wide', 'deep', 'wide_n_deep'}.")
flags.DEFINE_integer("train_steps", 100, "Number of training steps.")
flags.DEFINE_integer("BATCH_SIZE", 10000, "Number of examples iin a batch")
flags.DEFINE_string("train_data","","Path to the training data.")
flags.DEFINE_string("test_data","","Path to the test data.")

COLUMNS = ["index","label","I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12",
           "I13","C1","C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13",
           "C14","C15", "C16", "C17", "C18", "C19","C20", "C21", "C22", "C23", "C24", "C25", "C26"]

train_data="criteo.csv"
BATCH_SIZE=5000

LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = {"C1","C2", "C3", "C4", "C5", "C6", "C7","C8", "C9", "C10", "C11", "C12", "C13", "C14",
                   "C15", "C16", "C17", "C18", "C19","C20", "C21", "C22", "C23", "C24", "C25", "C26"}

CONTINUOUS_COLUMNS = ["I1", "I2", "I3", "I4", "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13"]

def train_and_eval(train_file):
    """
    Generate tarin model for CTR
    :param train_file:
    :return:
    """

    model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
    print("model directory = %s" % model_dir)

    print("****************Starting to build the estimator**********")
    m = build_estimator(model_dir)

    print("Done Building Estimator")

    m.fit(input_fn=lambda: input_fn(BATCH_SIZE,train_data), steps=FLAGS.train_steps)


    with tf.Session() as sess:

        init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        sess = tf.Session(config=tf.ConfigProto())
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        coord.request_stop()
        coord.join(threads)

def build_estimator(model_dir):
    """
    Method to build wide and deep model
    :param model_dir:
    :return: DNNLinearCombinedClassifier model object
    """

    print("********Inside Build Estimator*******8")

    #Categorical features
    C9 = tf.contrib.layers.sparse_column_with_keys(column_name="C9",
                                                       keys=["7cc72ec2", "a18233ea", "a73ee510"], )

    C20 = tf.contrib.layers.sparse_column_with_keys(column_name="C20",
                                                       keys=["b1252a9d", "5840adea", "a458ea53"], )

    C1 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C1", hash_bucket_size=541)

    C2 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C2", hash_bucket_size=497)

    C3 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C3", hash_bucket_size=40000)

    C4 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C4", hash_bucket_size=25183)

    C5 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C5", hash_bucket_size=145)

    C6 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C6", hash_bucket_size=11)

    C7 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C7", hash_bucket_size=7623)

    C8 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C8", hash_bucket_size=257)

    C10 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C10", hash_bucket_size=10997)

    C11 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C11", hash_bucket_size=3799)

    C12 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C12", hash_bucket_size=41311)

    C13 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C13", hash_bucket_size=2796)

    C14 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C14", hash_bucket_size=26)

    C15 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C15", hash_bucket_size=5238)

    C16 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C16", hash_bucket_size=34616)

    C17 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C17", hash_bucket_size=10)

    C18 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C18", hash_bucket_size=2548)

    C19 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C19", hash_bucket_size=1302)

    C21 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C21", hash_bucket_size=38617)

    C22 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C22", hash_bucket_size=10)

    C23 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C23", hash_bucket_size=14)

    C24 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C24", hash_bucket_size=12334)

    C25 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C25", hash_bucket_size=50)

    C26 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "C26", hash_bucket_size=9526)

    #Contineous features
    I1 = tf.contrib.layers.real_valued_column("I1")
    I2 = tf.contrib.layers.real_valued_column("I2")
    I3 = tf.contrib.layers.real_valued_column("I3")
    I4 = tf.contrib.layers.real_valued_column("I4")
    I5 = tf.contrib.layers.real_valued_column("I5")
    I6 = tf.contrib.layers.real_valued_column("I6")
    I7 = tf.contrib.layers.real_valued_column("I7")
    I8 = tf.contrib.layers.real_valued_column("I8")
    I9 = tf.contrib.layers.real_valued_column("I9")
    I10 = tf.contrib.layers.real_valued_column("I10")
    I11 = tf.contrib.layers.real_valued_column("I11")
    I12 = tf.contrib.layers.real_valued_column("I12")
    I13 = tf.contrib.layers.real_valued_column("I13")

    # Wide columns and deep columns.
    wide_columns = [C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,
                    C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,
                    C25,C26,
                    tf.contrib.layers.crossed_column([C9,C20],hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([C17, C22],hash_bucket_size=int(1e6)),
                    ]

    deep_columns = [
        tf.contrib.layers.embedding_column(C1, dimension=8),
        tf.contrib.layers.embedding_column(C2, dimension=8),
        tf.contrib.layers.embedding_column(C3, dimension=8),
        tf.contrib.layers.embedding_column(C4, dimension=8),
        tf.contrib.layers.embedding_column(C5, dimension=8),
        tf.contrib.layers.embedding_column(C6, dimension=8),
        tf.contrib.layers.embedding_column(C7, dimension=8),
        tf.contrib.layers.embedding_column(C8, dimension=8),
        tf.contrib.layers.embedding_column(C9, dimension=8),
        tf.contrib.layers.embedding_column(C10, dimension=8),
        tf.contrib.layers.embedding_column(C11, dimension=8),
        tf.contrib.layers.embedding_column(C12, dimension=8),
        tf.contrib.layers.embedding_column(C13, dimension=8),
        tf.contrib.layers.embedding_column(C14, dimension=8),
        tf.contrib.layers.embedding_column(C15, dimension=8),
        tf.contrib.layers.embedding_column(C16, dimension=8),
        tf.contrib.layers.embedding_column(C17, dimension=8),
        tf.contrib.layers.embedding_column(C18, dimension=8),
        tf.contrib.layers.embedding_column(C19, dimension=8),
        tf.contrib.layers.embedding_column(C20, dimension=8),
        tf.contrib.layers.embedding_column(C21, dimension=8),
        tf.contrib.layers.embedding_column(C22, dimension=8),
        tf.contrib.layers.embedding_column(C23, dimension=8),
        tf.contrib.layers.embedding_column(C24, dimension=8),
        tf.contrib.layers.embedding_column(C25, dimension=8),
        tf.contrib.layers.embedding_column(C26, dimension=8),

        I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13]

    if FLAGS.model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif FLAGS.model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[512, 256,128,64],dnn_optimizer="Adagrad",linear_optimizer="SGD")
    return m


#################################################
# Method to read the training features in batches
#################################################


def input_fn(batch_size,file_name):
    """
    :param batch_size:
    :param file_name:
    :return: features and label dict
    """
    examples_op = tf.contrib.learn.read_batch_examples(
        file_name,
        batch_size=batch_size,
        reader=tf.TextLineReader,
        num_epochs=1,

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


def dense_to_sparse(dense_tensor):
    """
    :param dense_tensor:
    :return: Sparse categorical features
    """
    indices = tf.to_int64(tf.transpose([tf.range(tf.shape(dense_tensor)[0]), tf.zeros_like(dense_tensor, dtype=tf.int32)]))
    values = dense_tensor
    shape = tf.to_int64([tf.shape(dense_tensor)[0], tf.constant(1)])

    return tf.SparseTensor(
        indices=indices,
        values=values,
        shape=shape
    )

def main():
    train_and_eval(sys.argv[1])

if __name__ == "__main__":
    main()


