import tensorflow as tf
import math
from . import utils


class TFFMCore():
    """
    This class implements underlying routines about creating computational graph.

    Its required `n_features` to be set at graph building time.


    Parameters
    ----------
    order : int, default: 2
        Order of corresponding polynomial model.
        All interaction from bias and linear to order will be included.

    rank : int, default: 5
        Number of factors in low-rank appoximation.
        This value is shared across different orders of interaction.

    input_type : str, 'dense' or 'sparse', default: 'dense'
        Type of input data. Only numpy.array allowed for 'dense' and
        scipy.sparse.csr_matrix for 'sparse'. This affects construction of
        computational graph and cannot be changed during training/testing.

    loss_function : function: (tf.Op, tf.Op) -> tf.Op, default: None
        Loss function.
        Take 2 tf.Ops: outputs and targets and should return tf.Op of loss
        See examples: .utils.loss_mse, .utils.loss_logistic

    optimizer : tf.train.Optimizer, default: AdamOptimizer(learning_rate=0.01)
        Optimization method used for training

    reg : float, default: 0
        Strength of L2 regularization

    use_diag : bool, default: False
        Use diagonal elements of weights matrix or not.
        In the other words, should terms like x^2 be included.
        Ofter reffered as a "Polynomial Network".
        Default value (False) corresponds to FM.

    reweight_reg : bool, default: False
        Use frequency of features as weights for regularization or not.
        Should be usefull for very sparse data and/or small batches

    init_std : float, default: 0.01
        Amplitude of random initialization

    seed : int or None, default: None
        Random seed used at graph creating time

    use_weights: bool, default: False
        Indicates if the training data has corresponding weights

    Attributes
    ----------
    graph : tf.Graph or None
        Initialized computational graph or None

    trainer : tf.Op
        TensorFlow operation node to perform learning on single batch

    n_features : int
        Number of features used in this dataset.
        Inferred during the first call of fit() method.

    saver : tf.Op
        tf.train.Saver instance, connected to graph

    summary_op : tf.Op
        tf.merge_all_summaries instance for export logging

    b : tf.Variable, shape: [1]
        Bias term.

    w : array of tf.Variable, shape: [order]
        Array of underlying representations.
        First element will have shape [n_features, 1],
        all the others -- [n_features, rank].

    Notes
    -----
    Parameter `rank` is shared across all orders of interactions (except bias and
    linear parts).
    tf.sparse_reorder doesn't requied since COO format is lexigraphical ordered.
    This implementation uses a generalized approach from referenced paper along
    with caching.

    References
    ----------
    Steffen Rendle, Factorization Machines
        http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
    """
    def __init__(
        self, n_features, order=2, rank=2, input_type='sparse',
        loss_function=utils.loss_logistic,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.01),
        batch_size=64, reg=0, init_std=0.01,
        use_diag=False, reweight_reg=False,
        seed=None, use_weights=False, model_type='classifier'
    ):
        self.n_features = n_features
        self.order = order
        self.rank = rank
        self.use_diag = use_diag
        self.input_type = input_type
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.reg = reg
        self.reweight_reg = reweight_reg
        self.init_std = init_std
        self.seed = seed
        self.graph = None
        self.use_weights = use_weights
        self.model_type = model_type

    def init_learnable_params(self):
        self.w = [None] * self.order
        for i in range(1, self.order + 1):
            r = self.rank
            if i == 1:
                r = 1
            rnd_weights = tf.random_uniform([self.n_features, r], -self.init_std, self.init_std)
            self.w[i - 1] = tf.verify_tensor_all_finite(
                tf.Variable(rnd_weights, trainable=True, name='embedding_' + str(i)),
                msg='NaN or Inf in w[{}].'.format(i-1))
        self.b = tf.Variable(self.init_std, trainable=True, name='bias')
        tf.summary.scalar('bias', self.b)

    def init_placeholders(self):
        self.tfrecord_placeholder = tf.placeholder(tf.string, shape=[None])
        self.init_iterator()
        self.train_x, self.train_y, self.train_s = self.iterator.get_next()
        with tf.name_scope('sparse_placeholders') as scope:
            self.raw_indices = self.train_x.indices
            self.raw_values = self.train_x.values
            self.raw_shape = self.train_x.dense_shape

    def init_iterator(self):
        files = tf.data.Dataset.list_files(self.tfrecord_placeholder)
        dataset = files.interleave(
            lambda x: tf.data.TFRecordDataset(x, compression_type="ZLIB").prefetch(10),
            cycle_length=8
        )
        dataset = dataset.shuffle(buffer_size=10000, seed=self.seed)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(self.tfrecord_parser)

        self.iterator = dataset.make_initializable_iterator()

    def tfrecord_parser(self, record):
        features = tf.parse_example(
          record,
          features={
            "sparse_feature": tf.SparseFeature(
            index_key="indices", value_key="data", dtype=tf.float32, size=self.n_features),
            "label": tf.VarLenFeature(tf.float32),
            "weight": tf.VarLenFeature(tf.float32)
        })

        labels = tf.reshape(tf.sparse_tensor_to_dense(features['label']), [-1])
        weights = tf.reshape(tf.sparse_tensor_to_dense(features['weight']), [-1])

        return features['sparse_feature'], labels, weights

    def pow_matmul(self, order, pow):
        if pow not in self.x_pow_cache:
            x_pow = utils.pow_wrapper(self.train_x, pow, self.input_type)
            self.x_pow_cache[pow] = x_pow
        if order not in self.matmul_cache:
            self.matmul_cache[order] = {}
        if pow not in self.matmul_cache[order]:
            w_pow = tf.pow(self.w[order - 1], pow)
            dot = utils.matmul_wrapper(self.x_pow_cache[pow], w_pow, self.input_type)
            self.matmul_cache[order][pow] = dot
        return self.matmul_cache[order][pow]

    def init_main_block(self):
        self.x_pow_cache = {}
        self.matmul_cache = {}
        self.outputs = self.b
        with tf.name_scope('linear_part') as scope:
            contribution = utils.matmul_wrapper(self.train_x, self.w[0], self.input_type)
        self.outputs += contribution
        for i in range(2, self.order + 1):
            with tf.name_scope('order_{}'.format(i)) as scope:
                raw_dot = utils.matmul_wrapper(self.train_x, self.w[i - 1], self.input_type)
                dot = tf.pow(raw_dot, i)
                if self.use_diag:
                    contribution = tf.reshape(tf.reduce_sum(dot, [1]), [-1, 1])
                    contribution /= 2.0**(i-1)
                else:
                    initialization_shape = tf.shape(dot)
                    for in_pows, out_pows, coef in utils.powers_and_coefs(i):
                        product_of_pows = tf.ones(initialization_shape)
                        for pow_idx in range(len(in_pows)):
                            pmm = self.pow_matmul(i, in_pows[pow_idx])
                            product_of_pows *= tf.pow(pmm, out_pows[pow_idx])
                        dot -= coef * product_of_pows
                    contribution = tf.reshape(tf.reduce_sum(dot, [1]), [-1, 1])
                    contribution /= float(math.factorial(i))
            self.outputs += contribution

    def init_regularization(self):
        with tf.name_scope('regularization') as scope:
            self.regularization = 0
            with tf.name_scope('reweights') as scope:
                if self.reweight_reg:
                    counts = utils.count_nonzero_wrapper(self.train_x, self.input_type)
                    sqrt_counts = tf.transpose(tf.sqrt(tf.to_float(counts)))
                else:
                    sqrt_counts = tf.ones_like(self.w[0])
                self.reweights = sqrt_counts / tf.reduce_sum(sqrt_counts)
            for order in range(1, self.order + 1):
                node_name = 'regularization_penalty_' + str(order)
                norm = tf.reduce_mean(tf.pow(self.w[order - 1]*self.reweights, 2), name=node_name)
                tf.summary.scalar('penalty_W_{}'.format(order), norm)
                self.regularization += norm
            tf.summary.scalar('regularization_penalty', self.regularization)

    def init_loss(self):
        with tf.name_scope('loss') as scope:
            if self.use_weights:
                self.loss = self.loss_function(self.outputs, self.train_y, self.train_s)
            else:
                self.loss = self.loss_function(self.outputs, self.train_y)

            self.reduced_loss = tf.reduce_mean(self.loss)
            tf.summary.scalar('loss', self.reduced_loss)

    def init_accuracy(self):
        with tf.name_scope('accuracy') as scope:
            if self.model_type == 'classifier':
                self.predictions = tf.cast(tf.greater(self.outputs, 0), tf.float32)
            elif self.model_type == 'regressor':
                self.predictions = self.outputs
            else:
                raise Exception('Unknown model type')

            _, self.accuracy = tf.metrics.accuracy(
                self.train_y,
                self.predictions
            )

            tf.summary.scalar('accuracy', self.accuracy)

    def init_target(self):
        with tf.name_scope('target') as scope:
            self.target = self.reduced_loss + self.reg * self.regularization
            self.checked_target = tf.verify_tensor_all_finite(
                self.target,
                msg='NaN or Inf in target value',
                name='target')
            tf.summary.scalar('target', self.checked_target)

    def build_graph(self):
        """Build computational graph according to params."""
        self.graph = tf.Graph()
        self.graph.seed = self.seed
        with self.graph.as_default():
            with tf.name_scope('learnable_params') as scope:
                self.init_learnable_params()
            with tf.name_scope('input_block') as scope:
                self.init_placeholders()
            with tf.name_scope('main_block') as scope:
                self.init_main_block()
            with tf.name_scope('optimization_criterion') as scope:
                self.init_regularization()
                self.init_loss()
                self.init_accuracy()
                self.init_target()
            self.trainer = self.optimizer.minimize(self.checked_target)
            self.init_global_vars = tf.global_variables_initializer()
            self.init_local_vars = tf.local_variables_initializer()
            self.summary_op = tf.summary.merge_all()
            self.saver = tf.train.Saver()
