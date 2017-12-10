import tensorflow as tf

def build_NN_layer(input_layer, shape, name, actFun=tf.nn.tanh):
    W = weight_variable(shape, name + '_W')
    b = bias_variable([shape[1]], name + '_b')
    a = actFun(tf.matmul(input_layer, W) + b)
    variable_summaries(a, name + '_a')
    return W, b, a

def weight_variable(shape,name):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.1)
    W = tf.Variable(initial)
    variable_summaries(W,name)
    return W

def bias_variable(shape,name):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1, shape=shape)
    b = tf.Variable(initial)
    variable_summaries(b,name)
    return b


def variable_summaries(var,name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('Mean', mean)
    with tf.name_scope('Stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('Stddev', stddev)
    tf.summary.scalar('Max', tf.reduce_max(var))
    tf.summary.scalar('Min', tf.reduce_min(var))
    tf.summary.histogram('Histogram', var)
