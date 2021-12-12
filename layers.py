import tensorflow as tf
import tensorflow_addons as tfa

def instance_norm(inputs, activation=None) :
    instance_norm = tfa.layers.InstanceNormalization()(
                                    inputs)
    return instance_norm

def gated_linear_unit(inputs,gates) :
    glu = tf.multiply(inputs,tf.sigmoid(gates))
    return glu

def downsample_1d(inputs, filters, kernel_size, strides) :
    conv = conv1d_layer(inputs,filters,kernel_size,strides=strides)
    conv_norm = instance_norm(conv)
    
    gates = conv1d_layer(inputs,filters,kernel_size,strides=strides)
    gates_norm = instance_norm(gates)
    
    glu = gated_linear_unit(conv_norm,gates_norm)
    return glu

def residual_block(inputs,filters,kernel_size,strides) :
    
    conv1_glu = downsample_1d(inputs, filters, kernel_size, strides)
    
    conv2 = conv1d_layer(conv1_glu, filters // 2, kernel_size, strides)
    conv2_norm = instance_norm(conv2)
    
    conv_sum = tf.add(inputs,conv2_norm)
    return conv_sum


def upsample_1d(inputs, filters, kernel_size, strides) :
    conv1 = conv1d_layer(inputs, filters, kernel_size, strides)
    conv1_pixel_shuffle = pixel_shuffler(conv1)
    conv1_norm = instance_norm(conv1_pixel_shuffle)
    
    gates = conv1d_layer(inputs, filters, kernel_size, strides)
    gates_pixel_shuffle = pixel_shuffler(conv1)
    gates_norm = instance_norm(gates_pixel_shuffle)
    
    glu = gated_linear_unit(conv1_norm,gates_norm)
    return glu


def gated_linear_layer(inputs, gates, name = None):
    
    activation = tf.multiply(x = inputs, y = tf.sigmoid(gates), name = name)
    
    return activation

def instance_norm_layer(
    inputs, 
    epsilon = 1e-06, 
    activation_fn = None, 
    name = None):

    instance_norm_layer = tfa.layers.InstanceNormalization(
        epsilon = epsilon)(inputs)

    return instance_norm_layer

def conv1d_layer(
    inputs, 
    filters, 
    kernel_size, 
    strides = 1, 
    padding = 'same', 
    activation = None,
    kernel_initializer = None,
    name = None):

    conv_layer = tf.compat.v1.layers.conv1d(
        inputs = inputs,
        filters = filters,
        kernel_size = kernel_size,
        strides = strides,
        padding = padding,
        activation = activation,
        kernel_initializer = kernel_initializer,
        name = name)

    return conv_layer

def residual1d_block(
    inputs, 
    filters = 1024, 
    kernel_size = 3, 
    strides = 1,
    name_prefix = 'residual1d_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')
    h2 = conv1d_layer(inputs = h1_glu, filters = filters // 2, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h2_conv')
    h2_norm = instance_norm_layer(inputs = h2, activation_fn = None, name = name_prefix + 'h2_norm')
    
    h3 = inputs + h2_norm

    return h3

def downsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    name_prefix = 'downsample1d_block_'):

    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_norm = instance_norm_layer(inputs = h1, activation_fn = None, name = name_prefix + 'h1_norm')
    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')
    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def upsample1d_block(
    inputs, 
    filters, 
    kernel_size, 
    strides,
    shuffle_size = 2 ,
    name_prefix = 'upsample1d_block_'):
    
    h1 = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_conv')
    h1_shuffle = pixel_shuffler(inputs = h1, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle')
    h1_norm = instance_norm_layer(inputs = h1_shuffle, activation_fn = None, name = name_prefix + 'h1_norm')

    h1_gates = conv1d_layer(inputs = inputs, filters = filters, kernel_size = kernel_size, strides = strides, activation = None, name = name_prefix + 'h1_gates')
    h1_shuffle_gates = pixel_shuffler(inputs = h1_gates, shuffle_size = shuffle_size, name = name_prefix + 'h1_shuffle_gates')
    h1_norm_gates = instance_norm_layer(inputs = h1_shuffle_gates, activation_fn = None, name = name_prefix + 'h1_norm_gates')

    h1_glu = gated_linear_layer(inputs = h1_norm, gates = h1_norm_gates, name = name_prefix + 'h1_glu')

    return h1_glu

def pixel_shuffler(inputs, shuffle_size = 2, name = None):
    
    if shuffle_size == 2:
        n = tf.shape(inputs)[0]
        w = tf.shape(inputs)[1]
        c = inputs.get_shape().as_list()[2]

        oc = c // shuffle_size
        ow = w * shuffle_size
        outputs = tf.reshape(tensor = inputs, shape = [n, ow, oc], name = name)
    else:
        outputs = inputs

    return outputs
    
