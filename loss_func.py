import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    A =


    And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B =

    ==========================================================================
    """
    
    u_o_transpose=tf.transpose(true_w)  #{u_o}^T 
    
    mult_u_o_t_v_c=tf.matmul(u_o_transpose,inputs) # {u_o}^T v_c

    softmax_numerator=tf.exp(mult_u_o_t_v_c) # (exp({u_o}^T v_c)
    
    softmax_denominator=tf.reduce_sum(softmax_numerator,1) # (\sum{exp({u_w}^T v_c)})
    
    A=tf.log(softmax_numerator + 1e-10 ) # log(exp({u_o}^T v_c))
    B=tf.log(softmax_denominator + 1e-10 ) # log(\sum{exp({u_w}^T v_c)})

    return tf.subtract(B,A)


# In[3]:


def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================

    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """

    w_o=tf.nn.embedding_lookup(weights, labels)
    w_x=tf.nn.embedding_lookup(weights, sample)

    shape1 = weights.shape
    shape2 = sample.shape
    embedding_size = shape1[1]
    k=shape2[0]
    
    w_o = tf.reshape(w_o, [-1, embedding_size])
    
    w_o_transpose=tf.transpose(w_o)
    w_x_transpose=tf.transpose(w_x)
    
    mult_w_o_t_v_c = tf.matmul(inputs, w_o_transpose)
    mult_q_w_t_v_c = tf.matmul(inputs, w_x_transpose)

    noise_bias=tf.nn.embedding_lookup(biases,sample)
    target_bias=tf.nn.embedding_lookup(biases,labels)
    
    noise_bias = tf.reshape(noise_bias, [-1])
    target_bias = tf.reshape(target_bias, [-1])
    
    sigma_target = tf.nn.bias_add(mult_w_o_t_v_c , target_bias)
    sigma_noise = tf.nn.bias_add(mult_q_w_t_v_c , noise_bias)

    unigram_prob_=tf.convert_to_tensor(unigram_prob, dtype=tf.float32)

    P_noise = tf.gather(unigram_prob_,sample)
    P_target = tf.gather(unigram_prob_,labels)
    P_target = tf.reshape(P_target, [-1])

    first_term =tf.subtract(sigma_target, tf.log(tf.scalar_mul(k,P_target)+1e-10))
    second_term = tf.subtract(sigma_noise,tf.log(tf.scalar_mul(k,P_noise)+1e-10))

    sigmoid_1 = tf.sigmoid(first_term)
    sigmoid_2 = tf.sigmoid(second_term)

    A = (tf.log(sigmoid_1 + 1e-10))
    B = (tf.reduce_sum(tf.log(1-sigmoid_2 + 1e-10),1))

    return -tf.add(A,B)

