##################################################################################################
##################################################################################################
##################################################################################################
#### 																						######
####				Cross Entropy Implementation											######
####																						######
##################################################################################################
##################################################################################################
##################################################################################################

inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].


# Write the equation in the tf.
# A = log(exp({u_o}^T v_c))
# B = log(\sum{exp({u_w}^T v_c)})

# make the transpose of the embadding
u_o_transpose=tf.transpose(true_w)  #{u_o}^T 

# do matrix multiplication with inputs
mult_u_o_t_v_c=tf.matmul(u_o_transpose,inputs) # {u_o}^T v_c

# the exponential of these as softmax_numerator
softmax_numerator=tf.exp(mult_u_o_t_v_c) # (exp({u_o}^T v_c)

# reduce_sum on the softmax_numerator and generate the softmax_denominator
softmax_numerator=tf.reduce_sum(softmax_numerator,1) # (\sum{exp({u_w}^T v_c)})

# take log of the both softmax_numerator and softmax_numerator with +1e-10 for avoiding log(0).
A=tf.log(softmax_numerator + 1e-10 ) # log(exp({u_o}^T v_c))
B=tf.log(softmax_denominator + 1e-10 ) # log(\sum{exp({u_w}^T v_c)})

return tf.subtract(B,A)

##################################################################################################
##################################################################################################
##################################################################################################
#### 																						######
####				Noise Contrastive Estimation 											######
####																						######
##################################################################################################
##################################################################################################
##################################################################################################

inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
biases: Biases for nce loss. Dimension is [Vocabulary, 1].
labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
samples: Word_ids for negative samples. Dimension is [num_sampled].
unigram_prob: Unigram probability. Dimesion is [Vocabulary].

# First calculate the target and noise words embedding by labels and samples 
# multiplying with the weights. Then, to compute score, these two values are
# transposed and multiply with the context embedding. Biases for the target and
# noise is calculated from the unigram probability. Next, using the equations
# in the pdf tensor operators are applied to the respective variables. 

# Get the embadding for the target and noise
w_o=tf.nn.embedding_lookup(weights, labels)
w_x=tf.nn.embedding_lookup(weights, sample)

shape1 = weights.shape
shape2 = sample.shape
embedding_size = shape1[1]
k=shape2[0]

w_o = tf.reshape(w_o, [-1, embedding_size])

# Get the transpose matrix
w_o_transpose=tf.transpose(w_o)
w_x_transpose=tf.transpose(w_x)

# Multiply with the context embedding to get the score
mult_w_o_t_v_c = tf.matmul(inputs, w_o_transpose)
mult_q_w_t_v_c = tf.matmul(inputs, w_x_transpose)

# Generate nosie for the both target and noise
noise_bias=tf.nn.embedding_lookup(biases,sample)
target_bias=tf.nn.embedding_lookup(biases,labels)

noise_bias = tf.reshape(noise_bias, [-1])
target_bias = tf.reshape(target_bias, [-1])

# Add biases
sigma_target = tf.nn.bias_add(mult_w_o_t_v_c , target_bias)
sigma_noise = tf.nn.bias_add(mult_q_w_t_v_c , noise_bias)

# Get the unigram probability in tensor
unigram_prob_=tf.convert_to_tensor(unigram_prob, dtype=tf.float32)

# Calculate the unigram probabiloity for the target and noise 
P_noise = tf.gather(unigram_prob_,sample)
P_target = tf.gather(unigram_prob_,labels)
P_target = tf.reshape(P_target, [-1])

# Equation calculated as pdf
first_term =tf.subtract(sigma_target, tf.log(tf.scalar_mul(k,P_target)+1e-10))
second_term = tf.subtract(sigma_noise,tf.log(tf.scalar_mul(k,P_noise)+1e-10))

# Sigmoid applied
sigmoid_1 = tf.sigmoid(first_term)
sigmoid_2 = tf.sigmoid(second_term)

A = (tf.log(sigmoid_1 + 1e-10))
B = (tf.reduce_sum(tf.log(1-sigmoid_2 + 1e-10),1))

return -tf.add(A,B)

##################################################################################################
##################################################################################################
##################################################################################################
#### 																						######
####				Word Analogy				 											######
####																						######
##################################################################################################
##################################################################################################
##################################################################################################

textpath = 'word_analogy_dev.txt'
output_path = file_name+'.txt'  

f = open(output_path, 'w')

with open(textpath, 'r+') as file:
    for i,data in enumerate(file):
        qsn = []    
        qry = []            
        output = []
        vec2 = []
		
# First, split the left side and right side of the input. 
        
		example = data.split('||')
		
# Get the word vector using embeddings[word_id_w1] for every pair (a,b), take the difference in the vector (b-a) and append to a list. 
        for pairs in example[0].split(","):
            
            w1,w2 = pairs.split(":")
            
            w1 = w1.replace('"',' ').strip()
            w2 = w2.replace('"',' ').strip()

            word_id_w1 = dictionary[w1]
            word_id_w2 = dictionary[w2]

            a = np.array(embeddings[word_id_w1]) 
            b = np.array(embeddings[word_id_w2]) 
            vec2.append(b)
            
            diff = b-a
            
            qsn.append(diff)
            #print(w1,w2)
        
        #print(qsn)
# Next take the mean of these difference vectors. 
        vec_qsn = np.mean(qsn,axis=0)
        
        qry = []
        can = []
# Now, for the right side, query relation, again get the vector and take the difference. 

        for i,pairs in enumerate(example[1].split(",")):
            
            w1,w2 = pairs.split(":")
            
            w1 = w1.replace('"',' ').strip()
            w2 = w2.replace('"',' ').strip()

            word_id_w1 = dictionary[w1]
            word_id_w2 = dictionary[w2]
            
            a = np.array(embeddings[word_id_w1]) 
            b = np.array(embeddings[word_id_w2]) 
            
            ab = w1+":"+w2
            diff = a - b
            qry.append((ab,diff))
            #print(diff)
            output.append(w1+":"+w2)
        
            
        max_dist=-float("inf")
        min_dist=float("inf")
		
# Now calculate the cosine similarity between this query vector and the mean vector.        
        
		for k,b in qry:
            dist = 1 - sp.distance.cosine(vec_qsn,b)
            
            if dist<min_dist:
                min_dist=dist
                ls = k

            if dist>max_dist:
                max_dist = dist
                ms = k
# The minimum and maximum values are most and least illustrative.

        output.append(ls)
        output.append(ms)
        
        for o in output:
            f.write("\""+ o +"\" ")#.replace("\'","\""))
        f.write("\n")
f.close()
print()


##################################################################################################
##################################################################################################
##################################################################################################
#### 																						######
####				Generate Batch 				 											######
####																						######
##################################################################################################
##################################################################################################
##################################################################################################

temp_batch_size = batch_size
context_list = [] # Generate context words
neighbour_list = [] # Heghbouring owrds of the context word
left = data_index # Starting point
width = (2 * skip_window)+1 # Size of the window
right = left+width # Ending poinr

while(batch_size > 0): # Keep repeating untill batchsize
	window = []
	
	for j in range(left, right): # Generate the small window based on skip_window
		window.append(data[j]) 
	
	middle = (right-left)//2
	left += 1
	right = width + left
	context_word = window[middle] # Context word
	
	for i in range(len(window)): # For context word, generate left and right sides neighbouring words
		if(i != middle):
			context_list.append(context_word)
			neighbour_list.append(window[i])
			batch_size = batch_size - 1 # decrease the batch_size

batch = np.array(context_list)
labels = np.array(neighbour_list).reshape(temp_batch_size, 1)
return batch,labels

First, I generate all the intermideate data window.
Then data in each window, I split the context word and left/right words.
I repeadetly add the neighbouring words with the context word in a list.
Everytime a new neighbouring word is added, i decrease the batch_size.
Keep repeating until batch_size is zero.


##################################################################################################
##################################################################################################
##################################################################################################
#### 																						######
####				Configuration for the Word Analogy										######
####																						######
##################################################################################################
##################################################################################################
##################################################################################################

1. Cross Entropy: 

Batch size: 64
Embedded size: 128
Skip window: 32
Number of skips: 16 
Number of steps: 400001


*****************************************

2. Noise Contrastive Estimation

Batch size: 512
Embedded size: 128
Skip window: 1
Number of skips: 2 
Number of steps: 500001


##################################################################################################
##################################################################################################
##################################################################################################
#### 																						######
####				Configuration for the top 20 words										######
####																						######
##################################################################################################
##################################################################################################
##################################################################################################

1. Cross Entropy: 

Batch Size: 128
Embedded Size: 128 
Skip Window: 2
# of Skips: 2 
# of Steps: 400001


*****************************************

2. Noise Contrastive Estimation

Batch Size: 512
Embedded Size: 128
Skip Window: 8
# of Skips: 16
# of Steps: 400001