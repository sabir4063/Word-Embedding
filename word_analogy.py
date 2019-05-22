import os
import pickle, math
import numpy as np
from scipy import spatial as sp
from sklearn.metrics.pairwise import cosine_similarity


model_path = './models/'
loss_model = 'cross_entropy'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

textpath = 'word_analogy_test.txt'
output_path = 'word_analogy_test_predictions_cross_entropy'+'.txt'  

f = open(output_path, 'w')

with open(textpath, 'r+') as file:
    for i,data in enumerate(file):
#         if i==10:
#             break
#         print(data)
        qsn = []    
        qry = []            
        output = []
        vec2 = []
        
        example = data.split('||')
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
        vec_qsn = np.mean(qsn,axis=0)
        
        qry = []
        can = []
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
        
        for k,b in qry:
            #dist = 1 - sp.distance.cosine(vec_qsn,b)
            dist = cos_sim(vec_qsn,b)
            
            if dist<min_dist:
                min_dist=dist
                ls = k

            if dist>max_dist:
                max_dist = dist
                ms = k
    
        output.append(ls)
        output.append(ms)
        
        for o in output:
            f.write("\""+ o +"\" ")#.replace("\'","\""))
        f.write("\n")
f.close()
print()