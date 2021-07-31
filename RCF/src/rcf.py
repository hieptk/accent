"""
Tensorflow implementation of RCF

@references:
"""
import os
import random

import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.python.ops.parallel_for.gradients import jacobian
from tensorflow_addons.activations import gelu

from .Utilis import get_relational_data

tf.disable_v2_behavior()
os.environ['TF_DETERMINISTIC_OPS'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'
tf.get_logger().setLevel('ERROR')


class MF(BaseEstimator, TransformerMixin):
    def __init__(self, num_users, num_items, num_genres, num_directors, num_actors, num_samples, num_rel_samples,
                 pretrain_flag, hidden_factor, epoch, batch_size, learning_rate, lamda_bilinear, optimizer_type, verbose,
                 layers, activation_function, keep_prob, save_file, attention_size, reg_t, random_seed=2016):
        np.random.seed(random_seed)
        random.seed(random_seed)
        # bind params to class
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.num_users = num_users
        self.num_items = num_items
        self.num_genres = num_genres
        self.num_directors = num_directors
        self.num_actors = num_actors
        self.num_samples = num_samples
        self.num_rel_samples = num_rel_samples
        self.lamda_bilinear = lamda_bilinear
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.verbose = verbose
        self.layers = layers
        self.activation_function = activation_function
        self.keep_prob = np.array(keep_prob)
        self.no_dropout = np.ones(len(keep_prob), dtype=np.int64)
        self.attention_size = attention_size
        self.reg_t = reg_t
        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        """
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        """
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.user = tf.placeholder(tf.int32, shape=[None])  # None
            self.item_pos = tf.placeholder(tf.int32, shape=[None])  # None * 1
            self.item_neg = tf.placeholder(tf.int32, shape=[None])
            self.alpha = tf.placeholder(tf.float32, shape=[None])

            # (positive part)
            self.r0_p = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt0_p = tf.placeholder(tf.float32, shape=[None])
            self.len0_p = tf.placeholder(tf.int32)

            self.r1_p = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt1_p = tf.placeholder(tf.float32, shape=[None])
            self.e1_p = tf.placeholder(tf.int32, shape=[None, None])
            self.len1_p = tf.placeholder(tf.int32)

            self.r2_p = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt2_p = tf.placeholder(tf.float32, shape=[None])
            self.e2_p = tf.placeholder(tf.int32, shape=[None, None])
            self.len2_p = tf.placeholder(tf.int32)

            self.r3_p = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt3_p = tf.placeholder(tf.float32, shape=[None])
            self.e3_p = tf.placeholder(tf.int32, shape=[None, None])
            self.len3_p = tf.placeholder(tf.int32)

            # negative part
            self.r0_n = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt0_n = tf.placeholder(tf.float32, shape=[None])
            self.len0_n = tf.placeholder(tf.int32)

            self.r1_n = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt1_n = tf.placeholder(tf.float32, shape=[None])
            self.e1_n = tf.placeholder(tf.int32, shape=[None, None])
            self.len1_n = tf.placeholder(tf.int32)

            self.r2_n = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt2_n = tf.placeholder(tf.float32, shape=[None])
            self.e2_n = tf.placeholder(tf.int32, shape=[None, None])
            self.len2_n = tf.placeholder(tf.int32)

            self.r3_n = tf.placeholder(tf.int32, shape=[None, None])
            self.cnt3_n = tf.placeholder(tf.float32, shape=[None])
            self.e3_n = tf.placeholder(tf.int32, shape=[None, None])
            self.len3_n = tf.placeholder(tf.int32)

            # relational part
            self.head1 = tf.placeholder(tf.int32, shape=[None])
            self.relation1 = tf.placeholder(tf.int32, shape=[None])
            self.tail1_pos = tf.placeholder(tf.int32, shape=[None])
            self.tail1_neg = tf.placeholder(tf.int32, shape=[None])

            self.head2 = tf.placeholder(tf.int32, shape=[None])
            self.relation2 = tf.placeholder(tf.int32, shape=[None])
            self.tail2_pos = tf.placeholder(tf.int32, shape=[None])
            self.tail2_neg = tf.placeholder(tf.int32, shape=[None])

            self.head3 = tf.placeholder(tf.int32, shape=[None])
            self.relation3 = tf.placeholder(tf.int32, shape=[None])
            self.tail3_pos = tf.placeholder(tf.int32, shape=[None])
            self.tail3_neg = tf.placeholder(tf.int32, shape=[None])

            # Variables.
            self.weights = self._initialize_weights()
            self.dropout_keep = tf.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.placeholder(tf.bool)
            # Model.
            self.user_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user),
                                                 axis=1)  # [B,1,H]
            self.pos_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.item_pos)  # [B,H]
            self.neg_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.item_neg)  # [B,H]
            self.r0_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.weights['relation_type_embeddings'], 0),
                                               axis=0)  # [1,H]
            self.r1_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.weights['relation_type_embeddings'], 1),
                                               axis=0)  # [1,H]
            self.r2_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.weights['relation_type_embeddings'], 2),
                                               axis=0)  # [1,H]
            self.r3_embedding = tf.expand_dims(tf.nn.embedding_lookup(self.weights['relation_type_embeddings'], 3),
                                               axis=0)  # [1,H]
            # user's attention for relation type
            self.relation_type_embedding = tf.expand_dims(self.weights['relation_type_embeddings'], axis=0)  # [1,4,H]
            self.type_product = tf.multiply(self.user_embedding, self.relation_type_embedding)  # [B,4,H]
            self.type_w = tf.layers.dense(self.type_product, self.attention_size, name='first_layer_attention_type',
                                          reuse=tf.AUTO_REUSE, use_bias=True, activation=gelu)  # [B,4,A]
            self.logits_type = tf.reduce_sum(
                tf.layers.dense(self.type_w, 1, name='final_layer_attention_type', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,4]
            self.logits_exp_type = tf.exp(self.logits_type)  # [B,4]
            self.exp_sum_type = tf.reduce_sum(self.logits_exp_type, axis=-1, keepdims=True)  # [B,1]
            self.attention_type = self.logits_exp_type / self.exp_sum_type  # [B,4]
            # positive part
            # r0
            self.r0_p_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r0_p)  # [B,?,H]
            self.att0_i_p = tf.layers.dense(self.pos_embedding, self.attention_size, name='w0_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att0_i_p = tf.expand_dims(self.att0_i_p, 1)  # [B,1,A]
            self.att0_j_p = tf.layers.dense(self.r0_p_embedding, self.attention_size, name='w0_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            # self.att0_e_p = tf.layers.dense(self.r0_embedding, self.attention_size, name='w0_e', reuse=tf.AUTO_REUSE,
            #                                 use_bias=False)  # [1,A]
            # self.att0_e_p = tf.expand_dims(self.att0_e_p, axis=0)  # [1,1,A]
            self.att0_sum_p = tf.add(self.att0_i_p, self.att0_j_p)
            # self.att0_sum_p = tf.add(self.att0_sum_p, self.att0_e_p)                    #[B,?,A]
            self.att0_sum_p = tf.tanh(self.att0_sum_p)  # [B,?,A]
            self.att0_losits_p = tf.reduce_sum(
                tf.layers.dense(self.att0_sum_p, 1, name='att0_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att0_losits_p = self.Mask(self.att0_losits_p, self.cnt0_p, self.len0_p,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_0_p = tf.exp(self.att0_losits_p)
            self.sum_0_p = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_0_p, axis=-1) + 1e-9, self.alpha), axis=-1)  # [B,1]
            self.att_0_p = tf.expand_dims(self.exp_0_p / self.sum_0_p, axis=1)  # [B,1,?]
            self.item_latent_0_p = tf.matmul(self.att_0_p, self.r0_p_embedding)  # [B,1,H]
            # r1
            self.r1_p_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r1_p)  # [B,?,H]
            self.att1_i_p = tf.layers.dense(self.pos_embedding, self.attention_size, name='w1_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att1_i_p = tf.expand_dims(self.att1_i_p, 1)  # [B,1,A]
            self.att1_j_p = tf.layers.dense(self.r1_p_embedding, self.attention_size, name='w1_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e1_p_embedding = tf.nn.embedding_lookup(self.weights['genre_embeddings'], self.e1_p)  # [B,?,H]
            # self.e1_p_embedding=tf.add(self.e1_p_embedding, tf.expand_dims(self.r1_embedding,axis=0))  #[B,?,H]
            self.att1_e_p = tf.layers.dense(self.e1_p_embedding, self.attention_size, name='w1_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att1_sum_p = tf.add(self.att1_i_p, self.att1_j_p)
            self.att1_sum_p = tf.add(self.att1_sum_p, self.att1_e_p)  # [B,?,A]
            self.att1_sum_p = tf.tanh(self.att1_sum_p)  # [B,?,A]
            self.att1_losits_p = tf.reduce_sum(
                tf.layers.dense(self.att1_sum_p, 1, name='att1_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att1_losits_p = self.Mask(self.att1_losits_p, self.cnt1_p, self.len1_p,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_1_p = tf.exp(self.att1_losits_p)
            self.sum_1_p = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_1_p, axis=-1) + 1e-9, self.alpha), axis=-1)  # [B,1]
            self.att_1_p = tf.expand_dims(self.exp_1_p / self.sum_1_p, axis=1)  # [B,1,?]
            self.item_latent_1_p = tf.matmul(self.att_1_p, self.r1_p_embedding)  # [B,1,H]
            # r2
            self.r2_p_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r2_p)  # [B,?,H]
            self.att2_i_p = tf.layers.dense(self.pos_embedding, self.attention_size, name='w2_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att2_i_p = tf.expand_dims(self.att2_i_p, 1)  # [B,1,A]
            self.att2_j_p = tf.layers.dense(self.r2_p_embedding, self.attention_size, name='w2_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e2_p_embedding = tf.nn.embedding_lookup(self.weights['director_embeddings'], self.e2_p)  # [B,?,H]
            # self.e2_p_embedding = tf.add(self.e2_p_embedding, tf.expand_dims(self.r2_embedding, axis=0))  # [B,?,H]
            self.att2_e_p = tf.layers.dense(self.e2_p_embedding, self.attention_size, name='w2_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att2_sum_p = tf.add(self.att2_i_p, self.att2_j_p)
            self.att2_sum_p = tf.add(self.att2_sum_p, self.att2_e_p)  # [B,?,A]
            self.att2_sum_p = tf.tanh(self.att2_sum_p)  # [B,?,A]
            self.att2_losits_p = tf.reduce_sum(
                tf.layers.dense(self.att2_sum_p, 1, name='att2_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att2_losits_p = self.Mask(self.att2_losits_p, self.cnt2_p, self.len2_p,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_2_p = tf.exp(self.att2_losits_p)
            self.sum_2_p = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_2_p, axis=-1) + 1e-9, self.alpha), axis=-1)  # [B,1]
            self.att_2_p = tf.expand_dims(self.exp_2_p / self.sum_2_p, axis=1)  # [B,1,?]
            self.item_latent_2_p = tf.matmul(self.att_2_p, self.r2_p_embedding)  # [B,1,H]
            # r3
            self.r3_p_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r3_p)  # [B,?,H]
            self.att3_i_p = tf.layers.dense(self.pos_embedding, self.attention_size, name='w3_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att3_i_p = tf.expand_dims(self.att3_i_p, 1)  # [B,1,A]
            self.att3_j_p = tf.layers.dense(self.r3_p_embedding, self.attention_size, name='w3_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e3_p_embedding = tf.nn.embedding_lookup(self.weights['actor_embeddings'], self.e3_p)  # [B,?,H]
            # self.e3_p_embedding = tf.add(self.e3_p_embedding, tf.expand_dims(self.r3_embedding, axis=0))  # [B,?,H]
            self.att3_e_p = tf.layers.dense(self.e3_p_embedding, self.attention_size, name='w3_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att3_sum_p = tf.add(self.att3_i_p, self.att3_j_p)
            self.att3_sum_p = tf.add(self.att3_sum_p, self.att3_e_p)  # [B,?,A]
            self.att3_sum_p = tf.tanh(self.att3_sum_p)  # [B,?,A]
            self.att3_losits_p = tf.reduce_sum(
                tf.layers.dense(self.att3_sum_p, 1, name='att3_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att3_losits_p = self.Mask(self.att3_losits_p, self.cnt3_p, self.len3_p,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_3_p = tf.exp(self.att3_losits_p)
            self.sum_3_p = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_3_p, axis=-1) + 1e-9, self.alpha), axis=-1)  # [B,1]
            self.att_3_p = tf.expand_dims(self.exp_3_p / self.sum_3_p, axis=1)  # [B,1,?]
            self.item_latent_3_p = tf.matmul(self.att_3_p, self.r3_p_embedding)  # [B,1,H]
            # merge all item latent in different relations
            # self.item_latent_p = tf.reduce_sum(self.item_latent_0_p, axis=1)
            self.item_latent_p = tf.concat(
                [self.item_latent_0_p, self.item_latent_1_p, self.item_latent_2_p, self.item_latent_3_p],
                axis=1)  # [B,4,H]
            self.item_latent_p = tf.reduce_sum(
                tf.matmul(tf.expand_dims(self.attention_type, axis=1), self.item_latent_p), axis=1)  # [B,H]
            self.mu_p = tf.add(tf.reduce_sum(self.user_embedding, axis=1), self.item_latent_p)
            # self.pos = tf.reduce_sum(tf.multiply(self.mu_p, self.pos_embedding), 1)
            self.pos = tf.multiply(self.mu_p, self.pos_embedding)
            self.pos = tf.nn.dropout(self.pos, rate=1 - self.dropout_keep[-1])
            for i in range(0, len(self.layers)):
                self.pos = tf.add(tf.matmul(self.pos, self.weights['layer_%d' % i]),
                                  self.weights['bias_%d' % i])  # None * layer[i] * 1
                self.pos = self.activation_function(self.pos)
                self.pos = tf.nn.dropout(self.pos, rate=1 - self.dropout_keep[i])  # dropout at each Deep layer
            self.pos = tf.matmul(self.pos, self.weights['prediction'])  # None * 1

            #################### negative parts ####################
            # r0
            self.r0_n_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r0_n)  # [B,?,H]
            self.att0_i_n = tf.layers.dense(self.neg_embedding, self.attention_size, name='w0_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att0_i_n = tf.expand_dims(self.att0_i_n, 1)  # [B,1,A]
            self.att0_j_n = tf.layers.dense(self.r0_n_embedding, self.attention_size, name='w0_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            # self.att0_e_n = tf.layers.dense(self.r0_embedding, self.attention_size, name='w0_e', reuse=tf.AUTO_REUSE,
            #                                 use_bias=False)  # [1,A]
            # self.att0_e_n = tf.expand_dims(self.att0_e_n, axis=0)  # [1,1,A]
            self.att0_sum_n = tf.add(self.att0_i_n, self.att0_j_n)
            # self.att0_sum_n = tf.add(self.att0_sum_n, self.att0_e_n)                    #[B,?,A]
            self.att0_sum_n = tf.tanh(self.att0_sum_n)  # [B,?,A]
            self.att0_losits_n = tf.reduce_sum(
                tf.layers.dense(self.att0_sum_n, 1, name='att0_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att0_losits_n = self.Mask(self.att0_losits_n, self.cnt0_n, self.len0_n,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_0_n = tf.exp(self.att0_losits_n)
            self.sum_0_n = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_0_n, axis=-1) + 1e-9, self.alpha), axis=-1)  # [B,1]
            self.att_0_n = tf.expand_dims(self.exp_0_n / self.sum_0_n, axis=1)  # [B,1,?]
            self.item_latent_0_n = tf.matmul(self.att_0_n, self.r0_n_embedding)  # [B,1,H]
            # r1
            self.r1_n_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r1_n)  # [B,?,H]
            self.att1_i_n = tf.layers.dense(self.neg_embedding, self.attention_size, name='w1_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att1_i_n = tf.expand_dims(self.att1_i_n, 1)  # [B,1,A]
            self.att1_j_n = tf.layers.dense(self.r1_n_embedding, self.attention_size, name='w1_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e1_n_embedding = tf.nn.embedding_lookup(self.weights['genre_embeddings'], self.e1_n)  # [B,?,H]
            # self.e1_n_embedding=tf.add(self.e1_n_embedding, tf.expand_dims(self.r1_embedding,axis=0))  #[B,?,H]
            self.att1_e_n = tf.layers.dense(self.e1_n_embedding, self.attention_size, name='w1_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att1_sum_n = tf.add(self.att1_i_n, self.att1_j_n)
            self.att1_sum_n = tf.add(self.att1_sum_n, self.att1_e_n)  # [B,?,A]
            self.att1_sum_n = tf.tanh(self.att1_sum_n)  # [B,?,A]
            self.att1_losits_n = tf.reduce_sum(
                tf.layers.dense(self.att1_sum_n, 1, name='att1_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att1_losits_n = self.Mask(self.att1_losits_n, self.cnt1_n, self.len1_n,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_1_n = tf.exp(self.att1_losits_n)
            self.sum_1_n = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_1_n, axis=-1) + 1e-9, self.alpha), axis=-1)  # [B,1]
            self.att_1_n = tf.expand_dims(self.exp_1_n / self.sum_1_n, axis=1)  # [B,1,?]
            self.item_latent_1_n = tf.matmul(self.att_1_n, self.r1_n_embedding)  # [B,1,H]
            # r2
            self.r2_n_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r2_n)  # [B,?,H]
            self.att2_i_n = tf.layers.dense(self.neg_embedding, self.attention_size, name='w2_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att2_i_n = tf.expand_dims(self.att2_i_n, 1)  # [B,1,A]
            self.att2_j_n = tf.layers.dense(self.r2_n_embedding, self.attention_size, name='w2_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e2_n_embedding = tf.nn.embedding_lookup(self.weights['director_embeddings'], self.e2_n)  # [B,?,H]
            # self.e2_n_embedding = tf.add(self.e2_n_embedding, tf.expand_dims(self.r2_embedding, axis=0))  # [B,?,H]
            self.att2_e_n = tf.layers.dense(self.e2_n_embedding, self.attention_size, name='w2_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att2_sum_n = tf.add(self.att2_i_n, self.att2_j_n)
            self.att2_sum_n = tf.add(self.att2_sum_n, self.att2_e_n)  # [B,?,A]
            self.att2_sum_n = tf.tanh(self.att2_sum_n)  # [B,?,A]
            self.att2_losits_n = tf.reduce_sum(
                tf.layers.dense(self.att2_sum_n, 1, name='att2_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att2_losits_n = self.Mask(self.att2_losits_n, self.cnt2_n, self.len2_n,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_2_n = tf.exp(self.att2_losits_n)
            self.sum_2_n = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_2_n, axis=-1) + 1e-9, self.alpha), axis=-1)  # [B,1]
            self.att_2_n = tf.expand_dims(self.exp_2_n / self.sum_2_n, axis=1)  # [B,1,?]
            self.item_latent_2_n = tf.matmul(self.att_2_n, self.r2_n_embedding)  # [B,1,H]
            # r3
            self.r3_n_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.r3_n)  # [B,?,H]
            self.att3_i_n = tf.layers.dense(self.neg_embedding, self.attention_size, name='w3_i', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,A]
            self.att3_i_n = tf.expand_dims(self.att3_i_n, 1)  # [B,1,A]
            self.att3_j_n = tf.layers.dense(self.r3_n_embedding, self.attention_size, name='w3_j', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.e3_n_embedding = tf.nn.embedding_lookup(self.weights['actor_embeddings'], self.e3_n)  # [B,?,H]
            # self.e3_n_embedding = tf.add(self.e3_n_embedding, tf.expand_dims(self.r3_embedding, axis=0))  # [B,?,H]
            self.att3_e_n = tf.layers.dense(self.e3_n_embedding, self.attention_size, name='w3_e', reuse=tf.AUTO_REUSE,
                                            use_bias=False)  # [B,?,A]
            self.att3_sum_n = tf.add(self.att3_i_n, self.att3_j_n)
            self.att3_sum_n = tf.add(self.att3_sum_n, self.att3_e_n)  # [B,?,A]
            self.att3_sum_n = tf.tanh(self.att3_sum_n)  # [B,?,A]
            self.att3_losits_n = tf.reduce_sum(
                tf.layers.dense(self.att3_sum_n, 1, name='att3_weight', reuse=tf.AUTO_REUSE, use_bias=False),
                axis=-1)  # [B,?]
            self.att3_losits_n = self.Mask(self.att3_losits_n, self.cnt3_n, self.len3_n,
                                           mode='add')  # [B,?] Mask the filling positions
            self.exp_3_n = tf.exp(self.att3_losits_n)
            self.sum_3_n = tf.expand_dims(tf.pow(tf.reduce_sum(self.exp_3_n, axis=-1) + 1e-9, self.alpha), axis=-1)  # [B,1]
            self.att_3_n = tf.expand_dims(self.exp_3_n / self.sum_3_n, axis=1)  # [B,1,?]
            self.item_latent_3_n = tf.matmul(self.att_3_n, self.r3_n_embedding)  # [B,1,H]
            # merge all item latent in different relations
            # self.item_latent_n = tf.reduce_sum(self.item_latent_0_n,axis=1)  # [B,4,H]
            self.item_latent_n = tf.concat(
                [self.item_latent_0_n, self.item_latent_1_n, self.item_latent_2_n, self.item_latent_3_n],
                axis=1)  # [B,4,H]
            self.item_latent_n = tf.reduce_sum(
                tf.matmul(tf.expand_dims(self.attention_type, axis=1), self.item_latent_n), axis=1)  # [B,H]
            self.mu_n = tf.add(tf.reduce_sum(self.user_embedding, axis=1), self.item_latent_n)
            # self.neg = tf.reduce_sum(tf.multiply(self.mu_n, self.neg_embedding), 1)
            self.neg = tf.multiply(self.mu_n, self.neg_embedding)
            self.neg = tf.nn.dropout(self.neg, rate=1 - self.dropout_keep[-1])
            for i in range(0, len(self.layers)):
                self.neg = tf.add(tf.matmul(self.neg, self.weights['layer_%d' % i]),
                                  self.weights['bias_%d' % i])  # None * layer[i] * 1
                self.neg = self.activation_function(self.neg)
                self.neg = tf.nn.dropout(self.neg, rate=1 - self.dropout_keep[i])  # dropout at each Deep layer
            self.neg = tf.matmul(self.neg, self.weights['prediction'])  # None * 1
            # Compute the loss.
            self.individual_loss = -tf.log(tf.sigmoid(self.pos - self.neg))  # [B,1]
            self.rec_loss = tf.reduce_sum(self.individual_loss) / self.num_samples

            # regularization on translation data
            self.head1_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.head1)  # [B,H]
            self.translation1_embedding = tf.add(self.r1_embedding,
                                                 tf.nn.embedding_lookup(self.weights['genre_embeddings'],
                                                                        self.relation1))  # [B,H]
            self.tail1_pos_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail1_pos)  # [B,H]
            self.score1_pos = tf.multiply(self.head1_embedding, self.translation1_embedding)  # [B,H]
            self.score1_pos = tf.reduce_sum(tf.multiply(self.score1_pos, self.tail1_pos_embedding), axis=-1)  # [B]

            self.tail1_neg_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail1_neg)  # [B,H]
            self.score1_neg = tf.multiply(self.head1_embedding, self.translation1_embedding)
            self.score1_neg = tf.reduce_sum(tf.multiply(self.score1_neg, self.tail1_neg_embedding), axis=-1)

            self.rel_loss_1 = tf.reduce_sum(-tf.log(tf.sigmoid(self.score1_pos - self.score1_neg)))
            #########################################################################
            self.head2_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.head2)  # [B,H]
            self.translation2_embedding = tf.add(self.r2_embedding,
                                                 tf.nn.embedding_lookup(self.weights['director_embeddings'],
                                                                        self.relation2))  # [B,H]
            self.tail2_pos_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail2_pos)  # [B,H]
            self.score2_pos = tf.multiply(self.head2_embedding, self.translation2_embedding)  # [B,H]
            self.score2_pos = tf.reduce_sum(tf.multiply(self.score2_pos, self.tail2_pos_embedding), axis=-1)  # [B]

            self.tail2_neg_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail2_neg)  # [B,H]
            self.score2_neg = tf.multiply(self.head2_embedding, self.translation2_embedding)
            self.score2_neg = tf.reduce_sum(tf.multiply(self.score2_neg, self.tail2_neg_embedding), axis=-1)

            self.rel_loss_2 = tf.reduce_sum(-tf.log(tf.sigmoid(self.score2_pos - self.score2_neg)))
            #########################################################################
            self.head3_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.head3)  # [B,H]
            self.translation3_embedding = tf.add(self.r3_embedding,
                                                 tf.nn.embedding_lookup(self.weights['actor_embeddings'],
                                                                        self.relation3))  # [B,H]
            self.tail3_pos_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail3_pos)  # [B,H]
            self.score3_pos = tf.multiply(self.head3_embedding, self.translation3_embedding)  # [B,H]
            self.score3_pos = tf.reduce_sum(tf.multiply(self.score3_pos, self.tail3_pos_embedding), axis=-1)  # [B]

            self.tail3_neg_embedding = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.tail3_neg)  # [B,H]
            self.score3_neg = tf.multiply(self.head3_embedding, self.translation3_embedding)
            self.score3_neg = tf.reduce_sum(tf.multiply(self.score3_neg, self.tail3_neg_embedding), axis=-1)

            self.rel_loss_3 = tf.reduce_sum(-tf.log(tf.sigmoid(self.score3_pos - self.score3_neg)))
            self.rel_loss = self.reg_t * (self.rel_loss_1 + self.rel_loss_2 + self.rel_loss_3) / self.num_rel_samples

            self.loss = tf.add(self.rec_loss, self.rel_loss)
            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # gradients and hessian
            self.user_explained = tf.placeholder(tf.int32, shape=())
            self.item_explained = tf.placeholder(tf.int32, shape=())

            self.params = [self.weights['user_embeddings']]
            self.num_params = self.hidden_factor

            self.pred_grads = tf.gradients(self.pos, self.params)
            self.get_explained_grad(self.pred_grads)

            self.loss_grads = jacobian(self.individual_loss, self.params)
            self.loss_grads[0] = self.loss_grads[0][:, 0, self.user_explained, :]
            # self.loss_grads[1] = self.loss_grads[1][:, 0, self.item_explained, :]
            self.loss_grads = tf.concat(self.loss_grads, axis=1)

            self.loss_grads2 = tf.gradients(self.loss, self.params)
            self.get_explained_grad(self.loss_grads2)
            self.loss_grads2 = tf.concat(self.loss_grads2, axis=0)

            self.hessian = jacobian(self.loss_grads2, self.params)
            self.hessian[0] = self.hessian[0][:, self.user_explained, :]
            # self.hessian[1] = self.hessian[1][:, self.item_explained, :]
            self.hessian = tf.concat(self.hessian, axis=1)

            self.damping = 0.01

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            if self.pretrain_flag > 0:
                weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
                weight_saver.restore(self.sess, self.save_file)
                # self.print_weights()

    def print_weights(self):
        tensor_names = ['user_embeddings:0', 'relation_type_embeddings:0', 'item_embeddings:0', 'genre_embeddings:0',
                        'director_embeddings:0', 'actor_embeddings:0', 'mask_embedding:0', 'prediction:0']
        tensor_names.extend(['layer_{}:0'.format(i) for i in range(len(self.layers))])
        tensor_names.extend(['bias_{}:0'.format(i) for i in range(len(self.layers))])
        tensors = self.sess.run(tensor_names)
        for tensor, value in zip(tensor_names, tensors):
            print(tensor)
            print(value)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['user_embeddings'] = tf.Variable(
            tf.random_normal([self.num_users, self.hidden_factor], 0.0, 0.05),
            name='user_embeddings')  # user_num * H
        all_weights['relation_type_embeddings'] = tf.Variable(
            tf.random_normal([4, self.hidden_factor], 0.0, 0.05), name='relation_type_embeddings')
        ie = tf.Variable(tf.random_normal([self.num_items, self.hidden_factor], 0.0, 0.05), name='item_embeddings')
        ge = tf.Variable(tf.random_normal([self.num_genres, self.hidden_factor], 0.0, 0.05),
                         name='genre_embeddings')
        de = tf.Variable(tf.random_normal([self.num_directors, self.hidden_factor], 0.0, 0.05),
                         name='director_embeddings')
        ae = tf.Variable(tf.random_normal([self.num_actors, self.hidden_factor], 0.0, 0.05),
                         name='actor_embeddings')
        mask_embedding = tf.Variable(tf.constant(0.0, shape=[1, self.hidden_factor], dtype=tf.float32),
                                     name='mask_embedding', trainable=False)
        all_weights['item_embeddings'] = tf.concat([ie, mask_embedding], axis=0)
        all_weights['genre_embeddings'] = tf.concat([ge, mask_embedding], axis=0)
        all_weights['director_embeddings'] = tf.concat([de, mask_embedding], axis=0)
        all_weights['actor_embeddings'] = tf.concat([ae, mask_embedding], axis=0)

        num_layer = len(self.layers)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.layers[0]))
            all_weights['layer_0'] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.layers[0])), dtype=np.float32,
                name='layer_0')
            all_weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])),
                                                dtype=np.float32, name='bias_0')  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i - 1] + self.layers[i]))
                all_weights['layer_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i - 1], self.layers[i])), dtype=np.float32,
                    name='layer_%d' % i)  # layers[i-1]*layers[i]
                all_weights['bias_%d' % i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32,
                    name='bias_%d' % i)  # 1 * layer[i]
            # prediction layer
            glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)),
                                                    dtype=np.float32, name='prediction')  # layers[-1] * 1
        return all_weights

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.user: data['user'], self.item_pos: data['positive'], self.item_neg: data['negative'],
                     self.alpha: data['alpha'],
                     self.r0_p: data['r0_p'], self.r1_p: data['r1_p'], self.r2_p: data['r2_p'], self.r3_p: data['r3_p'],
                     self.cnt0_p: data['cnt0_p'], self.cnt1_p: data['cnt1_p'], self.cnt2_p: data['cnt2_p'],
                     self.cnt3_p: data['cnt3_p'],
                     self.e1_p: data['e1_p'], self.e2_p: data['e2_p'], self.e3_p: data['e3_p'],
                     self.len0_p: data['len0_p'], self.len1_p: data['len1_p'], self.len2_p: data['len2_p'],
                     self.len3_p: data['len3_p'],
                     self.r0_n: data['r0_n'], self.r1_n: data['r1_n'], self.r2_n: data['r2_n'], self.r3_n: data['r3_n'],
                     self.cnt0_n: data['cnt0_n'], self.cnt1_n: data['cnt1_n'], self.cnt2_n: data['cnt2_n'],
                     self.cnt3_n: data['cnt3_n'],
                     self.e1_n: data['e1_n'], self.e2_n: data['e2_n'], self.e3_n: data['e3_n'],
                     self.len0_n: data['len0_n'], self.len1_n: data['len1_n'], self.len2_n: data['len2_n'],
                     self.len3_n: data['len3_n'],
                     self.head1: data['head1'], self.relation1: data['relation1'], self.tail1_pos: data['tail1_pos'],
                     self.tail1_neg: data['tail1_neg'],
                     self.head2: data['head2'], self.relation2: data['relation2'], self.tail2_pos: data['tail2_pos'],
                     self.tail2_neg: data['tail2_neg'],
                     self.head3: data['head3'], self.relation3: data['relation3'], self.tail3_pos: data['tail3_pos'],
                     self.tail3_neg: data['tail3_neg'],
                     self.dropout_keep: self.keep_prob, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss

    def prepare_rel_data_batch(self, rel_batch):
        head = [[] for i in range(4)]
        relation = [[] for i in range(4)]
        tail_pos = [[] for i in range(4)]
        tail_neg = [[] for i in range(4)]
        for _, row in rel_batch.iterrows():
            type = row['type']
            head[type].append(row['head'])
            relation[type].append(row['value'])
            tail_pos[type].append(row['tail_pos'])
            tail_neg[type].append(row['tail_neg'])
        return {'head1': head[1], 'relation1': relation[1], 'tail1_pos': tail_pos[1], 'tail1_neg': tail_neg[1],
                'head2': head[2], 'relation2': relation[2], 'tail2_pos': tail_pos[2], 'tail2_neg': tail_neg[2],
                'head3': head[3], 'relation3': relation[3], 'tail3_pos': tail_pos[3], 'tail3_neg': tail_neg[3]}

    def prepare_batch(self, train_batch, data, args):  # generate a random block of training data
        user, positive, negative, alpha = [], [], [], []
        r0_p, r1_p, r2_p, r3_p = [], [], [], []  # for positive item i, the item set in ru+ which has relationship
        # r0,r1,r2,r3 with i
        cnt0_p, cnt1_p, cnt2_p, cnt3_p = [], [], [], []  # the number of corresponding r, for masking
        e1_p, e2_p, e3_p = [], [], []  # the set of specific attribute value for corresponding r except r0
        # for negative part
        r0_n, r1_n, r2_n, r3_n = [], [], [], []
        cnt0_n, cnt1_n, cnt2_n, cnt3_n = [], [], [], []
        e1_n, e2_n, e3_n = [], [], []
        head1, relation1, tail1_pos, tail1_neg = [], [], [], []
        head2, relation2, tail2_pos, tail2_neg = [], [], [], []
        head3, relation3, tail3_pos, tail3_neg = [], [], [], []
        # get sample
        for _, row in train_batch.iterrows():
            user_id = row['user']
            item_id = row['pos_item']
            user.append(user_id)
            positive.append(item_id)
            neg = row['neg_item']
            # pos = data.user_positive_list[user_id]
            # neg = np.random.randint(self.num_items)  # uniform sample a negative itemID from negative set
            # while neg in pos:
            #     neg = np.random.randint(self.num_items)
            negative.append(neg)
            alpha.append(args.alpha)
            # t1 = time()
            r0_temp, r1_temp, r2_temp, r3_temp, e1_temp, e2_temp, e3_temp, cnt0_temp, cnt1_temp, cnt2_temp, cnt3_temp \
                = get_relational_data(user_id, item_id, data)
            # t2= time()
            # print ('the time of generating batch:%f' % (t2 - t1))
            r0_p.append(r0_temp)
            r1_p.append(r1_temp)
            r2_p.append(r2_temp)
            r3_p.append(r3_temp)
            e1_p.append(e1_temp)
            e2_p.append(e2_temp)
            e3_p.append(e3_temp)
            cnt0_p.append(cnt0_temp)
            cnt1_p.append(cnt1_temp)
            cnt2_p.append(cnt2_temp)
            cnt3_p.append(cnt3_temp)
            # for negative part
            r0_temp, r1_temp, r2_temp, r3_temp, e1_temp, e2_temp, e3_temp, cnt0_temp, cnt1_temp, cnt2_temp, cnt3_temp \
                = get_relational_data(user_id, neg, data)
            r0_n.append(r0_temp)
            r1_n.append(r1_temp)
            r2_n.append(r2_temp)
            r3_n.append(r3_temp)
            e1_n.append(e1_temp)
            e2_n.append(e2_temp)
            e3_n.append(e3_temp)
            cnt0_n.append(cnt0_temp)
            cnt1_n.append(cnt1_temp)
            cnt2_n.append(cnt2_temp)
            cnt3_n.append(cnt3_temp)
        # fill out a fixed length for each batch
        len0_p = max(cnt0_p)
        len1_p = max(cnt1_p)
        len2_p = max(cnt2_p)
        len3_p = max(cnt3_p)
        for index in range(len(r0_p)):
            if len(r0_p[index]) < len0_p:
                r0_p[index].extend(np.array([self.num_items]).repeat(len0_p - len(r0_p[index])))
            if len(r1_p[index]) < len1_p:
                r1_p[index].extend(np.array([self.num_items]).repeat(len1_p - len(r1_p[index])))
                e1_p[index].extend(np.array([self.num_genres]).repeat(len1_p - len(e1_p[index])))
            if len(r2_p[index]) < len2_p:
                r2_p[index].extend(np.array([self.num_items]).repeat(len2_p - len(r2_p[index])))
                e2_p[index].extend(np.array([self.num_directors]).repeat(len2_p - len(e2_p[index])))
            if len(r3_p[index]) < len3_p:
                r3_p[index].extend(np.array([self.num_items]).repeat(len3_p - len(r3_p[index])))
                e3_p[index].extend(np.array([self.num_actors]).repeat(len3_p - len(e3_p[index])))
        len0_n = max(cnt0_n)
        len1_n = max(cnt1_n)
        len2_n = max(cnt2_n)
        len3_n = max(cnt3_n)
        for index in range(len(r0_n)):
            if len(r0_n[index]) < len0_n:
                r0_n[index].extend(np.array([self.num_items]).repeat(len0_n - len(r0_n[index])))
            if len(r1_n[index]) < len1_n:
                r1_n[index].extend(np.array([self.num_items]).repeat(len1_n - len(r1_n[index])))
                e1_n[index].extend(np.array([self.num_genres]).repeat(len1_n - len(e1_n[index])))
            if len(r2_n[index]) < len2_n:
                r2_n[index].extend(np.array([self.num_items]).repeat(len2_n - len(r2_n[index])))
                e2_n[index].extend(np.array([self.num_directors]).repeat(len2_n - len(e2_n[index])))
            if len(r3_n[index]) < len3_n:
                r3_n[index].extend(np.array([self.num_items]).repeat(len3_n - len(r3_n[index])))
                e3_n[index].extend(np.array([self.num_actors]).repeat(len3_n - len(e3_n[index])))

        return {'user': user, 'positive': positive, 'negative': negative, 'alpha': alpha,
                'r0_p': r0_p, 'r1_p': r1_p, 'r2_p': r2_p, 'r3_p': r3_p,
                'e1_p': e1_p, 'e2_p': e2_p, 'e3_p': e3_p,
                'cnt0_p': cnt0_p, 'cnt1_p': cnt1_p, 'cnt2_p': cnt2_p, 'cnt3_p': cnt3_p,
                'len0_p': len0_p, 'len1_p': len1_p, 'len2_p': len2_p, 'len3_p': len3_p,
                'r0_n': r0_n, 'r1_n': r1_n, 'r2_n': r2_n, 'r3_n': r3_n,
                'e1_n': e1_n, 'e2_n': e2_n, 'e3_n': e3_n,
                'cnt0_n': cnt0_n, 'cnt1_n': cnt1_n, 'cnt2_n': cnt2_n, 'cnt3_n': cnt3_n,
                'len0_n': len0_n, 'len1_n': len1_n, 'len2_n': len2_n, 'len3_n': len3_n,
                'head1': head1, 'relation1': relation1, 'tail1_pos': tail1_pos, 'tail1_neg': tail1_neg,
                'head2': head2, 'relation2': relation2, 'tail2_pos': tail2_pos, 'tail2_neg': tail2_neg,
                'head3': head3, 'relation3': relation3, 'tail3_pos': tail3_pos, 'tail3_neg': tail3_neg}

    def train(self, data, args, seed):  # fit a dataset
        np.random.seed(seed)
        for epoch in range(self.epoch):
            train_data = data.train_data.sample(frac=1)
            train_data = train_data.reset_index(drop=True)
            rel_data = data.rel_data.sample(frac=1)
            rel_data = rel_data.reset_index(drop=True)
            num_iter = (train_data.shape[0] + self.batch_size - 1) // self.batch_size
            rel_batch_size = (rel_data.shape[0] + num_iter - 1) // num_iter
            total_loss = 0
            i2 = 0
            for i in range(0, train_data.shape[0], self.batch_size):
                j = min(i + self.batch_size, train_data.shape[0])
                # generate a batch
                # t1 = time()
                batch_xs = self.prepare_batch(train_data.iloc[i:j], data, args)
                j2 = min(i2 + rel_batch_size, rel_data.shape[0])
                batch_xs.update(self.prepare_rel_data_batch(rel_data.iloc[i2:j2]))
                i2 += rel_batch_size
                # t2 = time()
                # print ('the time of generating batch:%f'%(t2-t1))
                # Fit training
                loss = self.partial_fit(batch_xs)
                # t3=time()
                # print ('the time of optimizting:%f' % (t3 - t2))
                # print(loss)
                total_loss = total_loss + loss


            attention_type = self.get_attention_type_scalar()
            avearge = np.mean(attention_type, axis=0)
            print("the total loss in %d th iteration is: %f, the attentions are %.4f, %.4f, %.4f, %.4f" % (
                epoch, total_loss, avearge[0], avearge[1], avearge[2], avearge[3]))
            # self.evaluate(data, args)
        if self.pretrain_flag < 0:
            print("Save model to file as pretrain.")
            self.print_weights()
            self.saver.save(self.sess, self.save_file)

    @staticmethod
    def Mask(inputs, seq_len, long, mode):
        if seq_len is None:
            return inputs
        else:
            mask = tf.cast(tf.sequence_mask(seq_len, long), tf.float32)
            # for _ in range(len(inputs.shape) - 2):
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def evaluate(self, data, args):
        self.graph.finalize()
        count = [0, 0, 0, 0, 0]
        rank = [[], [], [], [], []]
        hit_user_list = []
        hit_item_list = []
        for index, row in data.test_data.iterrows():
            user_id = row['user']
            scores = self.get_scores_per_user(user_id, data, args)
            # get true item score
            true_item_id = row['pos_item']
            true_item_score = scores[true_item_id]
            # delete visited scores
            visited = data.user_positive_list[user_id]  # get positive list for the userID
            scores = np.delete(scores, visited)
            # whether hit
            sorted_scores = sorted(scores, reverse=True)
            label = [sorted_scores[4], [sorted_scores[9]], [sorted_scores[14]], [sorted_scores[19]],
                     [sorted_scores[24]]]

            if true_item_score >= label[0]:
                count[0] = count[0] + 1
                rank[0].append(sorted_scores.index(true_item_score) + 1)
                hit_user_list.append(user_id)
                hit_item_list.append(true_item_id)
            if true_item_score >= label[1]:
                count[1] = count[1] + 1
                rank[1].append(sorted_scores.index(true_item_score) + 1)
            if true_item_score >= label[2]:
                count[2] = count[2] + 1
                rank[2].append(sorted_scores.index(true_item_score) + 1)
            if true_item_score >= label[3]:
                count[3] = count[3] + 1
                rank[3].append(sorted_scores.index(true_item_score) + 1)
            if true_item_score >= label[4]:
                count[4] = count[4] + 1
                rank[4].append(sorted_scores.index(true_item_score) + 1)
            print(index, true_item_score)
        for i in range(5):
            mrr = 0
            ndcg = 0
            hit_rate = float(count[i]) / data.test_data.shape[0]
            for item in rank[i]:
                mrr = mrr + float(1.0) / item
                ndcg = ndcg + float(1.0) / np.log2(item + 1)
            mrr = mrr / data.test_data.shape[0]
            ndcg = ndcg / data.test_data.shape[0]
            k = (i + 1) * 5
            print("top:%d" % k)
            print("the Hit Rate is: %f" % hit_rate)
            print("the MRR is: %f" % mrr)
            print("the NDCG is: %f" % ndcg)

    def prepare_input(self, user_ids, item_ids, data, ignored_user=-1, ignored_id=None):
        r0, r1, r2, r3 = [], [], [], []  # for positive item i, the item set in ru+ which has relationship r0,r1,
        # r2,r3 with i
        cnt0, cnt1, cnt2, cnt3 = [], [], [], []  # the number of corresponding r, for masking
        e1, e2, e3 = [], [], []  # the set of specific attribute value for corresponding r except r0
        for user_id, item_id in zip(user_ids, item_ids):
            r0_temp, r1_temp, r2_temp, r3_temp, e1_temp, e2_temp, e3_temp, cnt0_temp, cnt1_temp, cnt2_temp, \
            cnt3_temp = get_relational_data(user_id, item_id, data, ignored_id if user_id == ignored_user else None)
            r0.append(r0_temp)
            r1.append(r1_temp)
            r2.append(r2_temp)
            r3.append(r3_temp)
            e1.append(e1_temp)
            e2.append(e2_temp)
            e3.append(e3_temp)
            cnt0.append(cnt0_temp)
            cnt1.append(cnt1_temp)
            cnt2.append(cnt2_temp)
            cnt3.append(cnt3_temp)
        len0 = max(cnt0)
        len1 = max(cnt1)
        len2 = max(cnt2)
        len3 = max(cnt3)
        for index in range(len(r0)):
            if len(r0[index]) < len0:
                r0[index].extend(np.array([self.num_items]).repeat(len0 - len(r0[index])))
            if len(r1[index]) < len1:
                r1[index].extend(np.array([self.num_items]).repeat(len1 - len(r1[index])))
                e1[index].extend(np.array([self.num_genres]).repeat(len1 - len(e1[index])))
            if len(r2[index]) < len2:
                r2[index].extend(np.array([self.num_items]).repeat(len2 - len(r2[index])))
                e2[index].extend(np.array([self.num_directors]).repeat(len2 - len(e2[index])))
            if len(r3[index]) < len3:
                r3[index].extend(np.array([self.num_items]).repeat(len3 - len(r3[index])))
                e3[index].extend(np.array([self.num_actors]).repeat(len3 - len(e3[index])))
        return r0, r1, r2, r3, cnt0, cnt1, cnt2, cnt3, e1, e2, e3, len0, len1, len2, len3

    def prepare_feed_dict_batch(self, user_ids, pos_items, data, args, ignored_user=-1, ignored_id=None, neg_items=None,
                                rel_data=None, user_explained=None, item_explained=None):
        alpha = [args.alpha] * len(pos_items)
        r0, r1, r2, r3, cnt0, cnt1, cnt2, cnt3, e1, e2, e3, len0, len1, len2, len3 = \
            self.prepare_input(user_ids, pos_items, data, ignored_user=ignored_user, ignored_id=ignored_id)
        feed_dict = {self.user: user_ids, self.item_pos: pos_items, self.alpha: alpha,
                     self.r0_p: r0, self.r1_p: r1, self.r2_p: r2, self.r3_p: r3,
                     self.cnt0_p: cnt0, self.cnt1_p: cnt1, self.cnt2_p: cnt2, self.cnt3_p: cnt3,
                     self.e1_p: e1, self.e2_p: e2, self.e3_p: e3,
                     self.len0_p: len0, self.len1_p: len1, self.len2_p: len2, self.len3_p: len3,
                     self.dropout_keep: self.no_dropout, self.train_phase: False}
        if neg_items is not None:
            r0_n, r1_n, r2_n, r3_n, cnt0_n, cnt1_n, cnt2_n, cnt3_n, e1_n, e2_n, e3_n, len0_n, len1_n, len2_n, len3_n = \
                self.prepare_input(user_ids, neg_items, data, ignored_user=ignored_user, ignored_id=ignored_id)
            feed_dict_n = {self.item_neg: neg_items,
                         self.r0_n: r0_n, self.r1_n: r1_n, self.r2_n: r2_n, self.r3_n: r3_n,
                         self.cnt0_n: cnt0_n, self.cnt1_n: cnt1_n, self.cnt2_n: cnt2_n, self.cnt3_n: cnt3_n,
                         self.e1_n: e1_n, self.e2_n: e2_n, self.e3_n: e3_n,
                         self.len0_n: len0_n, self.len1_n: len1_n, self.len2_n: len2_n, self.len3_n: len3_n}
            feed_dict.update(feed_dict_n)
        if rel_data is not None:
            head = [[] for i in range(4)]
            relation = [[] for i in range(4)]
            tail_pos = [[] for i in range(4)]
            tail_neg = [[] for i in range(4)]
            for _, row in rel_data.iterrows():
                type = row['type']
                head[type].append(row['head'])
                relation[type].append(row['value'])
                tail_pos[type].append(row['tail_pos'])
                tail_neg[type].append(row['tail_neg'])
            feed_dict.update({
                self.head1: head[1], self.head2: head[2], self.head3: head[3],
                self.relation1: relation[1], self.relation2: relation[2], self.relation3: relation[3],
                self.tail1_pos: tail_pos[1], self.tail2_pos: tail_pos[2], self.tail3_pos: tail_pos[3],
                self.tail1_neg: tail_neg[1], self.tail2_neg: tail_neg[2], self.tail3_neg: tail_neg[3]
            })
        else:
            feed_dict.update({
                self.head1: [], self.head2: [], self.head3: [],
                self.relation1: [], self.relation2: [], self.relation3: [],
                self.tail1_pos: [], self.tail2_pos: [], self.tail3_pos: [],
                self.tail1_neg: [], self.tail2_neg: [], self.tail3_neg: []
            })
        if user_explained is not None:
            feed_dict[self.user_explained] = user_explained
        if item_explained is not None:
            feed_dict[self.item_explained] = item_explained
        return feed_dict

    def get_scores_per_user(self, user_id, data, args,
                            ignored_id=None):  # evaluate the results for an user context, return scorelist
        scorelist = []
        users = [user_id] * self.batch_size
        for j in range(0, self.num_items, self.batch_size):
            k = min(j + self.batch_size, self.num_items)
            feed_dict = self.prepare_feed_dict_batch(users[:(k - j)], range(j, k), data, args, ignored_user=user_id,
                                                     ignored_id=ignored_id)
            # print(X_item)
            scores = self.sess.run(self.pos, feed_dict=feed_dict)
            scores = scores.reshape(k - j)
            scorelist = np.append(scorelist, scores)
        return scorelist

    def prepare_feed_dict(self, user_id, item_id, data, args, r=None, e=None, cnt=None, item_ignored=None):
        if r is None:
            n_types = 4
            r = [None] * n_types
            e = [0] * n_types
            cnt = [0] * n_types

            r[0], r[1], r[2], r[3], e[1], e[2], e[3], cnt[0], cnt[1], cnt[2], cnt[3] = \
                get_relational_data(user_id, item_id, data, ignored_id={item_ignored})

        feed_dict = {self.user: [user_id], self.item_pos: [item_id], self.alpha: [args.alpha],
                     self.r0_p: [r[0]], self.r1_p: [r[1]], self.r2_p: [r[2]], self.r3_p: [r[3]],
                     self.cnt0_p: [cnt[0]], self.cnt1_p: [cnt[1]], self.cnt2_p: [cnt[2]], self.cnt3_p: [cnt[3]],
                     self.e1_p: [e[1]], self.e2_p: [e[2]], self.e3_p: [e[3]],
                     self.len0_p: cnt[0], self.len1_p: cnt[1], self.len2_p: cnt[2], self.len3_p: cnt[3],
                     self.dropout_keep: self.no_dropout, self.train_phase: False}
        return feed_dict

    def get_attention_user_item(self, user_id, item_id, r, e, cnt, args):
        feed_dict = self.prepare_feed_dict(user_id, item_id, None, args, r, e, cnt)
        return self.sess.run([self.att_0_p, self.att_1_p, self.att_2_p, self.att_3_p], feed_dict=feed_dict)

    def get_attention_type_scalar(self):  # evaluate the results for an user context, return score list
        all_users = list(range(self.num_users))
        feed_dict = {self.user: all_users}
        results = self.sess.run(self.attention_type, feed_dict=feed_dict)
        return results

    def get_attention_type_user(self, user_id):
        return self.sess.run(self.attention_type, feed_dict={self.user: [user_id]})

    def get_loss_grad_individual(self, user_id, item_id, pos_item, neg_item, data, args):
        feed_dict = self.prepare_feed_dict_batch([user_id], [pos_item], data, args,
                                                 neg_items=[neg_item], user_explained=user_id,
                                                 item_explained=item_id, ignored_user=user_id, ignored_id={pos_item})
        return self.sess.run(self.loss_grads, feed_dict=feed_dict)

    def get_explained_grad(self, grads):
        grads[0] = tf.reduce_sum(grads[0].values[tf.equal(grads[0].indices, self.user_explained)], axis=0)

    def get_hessian(self, user_id, item_explained, item_removed, data, args, batch=32, verbose=1):
        train_data = data.train_data[(data.train_data['user'] == user_id)]
        res = np.zeros(shape=(self.num_params, self.num_params), dtype=np.float32)
        for i in range(0, train_data.shape[0], batch):
            j = min(i + batch, train_data.shape[0])
            if verbose > 0:
                print('hessian batch', i, j)
            feed_dict = self.prepare_feed_dict_batch(train_data['user'][i:j], train_data['pos_item'][i:j], data, args,
                                                     neg_items=train_data['neg_item'][i:j], user_explained=user_id,
                                                     item_explained=item_explained,
                                                     ignored_user=user_id, ignored_id={item_removed})
            tmp = self.sess.run(self.hessian, feed_dict=feed_dict)
            res += tmp
        res = res * self.num_samples / train_data.shape[0]
        return res + np.identity(res.shape[0]) * self.damping

    def get_influence3(self, user_id, item_id, data, args):  # old params - new params
        print(f'get influence {user_id} {item_id}')
        train_data = data.train_data[data.train_data['user'] == user_id].reset_index(drop=True)
        l = len(train_data['pos_item'])
        res = np.zeros(l)
        for i, row in train_data.iterrows():
            hessian = self.get_hessian(user_id, item_id, row['pos_item'], data, args, batch=32, verbose=0)
            inv_hessian = np.linalg.pinv(hessian)
            loss_grad = self.get_loss_grad_individual(user_id, item_id, row['pos_item'], row['neg_item'], data, args)
            params_infl = -np.matmul(loss_grad, inv_hessian) / train_data.shape[0]
            res[i] = self.get_score_influence(user_id, item_id, row['pos_item'], params_infl, data, args)
        return res

    def get_score_influence(self, user_id, rec_id, removed_id, params_influences, data, args):
        feed_dict = self.prepare_feed_dict(user_id, rec_id, data, args)
        user_emb, item_emb, score = self.sess.run([self.user_embedding, self.pos_embedding, self.pos], feed_dict=feed_dict)

        feed_dict = self.prepare_feed_dict(user_id, rec_id, data, args, item_ignored=removed_id)
        feed_dict.pop(self.user)

        feed_dict[self.user_embedding] = user_emb - params_influences[0, :self.hidden_factor]
        new_score = self.sess.run(self.pos, feed_dict=feed_dict)
        return score - new_score
