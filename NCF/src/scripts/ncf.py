from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib
import numpy as np

matplotlib.use('agg')
import os.path
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import math
from scipy.optimize import fmin_ncg

from NCF.src.influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay
from NCF.src.influence.dataset import DataSet
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


class NCF(GenericNeuralNet):
    def __init__(self, num_users, num_items, embedding_size, weight_decay, **kwargs):
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.weight_decay = weight_decay
        # self.num_classes = 1
        super(NCF, self).__init__(**kwargs)

    def get_all_params(self):
        all_params = []
        for layer in ['embedding_layer', 'h1', 'h2', 'h3']:
            # for var_name in ['embedding_users', 'embedding_items', 'bias_users', 'bias_items', 'global_bias']:
            for var_name in ['mlp/embedding_users', 'mlp/embedding_items', 'gmf/embedding_users', 'gmf/embedding_items',
                             'weights', 'biases']:
                try:
                    temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))
                    all_params.append(temp_tensor)
                except KeyError:
                    continue
        print('length of all_params: %s, which should be equal to 10' % len(all_params))
        return all_params

    def get_test_params(self, test_index):
        test_params = []
        test_u, test_i = self.data_sets.test.x[test_index[0]]
        test_u, test_i = int(test_u), int(test_i)
        print('Test user: %s item: %s' % (test_u, test_i))
        for layer in ['embedding_layer']:
            for var_name in ['mlp/embedding_users', 'mlp/embedding_items', 'gmf/embedding_users',
                             'gmf/embedding_items']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))
                if 'embedding_users' in var_name:
                    temp_embedding = tf.nn.embedding_lookup(
                        tf.reshape(temp_tensor, (self.num_users, self.embedding_size)),
                        [test_u], name="test_user_embedding")
                    # temp_embedding = temp_tensor[self.embedding_size*test_u:self.embedding_size*(test_u+1)]
                elif 'embedding_items' in var_name:
                    temp_embedding = tf.nn.embedding_lookup(
                        tf.reshape(temp_tensor, (self.num_items, self.embedding_size)),
                        [test_i], name="test_item_embedding")
                    # temp_embedding = temp_tensor[self.embedding_size * test_i:self.embedding_size * (test_i + 1)]
                temp_embedding = tf.reshape(temp_embedding, [-1])
                test_params.append(temp_embedding)
            print("Length of test params: %d" % len(test_params))
        return test_params

    def retrain(self, num_steps, feed_dict):
        retrain_dataset = DataSet(feed_dict[self.input_placeholder], feed_dict[self.labels_placeholder])
        for step in range(num_steps):
            iter_feed_dict = self.fill_feed_dict_with_batch(retrain_dataset)
            self.sess.run(self.train_op, feed_dict=iter_feed_dict)

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.int32,
            shape=(None, 2),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.float32,
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder

    def fnn_layer(self, hidden_input, output_dim):
        input_dim = hidden_input.get_shape()[1].value
        print("input dim for hidden layer: %s" % input_dim)
        weights = variable_with_weight_decay(
            'weights',
            [input_dim * output_dim],
            stddev=1.0 / math.sqrt(float(input_dim)),
            # wd=0
            wd=self.weight_decay
        )
        bias = variable(
            'biases',
            [output_dim],
            tf.constant_initializer(0.0))
        hidden_output = tf.matmul(hidden_input, tf.reshape(weights, (int(input_dim), int(output_dim)))) + bias
        return hidden_output

    def inference(self, input_x):
        with tf.variable_scope('embedding_layer'):
            with tf.variable_scope('mlp'):
                embedding_users_mlp = variable_with_weight_decay("embedding_users",
                                                                 [self.num_users * self.embedding_size],
                                                                 stddev=1.0 / math.sqrt(float(self.embedding_size)),
                                                                 wd=self.weight_decay)
                embedding_items_mlp = variable_with_weight_decay("embedding_items",
                                                                 [self.num_items * self.embedding_size],
                                                                 stddev=1.0 / math.sqrt(float(self.embedding_size)),
                                                                 wd=self.weight_decay)
                user_embedding_mlp = tf.nn.embedding_lookup(
                    tf.reshape(embedding_users_mlp, (self.num_users, self.embedding_size)),
                    input_x[:, 0], name="user_embedding")
                item_embedding_mlp = tf.nn.embedding_lookup(
                    tf.reshape(embedding_items_mlp, (self.num_items, self.embedding_size)),
                    input_x[:, 1], name="item_embedding")
                hidden_input_mlp = tf.concat([user_embedding_mlp, item_embedding_mlp], axis=1,
                                             name='mlp_embedding_concat')
            with tf.variable_scope('gmf'):
                embedding_users_gmf = variable_with_weight_decay("embedding_users",
                                                                 [self.num_users * self.embedding_size],
                                                                 stddev=1.0 / math.sqrt(float(self.embedding_size)),
                                                                 wd=self.weight_decay)
                embedding_items_gmf = variable_with_weight_decay("embedding_items",
                                                                 [self.num_items * self.embedding_size],
                                                                 stddev=1.0 / math.sqrt(float(self.embedding_size)),
                                                                 wd=self.weight_decay)
                user_embedding_gmf = tf.nn.embedding_lookup(
                    tf.reshape(embedding_users_gmf, (self.num_users, self.embedding_size)),
                    input_x[:, 0], name="user_embedding")
                item_embedding_gmf = tf.nn.embedding_lookup(
                    tf.reshape(embedding_items_gmf, (self.num_items, self.embedding_size)),
                    input_x[:, 1], name="item_embedding")
                h_gmf = user_embedding_gmf * item_embedding_gmf
        with tf.variable_scope('h1'):
            h1_o = tf.nn.relu(self.fnn_layer(hidden_input_mlp, self.embedding_size), 'hidden_output')
        with tf.variable_scope('h2'):
            h2_o = tf.nn.relu(self.fnn_layer(h1_o, self.embedding_size // 2), 'hidden_output')
            h2_concat = tf.concat([h2_o, h_gmf], axis=1, name='hidden_concat')
        with tf.variable_scope('h3'):
            rating = tf.squeeze(self.fnn_layer(h2_concat, 1), name='rating')
        return rating

    def predictions(self, logits):
        preds = logits
        return preds

    def loss(self, logits, labels):

        squared_error = tf.square(logits - labels, name="squared_error")

        indiv_loss_no_reg = squared_error
        loss_no_reg = tf.reduce_mean(squared_error, name='squared_error_mean')
        tf.add_to_collection('losses', loss_no_reg)

        total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return total_loss, loss_no_reg, indiv_loss_no_reg

    def get_accuracy_op(self, logits, labels):
        """Evaluate the quality of the logits at predicting the label.
        Args:
          logits: Logits tensor, float - [batch_size, NUM_CLASSES].
          labels: Labels tensor, int32 - [batch_size], with values in the
            range [0, NUM_CLASSES).
        Returns:
          A scalar int32 tensor with the number of examples (out of batch_size)
          that were predicted correctly.
        """
        # correct = tf.nn.in_top_k(logits, labels, 1)
        # return tf.reduce_mean(1. - tf.abs(logits - labels) / (labels + 0.0001))
        return tf.reduce_mean(tf.abs(logits - labels))

    def adversarial_loss(self, logits, labels):

        return None, None  # , indiv_wrong_prob

    def get_test_grad(self, grad_total_loss_op):
        test_grad = []
        test_grad.append(
            grad_total_loss_op[0][self.test_u * self.embedding_size:(1 + self.test_u) * self.embedding_size])
        test_grad.append(
            grad_total_loss_op[1][self.test_i * self.embedding_size:(1 + self.test_i) * self.embedding_size])
        test_grad.append(
            grad_total_loss_op[2][self.test_u * self.embedding_size:(1 + self.test_u) * self.embedding_size])
        test_grad.append(
            grad_total_loss_op[3][self.test_i * self.embedding_size:(1 + self.test_i) * self.embedding_size])
        return test_grad

    def get_influence_on_test_loss(self, test_indices, train_idx,
                                   approx_type='cg', approx_params=None, force_refresh=True, test_description=None,
                                   loss_type='normal_loss',
                                   X=None, Y=None):
        # If train_idx is None then use X and Y (phantom points)
        # Need to make sure test_idx stays consistent between models
        # because mini-batching permutes dataset order

        if train_idx is None:
            if (X is None) or (Y is None): raise (ValueError, 'X and Y must be specified if using phantom points.')
            if X.shape[0] != len(Y): raise (ValueError, 'X and Y must have the same length.')
        else:
            if (X is not None) or (
                    Y is not None): raise (ValueError, 'X and Y cannot be specified if train_idx is specified.')

        assert len(test_indices) == 1
        self.test_index = test_indices[0]
        self.train_indices_of_test_case = self.get_train_indices_of_test_case(test_indices)
        self.params_test = self.get_test_params(test_index=test_indices)
        self.vec_to_list_test = self.get_vec_to_list_fn_test()
        # self.logits_test = self.inference_test()
        # self.total_loss_test, self.loss_no_reg_test, self.indiv_loss_no_reg_test = self.loss(
        #     self.logits_test,
        #     self.labels_placeholder)
        #
        # self.grad_total_loss_op_test = tf.gradients(self.total_loss_test, self.params_test)
        # self.grad_loss_no_reg_op_test = tf.gradients(self.loss_no_reg_test, self.params_test)
        self.grad_total_loss_op_test = self.get_test_grad(self.grad_total_loss_op)
        self.grad_loss_no_reg_op_test = self.get_test_grad(self.grad_loss_no_reg_op)
        self.grad_loss_r_test = self.get_test_grad(self.grad_loss_r)

        self.v_placeholder_test = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in self.params_test]
        self.hessian_vector_test = self.hessian_vector_product_test(self.total_loss, self.params,
                                                                    self.v_placeholder_test)

        # test_grad_loss_no_reg_val = self.get_test_grad_loss_no_reg_val(test_indices, loss_type=loss_type)
        test_grad_loss_r = self.get_r_grad_loss(test_indices, loss_type=loss_type)

        # print("Shape of test gradient: %s" % test_grad_loss_no_reg_val.shape)
        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_r)))

        # start_time = time.time()

        if test_description is None:
            test_description = test_indices

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (
            self.model_name, approx_type, loss_type, test_description))
        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            print('Loaded inverse HVP from %s' % approx_filename)
        else:
            start_time = time.time()
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_r,
                approx_type,
                approx_params)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            print('Saved inverse HVP to %s' % approx_filename)

        duration_1 = time.time() - start_time
        print('Inverse HVP took %s sec' % duration_1)

        start_time = time.time()
        if train_idx is None:
            num_to_remove = len(Y)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter in np.arange(num_to_remove):
                single_train_feed_dict = self.fill_feed_dict_manual(X[counter, :], [Y[counter]])
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val)) / self.num_train_examples

        else:
            num_to_remove = len(self.train_indices_of_test_case)
            predicted_loss_diffs = np.zeros([num_to_remove])
            for counter, idx_to_remove in enumerate(self.train_indices_of_test_case):
                single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
                train_grad_loss_val = self.sess.run(self.grad_total_loss_op_test, feed_dict=single_train_feed_dict)
                predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                       np.concatenate(train_grad_loss_val)) / \
                                                self.train_indices_of_test_case.shape[0]

        duration_2 = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration_2))
        print("Total time is %s sec" % (duration_1 + duration_2))

        return predicted_loss_diffs

    def get_r_grad_loss(self, test_indices, batch_size=100, loss_type='normal_loss'):

        if loss_type == 'normal_loss':
            op = self.grad_loss_r_test

        # elif loss_type == 'adversarial_loss':
        #     op = self.grad_adversarial_loss_op
        else:
            raise (ValueError, 'Loss must be normal')

        if test_indices is not None:
            num_iter = int(np.ceil(len(test_indices) / batch_size))

            test_grad_loss_no_reg_val = None
            for i in range(num_iter):
                start = i * batch_size
                end = int(min((i + 1) * batch_size, len(test_indices)))

                test_feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.test, test_indices[start:end])

                temp = self.sess.run(op, feed_dict=test_feed_dict)

                if test_grad_loss_no_reg_val is None:
                    test_grad_loss_no_reg_val = [a * (end - start) for a in temp]
                else:
                    test_grad_loss_no_reg_val = [a + b * (end - start) for (a, b) in
                                                 zip(test_grad_loss_no_reg_val, temp)]

            test_grad_loss_no_reg_val = [a / len(test_indices) for a in test_grad_loss_no_reg_val]

        else:
            test_grad_loss_no_reg_val = self.minibatch_mean_eval([op], self.data_sets.test)[0]

        return test_grad_loss_no_reg_val

    def minibatch_hessian_vector_val(self, v):

        num_iter = 1
        self.reset_datasets()
        hessian_vector_val = None
        for i in range(num_iter):
            # feed_dict = self.fill_feed_dict_with_batch(self.data_sets.train, batch_size=None)
            feed_dict = self.fill_feed_dict_with_some_ex(self.data_sets.train, self.train_indices_of_test_case.tolist())
            # Can optimize this
            feed_dict = self.update_feed_dict_with_v_placeholder_test(feed_dict, v)
            hessian_vector_val_temp = self.sess.run(self.hessian_vector_test, feed_dict=feed_dict)
            if hessian_vector_val is None:
                hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
            else:
                hessian_vector_val = [a + (b / float(num_iter)) for (a, b) in
                                      zip(hessian_vector_val, hessian_vector_val_temp)]

        hessian_vector_val = [a + self.damping * b for (a, b) in zip(hessian_vector_val, v)]

        return hessian_vector_val

    def update_feed_dict_with_v_placeholder_test(self, feed_dict, vec):
        for pl_block, vec_block in zip(self.v_placeholder_test, vec):
            feed_dict[pl_block] = vec_block
        return feed_dict

    def get_train_indices_of_test_case(self, test_indices):
        assert len(test_indices) == 1
        test_index = test_indices[0]
        test_u, test_i = self.data_sets.test.x[test_index]
        self.test_u, self.test_i = int(test_u), int(test_i)
        u_indices = np.where(self.data_sets.train.x[:, 0] == self.test_u)[0]
        i_indices = np.where(self.data_sets.train.x[:, 1] == self.test_i)[0]
        return np.concatenate((u_indices, i_indices))

    def hessian_vector_product_test(self, ys, xs, v):
        # Validate the input
        length = len(v)
        # if len(v) != length:
        #     raise ValueError("xs and v must have the same length.")

        # First backprop
        grads = tf.gradients(ys, xs)
        grads = self.get_test_grad(grads)

        # grads = xs

        assert len(grads) == length

        elemwise_products = [
            math_ops.multiply(grad_elem, array_ops.stop_gradient(v_elem))
            for grad_elem, v_elem in zip(grads, v) if grad_elem is not None
        ]

        # Second backprop
        grads_with_none = tf.gradients(elemwise_products, xs)
        return_grads = [
            grad_elem if grad_elem is not None \
                else tf.zeros_like(x) \
            for x, grad_elem in zip(xs, grads_with_none)]
        return_grads = self.get_test_grad(return_grads)

        return return_grads

    def get_vec_to_list_fn_test(self):
        params_val = self.sess.run(self.params_test)
        self.num_params = 0
        for param in params_val:
            self.num_params += np.array(param).flatten().shape[0]
        print('Total number of parameters: %s' % self.num_params)

        def vec_to_list(v):
            return_list = []
            cur_pos = 0
            for p in params_val:
                return_list.append(v[cur_pos: cur_pos + len(p)])
                cur_pos += len(p)

            assert cur_pos == len(v)
            return return_list

        return vec_to_list

    def get_fmin_loss_fn(self, v):

        def get_fmin_loss(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list_test(x))

            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)

        return get_fmin_loss

    def get_fmin_grad_fn(self, v):
        def get_fmin_grad(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list_test(x))

            return np.concatenate(hessian_vector_val) - np.concatenate(v)

        return get_fmin_grad

    def get_fmin_hvp(self, x, p):
        hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list_test(p))

        return np.concatenate(hessian_vector_val)

    def get_cg_callback(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)

        def fmin_loss_split(x):
            hessian_vector_val = self.minibatch_hessian_vector_val(self.vec_to_list_test(x))
            return 0.5 * np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)

        def cg_callback(x):
            # x is current params
            v = self.vec_to_list_test(x)
            idx_to_remove = 5

            single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
            train_grad_loss_val = self.sess.run(self.grad_total_loss_op_test, feed_dict=single_train_feed_dict)
            predicted_loss_diff = np.dot(np.concatenate(v),
                                         np.concatenate(train_grad_loss_val)) / self.train_indices_of_test_case.shape[0]

            if verbose:
                print('Function value: %s' % fmin_loss_fn(x))
                quad, lin = fmin_loss_split(x)
                print('Split function value: %s, %s' % (quad, lin))
                print('Predicted loss diff on train_idx %s: %s' % (idx_to_remove, predicted_loss_diff))

        return cg_callback

    def get_inverse_hvp_cg(self, v, verbose):
        fmin_loss_fn = self.get_fmin_loss_fn(v)
        fmin_grad_fn = self.get_fmin_grad_fn(v)
        cg_callback = self.get_cg_callback(v, verbose)

        fmin_results = fmin_ncg(
            f=fmin_loss_fn,
            x0=np.concatenate(v),
            fprime=fmin_grad_fn,
            fhess_p=self.get_fmin_hvp,
            callback=cg_callback,
            avextol=self.avextol,
            maxiter=100)

        return self.vec_to_list_test(fmin_results)
