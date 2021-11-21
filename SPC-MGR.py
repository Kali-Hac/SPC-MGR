"""
The SPC-MGR is built based in part on graph attention mechanism (https://arxiv.org/abs/1710.10903),
part on MG-SCR (https://www.ijcai.org/proceedings/2021/0135),
and includes open-source codes provided by
the project of Graph Attention Network (GAT) at https://github.com/PetarV-/GAT,
and the project of MG-SCR at https://github.com/Kali-Hac/MG-SCR.
"""

import time
import numpy as np
import tensorflow as tf
import os, sys
from models import GAT as MSRL  # (Veličković et al.)
from utils import process_L3 as process
from utils.faiss_rerank import compute_jaccard_distance
from tensorflow.python.layers.core import Dense
from sklearn.preprocessing import label_binarize
from sklearn.cluster import DBSCAN
import torch
import collections
from sklearn.metrics import average_precision_score

dataset = ''
probe = ''
pre_dir = 'ReID_Models/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

nb_nodes = 20       # number of nodes in joint-scale graph
nhood = 1           # structural relation learning (nhood=1 for neighbor nodes)
fusion_lambda = 1   # collaboration fusion coefficient
ft_size = 3         # originial node feature dimension (D)
time_step = 6       # sequence length (f)


# training params
batch_size = 256
nb_epochs = 100000
patience = 250     # patience for early stopping
hid_units = [8]  # numbers of hidden units per each attention head in each layer
Ms = [8, 1]  # additional entry for the output layer
k1, k2 = 20, 6  # parameters to compute feature distance matrix
residual = False
nonlinearity = tf.nn.elu


tf.app.flags.DEFINE_string('dataset', 'KS20', "Dataset: IAS, KS20, BIWI, CASIA-B or KGBD")
tf.app.flags.DEFINE_string('length', '6', "4, 6, 8, 10 or 12")
tf.app.flags.DEFINE_string('t', '0.07', "temperature for contrastive learning")
tf.app.flags.DEFINE_string('lr', '0.00035', "learning rate")
tf.app.flags.DEFINE_string('eps', '0.6', "distance parameter in DBSCAN")
tf.app.flags.DEFINE_string('min_samples', '2', "minimum sample number in DBSCAN")
tf.app.flags.DEFINE_string('probe', 'probe', "for testing probe")
tf.app.flags.DEFINE_string('gpu', '0', "GPU number")
tf.app.flags.DEFINE_string('probe_view', '', "test different views on CASIA B or KS20")
tf.app.flags.DEFINE_string('gallery_view', '', "test different views on CASIA B or KS20")
tf.app.flags.DEFINE_string('struct_only', '0', "struct_only")
tf.app.flags.DEFINE_string('m', '8', "structural relation heads")
tf.app.flags.DEFINE_string('probe_type', '', "probe.gallery")
tf.app.flags.DEFINE_string('patience', '200', "epochs for early stopping")
tf.app.flags.DEFINE_string('fusion_lambda', '1', "collaboration fusion coefficient")
tf.app.flags.DEFINE_string('S_dataset', '', "Source Dataset")
tf.app.flags.DEFINE_string('S_probe', '', "Source Dataset probe")
tf.app.flags.DEFINE_string('mode', 'UF', "Unsupervised Fine-tuning (UF) or Direct Generalization (DG)")
tf.app.flags.DEFINE_string('evaluate', '0', "evaluate on the best model")
FLAGS = tf.app.flags.FLAGS


# check parameters
if FLAGS.dataset not in ['IAS', 'KGBD', 'KS20', 'BIWI', 'CASIA_B']:
	raise Exception('Dataset must be IAS, KGBD, KS20, BIWI or CASIA B.')
if not FLAGS.gpu.isdigit() or int(FLAGS.gpu) < 0:
	raise Exception('GPU number must be a positive integer.')
if FLAGS.dataset == 'CASIA_B':
	pass
else:
	if FLAGS.length not in ['4', '6', '8', '10', '12']:
		raise Exception('Length number must be 4, 6, 8, 10 or 12.')
if FLAGS.probe not in ['probe', 'Walking', 'Still', 'A', 'B']:
	raise Exception('Dataset probe must be "A" (for IAS-A), "B" (for IAS-B), "probe" (for KS20, KGBD).')
if float(FLAGS.fusion_lambda) < 0 or float(FLAGS.fusion_lambda) > 1:
	raise Exception('Multi-Level Graph Fusion coefficient must be not less than 0 or not larger than 1.')
if FLAGS.mode not in ['UF', 'DG']:
	raise Exception('Mode must be UF or DG.')
if FLAGS.mode == 'DG' and FLAGS.S_dataset == '':
	raise Exception('DG mode must set a source dataset.')
if FLAGS.mode == 'UF' and FLAGS.S_dataset != '':
	raise Exception('UF mode does not use a source dataset.')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
dataset = FLAGS.dataset

# optimal paramters
if dataset == 'KGBD':
	batch_size = 256
	FLAGS.lr = '0.00035'
	FLAGS.min_samples = '4'
	FLAGS.t = '0.06'
elif dataset == 'CASIA_B':
	batch_size = 128
	FLAGS.lr = '0.00035'
	FLAGS.min_samples = '2'
	FLAGS.eps = '0.75'
	FLAGS.t = '0.075'
else:
	batch_size = 128
	FLAGS.lr = '0.00035'
if dataset == 'KS20' or dataset == 'IAS':
	FLAGS.t = '0.08'
	FLAGS.eps = '0.8'
elif dataset == 'BIWI':
	FLAGS.t = '0.07'


eps = float(FLAGS.eps)
min_samples = int(FLAGS.min_samples)

time_step = int(FLAGS.length)
fusion_lambda = float(FLAGS.fusion_lambda)
probe = FLAGS.probe
patience = int(FLAGS.patience)


global_att = False
struct_only = False
P = '8'


change = ''



if FLAGS.probe_type != '':
	change += '_CME'
if FLAGS.fusion_lambda != '1':
	change = '_lambda_' + FLAGS.fusion_lambda

if FLAGS.struct_only == '1':
	struct_only = True


if FLAGS.dataset == 'KGBD':
	FLAGS.m = '16'
if FLAGS.m != '8':
	m = FLAGS.m
	Ms = [int(m), 1]

try:
	os.mkdir(pre_dir)
except:
	pass

if struct_only:
	pre_dir += '_struct_only'
if P != '8':
	pre_dir += '_P_' + P


if dataset == 'KS20':
	nb_nodes = 25

if dataset == 'CASIA_B':
	nb_nodes = 14



print('----- Model hyperparams -----')
# print('skeleton_nodes: ' + str(nb_nodes))
print('seqence_length: ' + str(time_step))
print('fusion_lambda: ' + str(fusion_lambda))
print('batch_size: ' + str(batch_size))
print('lr: ' + str(FLAGS.lr))
print('temperature: ' + FLAGS.t)
print('eps: ' + FLAGS.eps)
print('min_samples: ' + FLAGS.min_samples)
print('m: ' + FLAGS.m)
print('fusion_lambda: ' + FLAGS.fusion_lambda)
# print('patience: ' + FLAGS.patience)

print('Mode: ' + FLAGS.mode)
print('Evaluate: ' + FLAGS.evaluate)

if FLAGS.mode == 'DG':
	print('----- Mode Information  -----')
	print('Source Dataset: ' + FLAGS.S_dataset)
	print('Target Dataset: ' + FLAGS.dataset)
	print('Target Probe: ' + FLAGS.probe)
elif FLAGS.mode == 'UF':
	print('----- Dataset Information  -----')
	print('Dataset: ' + dataset)
	if dataset == 'CASIA_B':
		print('Probe.Gallery: ', FLAGS.probe_type.split('.')[0], FLAGS.probe_type.split('.')[1])
	else:
		print('Probe: ' + FLAGS.probe)

"""
 Obtain training and testing data in part-level, body-scale, and hyper-body-scale.
 Generate corresponding adjacent matrix and bias.
"""
if FLAGS.probe_type == '':
	if FLAGS.probe_view == '' and FLAGS.gallery_view == '':
		_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_test_P, X_test_B, X_test_H_B, _, y_test, \
		_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
			process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
			                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size, )
	else:
		if dataset == 'KS20':
			_, _, _, _, _, _, _, X_test_P, X_test_B, X_test_H_B, _, y_test, \
			_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
				process.gen_train_data(dataset=dataset, split='view_'+FLAGS.probe_view, time_step=time_step,
				                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
				                       )
			X_train_P_all = []
			X_train_B_all = []
			X_train_H_B_all = []
			y_train_all = []
			for i in range(5):
				if str(i) not in [FLAGS.probe_view, FLAGS.gallery_view]:
					_, _, _, _, _, _, _, X_train_P, X_train_B, X_train_H_B, _, y_train, \
					_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='view_' + str(i), time_step=time_step,
						                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
						                       batch_size=batch_size,
						                       )
					X_train_H_B_all.extend(X_train_H_B)
					X_train_P_all.extend(X_train_P)
					X_train_B_all.extend(X_train_B)
					y_train_all.extend(y_train_all)
			X_train_P = np.array(X_train_P_all)
			X_train_B = np.array(X_train_B_all)
			X_train_H_B = np.array(X_train_H_B_all)
			y_train = np.array(y_train)

else:
	from utils import process_cme_L3 as process
	_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_test_P, X_test_B, X_test_H_B, _, y_test, \
	_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
		process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
		                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size, PG_type=FLAGS.probe_type.split('.')[0])
	print('## [Probe].[Gallery]', FLAGS.probe_type)


all_ftr_size = hid_units[0] * (15 + 3)
loaded_graph = tf.Graph()

cluster_epochs = 15000
display = 20

if FLAGS.evaluate == '1':
	FLAGS.S_dataset = FLAGS.dataset
	FLAGS.S_probe = FLAGS.probe
	FLAGS.mode = 'DG'

if FLAGS.mode == 'UF':
	with tf.Graph().as_default():
		with tf.name_scope('Input'):
			P_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, 10, ft_size))
			B_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, 5, ft_size))
			H_B_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, 3, ft_size))
			P_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, 10, 10))
			B_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, 5, 5))
			H_B_bias_in = tf.placeholder(dtype=tf.float32, shape=(1, 3, 3))
			attn_drop = tf.placeholder(dtype=tf.float32, shape=())
			ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
			is_train = tf.placeholder(dtype=tf.bool, shape=())
			pseudo_lab = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
			cluster_ftr = tf.placeholder(dtype=tf.float32, shape=(None, all_ftr_size))

		with tf.name_scope("MG"), tf.variable_scope("MG", reuse=tf.AUTO_REUSE):
			def SRL(J_in, J_bias_in, nb_nodes):
				W_h = tf.Variable(tf.random_normal([3, hid_units[-1]]))
				b_h = tf.Variable(tf.zeros(shape=[hid_units[-1], ]))
				J_h = tf.reshape(J_in, [-1, ft_size])

				J_h = tf.matmul(J_h, W_h) + b_h
				J_h = tf.reshape(J_h, [batch_size*time_step, nb_nodes, hid_units[-1]])
				J_seq_ftr = MSRL.inference(J_h, 0, nb_nodes, is_train,
				                         attn_drop, ffd_drop,
				                         bias_mat=J_bias_in,
				                         hid_units=hid_units, n_heads=Ms,
				                         residual=residual, activation=nonlinearity, r_pool=True)
				return J_seq_ftr


			def FCRL(s1, s2, s1_num, s2_num, hid_in):
				r_unorm = tf.matmul(s2, tf.transpose(s1, [0, 2, 1]))
				att_w = tf.nn.softmax(r_unorm)
				att_w = tf.expand_dims(att_w, axis=-1)
				s1 = tf.reshape(s1, [s1.shape[0], 1, s1.shape[1], hid_in])
				c_ftr = tf.reduce_sum(att_w * s1, axis=2)
				c_ftr = tf.reshape(c_ftr, [-1, hid_in])
				att_w = tf.reshape(att_w, [-1, s1_num * s2_num])
				return r_unorm, c_ftr


			def MSC(P_in, B_in, H_B_in, P_bias_in, B_bias_in,  H_B_bias_in, hid_in, hid_out):
				h_P_seq_ftr = SRL(J_in=P_in, J_bias_in=P_bias_in, nb_nodes=10)
				h_B_seq_ftr = SRL(J_in=B_in, J_bias_in=B_bias_in, nb_nodes=5)
				h_H_B_seq_ftr = SRL(J_in=H_B_in, J_bias_in=H_B_bias_in, nb_nodes=3)

				h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, 10, hid_in])
				h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, 5, hid_in])
				h_H_B_seq_ftr = tf.reshape(h_H_B_seq_ftr, [-1, 3, hid_in])


				W_cs_23 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_cs_24 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_cs_34 = tf.Variable(tf.random_normal([hid_in, hid_out]))


				W_self_2 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_self_3 = tf.Variable(tf.random_normal([hid_in, hid_out]))
				W_self_4 = tf.Variable(tf.random_normal([hid_in, hid_out]))

				self_a_2, self_r_2 = FCRL(h_P_seq_ftr, h_P_seq_ftr, 10, 10, hid_in)
				self_a_3, self_r_3 = FCRL(h_B_seq_ftr, h_B_seq_ftr, 5, 5, hid_in)
				self_a_4, self_r_4 = FCRL(h_H_B_seq_ftr, h_H_B_seq_ftr, 3, 3, hid_in)

				h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, hid_in])
				h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, hid_in])
				h_H_B_seq_ftr = tf.reshape(h_H_B_seq_ftr, [-1, hid_in])


				h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, 10, hid_in])
				h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, 5, hid_in])
				h_H_B_seq_ftr = tf.reshape(h_H_B_seq_ftr, [-1, 3, hid_in])


				a_23, r_23 = FCRL(h_B_seq_ftr, h_P_seq_ftr, 5, 10, hid_in)
				a_24, r_24 = FCRL(h_H_B_seq_ftr, h_P_seq_ftr, 3, 10, hid_in)
				a_34, r_34 = FCRL(h_H_B_seq_ftr, h_B_seq_ftr, 3, 5, hid_in)


				h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, hid_in])
				h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, hid_in])
				h_H_B_seq_ftr = tf.reshape(h_H_B_seq_ftr, [-1, hid_in])

				if not struct_only:
					h_P_seq_ftr = h_P_seq_ftr + float(FLAGS.fusion_lambda) * (
								tf.matmul(self_r_2, W_self_2) + tf.matmul(r_23, W_cs_23) + tf.matmul(r_24, W_cs_24))
					h_B_seq_ftr = h_B_seq_ftr + float(FLAGS.fusion_lambda) * (tf.matmul(self_r_3, W_self_3) + tf.matmul(r_34, W_cs_34))
					h_H_B_seq_ftr = h_H_B_seq_ftr + float(FLAGS.fusion_lambda) * (tf.matmul(self_r_4, W_self_4))

				h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, 10,  hid_out])
				h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, 5,  hid_out])
				h_H_B_seq_ftr = tf.reshape(h_H_B_seq_ftr, [-1, 3, hid_out])

				return h_H_B_seq_ftr, h_B_seq_ftr, h_P_seq_ftr

			h_H_B_seq_ftr, h_B_seq_ftr, h_P_seq_ftr = MSC(P_in, B_in, H_B_in, P_bias_in, B_bias_in, H_B_bias_in,
			                                            hid_units[-1], hid_units[-1])

			h_P_seq_ftr = tf.reshape(h_P_seq_ftr, [-1, hid_units[-1]])
			h_B_seq_ftr = tf.reshape(h_B_seq_ftr, [-1, hid_units[-1]])
			h_H_B_seq_ftr = tf.reshape(h_H_B_seq_ftr, [-1, hid_units[-1]])

			optimizer = tf.train.AdamOptimizer(learning_rate=float(FLAGS.lr))
			P_encode = tf.reduce_mean(tf.reshape(h_P_seq_ftr, [batch_size, time_step, -1]), axis=1)
			B_encode = tf.reduce_mean(tf.reshape(h_B_seq_ftr, [batch_size, time_step, -1]), axis=1)
			H_B_encode = tf.reduce_mean(tf.reshape(h_H_B_seq_ftr, [batch_size, time_step, -1]), axis=1)

			P_encode = tf.reshape(P_encode, [batch_size, -1])
			B_encode = tf.reshape(B_encode, [batch_size, -1])
			H_B_encode = tf.reshape(H_B_encode, [batch_size, -1])

			all_ftr = tf.concat([P_encode, B_encode, H_B_encode], axis=-1)
			all_ftr = tf.reshape(all_ftr, [batch_size, -1])

			output = tf.matmul(all_ftr, tf.transpose(cluster_ftr))

			def cluster_loss(pseudo_lab, all_ftr, cluster_ftr):
				all_ftr = tf.nn.l2_normalize(all_ftr, axis=-1)
				cluster_ftr = tf.nn.l2_normalize(cluster_ftr, axis=-1)
				output = tf.matmul(all_ftr, tf.transpose(cluster_ftr))
				output /= float(FLAGS.t)
				loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pseudo_lab, logits=output))
				return loss


			def empty_loss(b):
				return tf.zeros([1])


			contrastive_loss = tf.cond(tf.reduce_sum(pseudo_lab) > 0,
			                           lambda: cluster_loss(pseudo_lab, all_ftr, cluster_ftr),
			                           lambda: empty_loss(pseudo_lab))
			cluster_train_op = optimizer.minimize(contrastive_loss)



		saver = tf.train.Saver()
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		with tf.Session(config=config) as sess:
			sess.run(init_op)
			def train_loader(X_train_P, X_train_B, X_train_H_B, y_train):
				tr_step = 0
				tr_size = X_train_P.shape[0]
				train_logits_all = []
				train_labels_all = []
				train_features_all = []
				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_P = X_train_P[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_P = X_input_P.reshape([-1, 10, 3])
					X_input_B = X_train_B[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_B = X_input_B.reshape([-1, 5, 3])
					X_input_H_B = X_train_H_B[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_H_B = X_input_H_B.reshape([-1, 3, 3])
					labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
					P_en, B_en, all_features = sess.run([P_encode, B_encode, all_ftr],
					                                                          feed_dict={
						                                                          P_in: X_input_P,
						                                                          B_in: X_input_B,
						                                                          H_B_in: X_input_H_B,
						                                                          P_bias_in: biases_P,
						                                                          B_bias_in: biases_B,
						                                                          H_B_bias_in: biases_H_B,
						                                                          is_train: True,
						                                                          attn_drop: 0.0, ffd_drop: 0.0,
						                                                          pseudo_lab: np.zeros([batch_size, ]),
						                                                          cluster_ftr: np.zeros(
							                                                          [batch_size, all_ftr_size])})
					train_features_all.extend(all_features.tolist())
					train_labels_all.extend(labels.tolist())
					tr_step += 1

				train_features_all = np.array(train_features_all).astype(np.float32)
				train_features_all = torch.from_numpy(train_features_all)
				return train_features_all, train_labels_all


			def gal_loader(X_train_P, X_train_B, X_train_H_B, y_train):
				tr_step = 0
				tr_size = X_train_P.shape[0]
				gal_logits_all = []
				gal_labels_all = []
				gal_features_all = []

				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_P = X_train_P[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_P = X_input_P.reshape([-1, 10, 3])
					X_input_B = X_train_B[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_B = X_input_B.reshape([-1, 5, 3])
					X_input_H_B = X_train_H_B[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_H_B = X_input_H_B.reshape([-1, 3, 3])
					labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]

					P_en, B_en, all_features = sess.run([P_encode, B_encode, all_ftr],
					                                                          feed_dict={
						                                                          P_in: X_input_P,
						                                                          B_in: X_input_B,
						                                                          H_B_in: X_input_H_B,
						                                                          P_bias_in: biases_P,
						                                                          B_bias_in: biases_B,
						                                                          H_B_bias_in: biases_H_B,
						                                                          is_train: True,
						                                                          attn_drop: 0.0, ffd_drop: 0.0,
						                                                          pseudo_lab: np.zeros([batch_size, ]),
						                                                          cluster_ftr: np.zeros(
							                                                          [batch_size, all_ftr_size])})
					gal_features_all.extend(all_features.tolist())
					gal_labels_all.extend(labels.tolist())
					tr_step += 1

				return gal_features_all, gal_labels_all


			def evaluation():
				vl_step = 0
				vl_size = X_test_P.shape[0]
				pro_labels_all = []
				pro_features_all = []
				loaded_graph = tf.get_default_graph()
				while vl_step * batch_size < vl_size:
					if (vl_step + 1) * batch_size > vl_size:
						break
					X_input_P = X_test_P[vl_step * batch_size:(vl_step + 1) * batch_size]
					X_input_P = X_input_P.reshape([-1, 10, 3])
					X_input_B = X_test_B[vl_step * batch_size:(vl_step + 1) * batch_size]
					X_input_B = X_input_B.reshape([-1, 5, 3])
					X_input_H_B = X_test_H_B[vl_step * batch_size:(vl_step + 1) * batch_size]
					X_input_H_B = X_input_H_B.reshape([-1, 3, 3])
					labels = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]
					P_en, B_en, all_features = sess.run([P_encode, B_encode, all_ftr],
					                                                          feed_dict={
						                                                          P_in: X_input_P,
						                                                          B_in: X_input_B,
						                                                          H_B_in: X_input_H_B,
						                                                          P_bias_in: biases_P,
						                                                          B_bias_in: biases_B,
						                                                          H_B_bias_in: biases_H_B,
						                                                          is_train: False,
						                                                          attn_drop: 0.0, ffd_drop: 0.0,
						                                                          pseudo_lab: np.zeros([batch_size, ]),
						                                                          cluster_ftr: np.zeros(
							                                                          [batch_size, all_ftr_size])})
					pro_labels_all.extend(labels.tolist())
					pro_features_all.extend(all_features.tolist())
					vl_step += 1
				X = np.array(gal_features_all)
				y = np.array(gal_labels_all)
				t_X = np.array(pro_features_all)
				t_y = np.array(pro_labels_all)
				# print(X.shape, t_X.shape)
				t_y = np.argmax(t_y, axis=-1)
				y = np.argmax(y, axis=-1)

				def mean_ap(distmat, query_ids=None, gallery_ids=None,
				            query_cams=None, gallery_cams=None):
					# distmat = to_numpy(distmat)
					m, n = distmat.shape
					# Fill up default values
					if query_ids is None:
						query_ids = np.arange(m)
					if gallery_ids is None:
						gallery_ids = np.arange(n)
					if query_cams is None:
						query_cams = np.zeros(m).astype(np.int32)
					if gallery_cams is None:
						gallery_cams = np.ones(n).astype(np.int32)
					# Ensure numpy array
					query_ids = np.asarray(query_ids)
					gallery_ids = np.asarray(gallery_ids)
					query_cams = np.asarray(query_cams)
					gallery_cams = np.asarray(gallery_cams)
					# Sort and find correct matches
					indices = np.argsort(distmat, axis=1)
					matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
					# Compute AP for each query
					aps = []
					if (FLAGS.probe_view != '' and (FLAGS.probe_view == FLAGS.gallery_view or FLAGS.probe_type == 'nm.nm')) or (FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
						for i in range(1, m):
							valid = ((gallery_ids[indices[i]] != query_ids[i]) |
							         (gallery_cams[indices[i]] != query_cams[i]))
							y_true = matches[i, valid]
							y_score = -distmat[i][indices[i]][valid]
							if not np.any(y_true): continue
							aps.append(average_precision_score(y_true, y_score))
					else:
						for i in range(m):
							valid = ((gallery_ids[indices[i]] != query_ids[i]) |
							         (gallery_cams[indices[i]] != query_cams[i]))
							y_true = matches[i, valid]
							y_score = -distmat[i][indices[i]][valid]
							if not np.any(y_true): continue
							aps.append(average_precision_score(y_true, y_score))
					if len(aps) == 0:
						raise RuntimeError("No valid query")
					return np.mean(aps)


				def metrics(X, y, t_X, t_y):
					# compute Euclidean distance
					if dataset != 'CASIA_B':
						a, b = torch.from_numpy(t_X), torch.from_numpy(X)
						m, n = a.size(0), b.size(0)
						a = a.view(m, -1)
						b = b.view(n, -1)
						dist_m = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
						         torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
						dist_m.addmm_(1, -2, a, b.t())
						dist_m = dist_m.sqrt()
						mAP = mean_ap(distmat=dist_m.numpy(), query_ids=t_y, gallery_ids=y)
						_, dist_sort = dist_m.sort(1)
						dist_sort = dist_sort.numpy()
					else:
						X = np.array(X)
						t_X = np.array(t_X)
						# pred = [cp.argmin(cp.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
						dist_m = [(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
						dist_m = np.array(dist_m)
						mAP = mean_ap(distmat=dist_m, query_ids=t_y, gallery_ids=y)
						dist_sort = [np.argsort(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
						dist_sort = np.array(dist_sort)

					top_1 = top_5 = top_10 = 0
					probe_num = dist_sort.shape[0]
					if (FLAGS.probe_view != '' and (FLAGS.probe_view == FLAGS.gallery_view or FLAGS.probe_type == 'nm.nm')) or (FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
						for i in range(probe_num):
							# print(dist_sort[i, :10])
							if t_y[i] in y[dist_sort[i, 1:2]]:
								top_1 += 1
							if t_y[i] in y[dist_sort[i, 1:6]]:
								top_5 += 1
							if t_y[i] in y[dist_sort[i, 1:11]]:
								top_10 += 1
					else:
						for i in range(probe_num):
							# print(dist_sort[i, :10])
							if t_y[i] in y[dist_sort[i, :1]]:
								top_1 += 1
							if t_y[i] in y[dist_sort[i, :5]]:
								top_5 += 1
							if t_y[i] in y[dist_sort[i, :10]]:
								top_10 += 1
					return mAP, top_1 / probe_num, top_5 / probe_num, top_10 / probe_num

				mAP, top_1, top_5, top_10 = metrics(X, y, t_X, t_y)
				return mAP, top_1, top_5, top_10

			max_acc_1 = 0
			max_acc_2 = 0
			best_cluster_info_1 = [0, 0]
			best_cluster_info_2 = [0, 0]
			cur_patience = 0
			if dataset == 'KGBD' or dataset == 'KS20':
				if FLAGS.gallery_view == '' and FLAGS.probe_view == '':
					_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
					_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
						                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
						                       )
				else:
					_, _, _, _, _, _, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
					_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, b_, nb_classes = \
						process.gen_train_data(dataset=dataset, split='view_'+FLAGS.gallery_view, time_step=time_step,
						                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
						                       batch_size=batch_size,
						                       )
			elif dataset == 'BIWI':
				if probe == 'Walking':
					_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
					_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
						                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
						                       )
				else:
					_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
					_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
						                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
						                       batch_size=batch_size,
						                       )
			elif dataset == 'IAS':
				if probe == 'A':
					_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
					_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
						                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
						                       )
				else:
					_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
					_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
						process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
						                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
						                       batch_size=batch_size,
						                       )
			elif dataset == 'CASIA_B':
				_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
				_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
					                       PG_type=FLAGS.probe_type.split('.')[1])
			for epoch in range(cluster_epochs):
				train_features_all, train_labels_all = train_loader(X_train_P, X_train_B, X_train_H_B, y_train)
				gal_features_all, gal_labels_all = gal_loader(X_gal_P, X_gal_B, X_gal_H_B, y_gal)
				mAP, top_1, top_5, top_10 = evaluation()
				cur_patience += 1
				if epoch > 0 and top_1 > max_acc_2:
					max_acc_1 = mAP
					best_cluster_info_1[0] = num_cluster
					best_cluster_info_1[1] = outlier_num
					cur_patience = 0
					if FLAGS.mode == 'UF' and FLAGS.S_dataset == '':
						if FLAGS.probe_view == '' and FLAGS.gallery_view == '' and FLAGS.dataset != 'CASIA_B':
							# checkpt_file = pre_dir + dataset + '/' + probe + '_' + str(fusion_lambda) + '_' + str(
							# 	nhood) + '_' + str(
							# 	time_step) + '_' +  FLAGS.min_samples + '_' + FLAGS.lr + '_' + FLAGS.eps + '_' + \
							#                FLAGS.t + '_' + change + '_best.ckpt'
							checkpt_file = pre_dir + dataset + '/' + probe  + change + '_best.ckpt'
						elif FLAGS.dataset == 'CASIA_B':
							# checkpt_file = pre_dir + dataset + '/' + probe + '_' + str(fusion_lambda) + '_' + str(
							# 	nhood) + '_' + str(
							# 	time_step) + '_' + FLAGS.min_samples + '_' + FLAGS.lr + '_' + FLAGS.eps + '_' + \
							#                FLAGS.t + '_' + change + '_' + FLAGS.probe_type + '_best.ckpt'
							checkpt_file = pre_dir + dataset + '/' + probe + change + '_' + FLAGS.probe_type + '_best.ckpt'
						else:
							# checkpt_file = pre_dir + dataset + '/' + probe + '_' + str(fusion_lambda) + '_' + str(
							# 	nhood) + '_' + str(
							# 	time_step) + '_' +  FLAGS.min_samples + '_' + FLAGS.lr + '_' + FLAGS.eps + '_' + \
							#                FLAGS.t + '_' + FLAGS.probe_view + 'v' + FLAGS.gallery_view + change + '_best.ckpt'
							checkpt_file = pre_dir + dataset + '/' + probe + '_' + FLAGS.probe_view + 'v' + FLAGS.gallery_view + change + '_best.ckpt'
						print(checkpt_file)
						saver.save(sess, checkpt_file)
				if epoch > 0 and top_1 > max_acc_2:
					max_acc_2 = top_1
					best_cluster_info_2[0] = num_cluster
					best_cluster_info_2[1] = outlier_num
					cur_patience = 0
				if epoch > 0:
					if FLAGS.probe_view != '' and FLAGS.gallery_view != '':
						print('[UF] View: %s v %s | mAP: %.4f (%.4f) | Top-1: %.4f (%.4f) | Top-5: %.4f | Top-10: %.4f | % d + o: %d |' % (
							FLAGS.probe_view, FLAGS.gallery_view, mAP, max_acc_1,
							top_1, max_acc_2, top_5, top_10,
							best_cluster_info_2[0], best_cluster_info_2[1]))
					else:
						print(
							'[UF] %s - %s | mAP: %.4f (%.4f) | Top-1: %.4f (%.4f) | Top-5: %.4f | Top-10: %.4f | % d + o: %d |' % (
							FLAGS.dataset, FLAGS.probe, mAP, max_acc_1,
							top_1, max_acc_2, top_5, top_10,
							best_cluster_info_2[0], best_cluster_info_2[1]))
				if cur_patience == patience:
					break
				rerank_dist = compute_jaccard_distance(train_features_all, k1=k1, k2=k2)

				if dataset == 'IAS' or dataset == 'KS20':
					cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
				else:
					cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
				pseudo_labels = cluster.fit_predict(rerank_dist)
				# discard outliers
				train_features_all = train_features_all[np.where(pseudo_labels != -1)]
				X_train_P_new = X_train_P[np.where(pseudo_labels != -1)]
				X_train_B_new = X_train_B[np.where(pseudo_labels != -1)]
				X_train_H_B_new = X_train_H_B[np.where(pseudo_labels != -1)]
				outlier_num = np.sum(pseudo_labels == -1)
				pseudo_labels = pseudo_labels[np.where(pseudo_labels != -1)]
				num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)


				def generate_cluster_features(labels, features):
					centers = collections.defaultdict(list)
					for i, label in enumerate(labels):
						if label == -1:
							continue
						centers[labels[i]].append(features[i])

					centers = [
						torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
					]
					centers = torch.stack(centers, dim=0)
					return centers


				cluster_features = generate_cluster_features(pseudo_labels, train_features_all)
				cluster_features = cluster_features.numpy()
				cluster_features = cluster_features.astype(np.float64)

				tr_step = 0
				tr_size = X_train_P_new.shape[0]
				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_P = X_train_P_new[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_P = X_input_P.reshape([-1, 10, 3])
					X_input_B = X_train_B_new[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_B = X_input_B.reshape([-1, 5, 3])
					X_input_H_B = X_train_H_B_new[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_H_B = X_input_H_B.reshape([-1, 3, 3])
					labels = pseudo_labels[tr_step * batch_size:(tr_step + 1) * batch_size]
					_, loss, P_en, B_en, all_features  = sess.run(
						[cluster_train_op, contrastive_loss, P_encode, B_encode, all_ftr],
						feed_dict={
							P_in: X_input_P,
							B_in: X_input_B,
							H_B_in: X_input_H_B,
							P_bias_in: biases_P,
							B_bias_in: biases_B,
							H_B_bias_in: biases_H_B,
							is_train: True,
							attn_drop: 0.0, ffd_drop: 0.0,
							pseudo_lab: labels,
							cluster_ftr: cluster_features})
					if tr_step % display == 0:
						print('[%s] Batch num: %d | Cluser num: %d | Outlier: %d | Loss: %.5f |' %
						      (str(epoch), tr_step, num_cluster, outlier_num, loss))
					tr_step += 1
			sess.close()

elif FLAGS.mode == 'DG' and FLAGS.S_dataset != '':
	if FLAGS.S_dataset == 'KGBD':
		batch_size = 256
		FLAGS.lr = '0.00035'
		FLAGS.min_samples = '4'
		FLAGS.eps = '0.6'
	elif FLAGS.S_dataset == 'CASIA_B':
		batch_size = 128
		FLAGS.lr = '0.00035'
		FLAGS.min_samples = '2'
		FLAGS.eps = '0.75'
	else:
		batch_size = 128
		FLAGS.lr = '0.00035'
	if FLAGS.S_dataset == 'IAS' or FLAGS.S_dataset == 'KS20':
		# if FLAGS.mode != 'DG':
		FLAGS.eps = '0.8'
		FLAGS.min_samples = '2'
		if FLAGS.S_dataset == 'KS20':
			FLAGS.min_samples = '2'

	if FLAGS.S_dataset == 'BIWI':
		FLAGS.min_samples = '2'
		if FLAGS.S_probe == 'Walking':
			FLAGS.eps = '0.6'
		else:
			FLAGS.eps = '0.7'
	# checkpt_file = pre_dir + FLAGS.S_dataset + '/' + FLAGS.S_probe + '_' + str(fusion_lambda) + '_' + str(
	# 	nhood) + '_' + str(
	# 	time_step) + '_' +  FLAGS.min_samples + '_' + FLAGS.lr + '_' + FLAGS.eps + '_' + \
	# 						               FLAGS.t + '_' + change + '_best.ckpt'
	checkpt_file = pre_dir + dataset + '/' + probe + change + '_best.ckpt'
	change = '_DG'
	with tf.Session(graph=loaded_graph, config=config) as sess:
		loader = tf.train.import_meta_graph(checkpt_file + '.meta')
		P_in = loaded_graph.get_tensor_by_name("Input/Placeholder:0")
		B_in = loaded_graph.get_tensor_by_name("Input/Placeholder_1:0")
		H_B_in = loaded_graph.get_tensor_by_name("Input/Placeholder_2:0")
		P_bias_in = loaded_graph.get_tensor_by_name("Input/Placeholder_3:0")
		B_bias_in = loaded_graph.get_tensor_by_name("Input/Placeholder_4:0")
		H_B_bias_in = loaded_graph.get_tensor_by_name("Input/Placeholder_5:0")
		attn_drop = loaded_graph.get_tensor_by_name("Input/Placeholder_6:0")
		ffd_drop = loaded_graph.get_tensor_by_name("Input/Placeholder_7:0")
		is_train = loaded_graph.get_tensor_by_name("Input/Placeholder_8:0")
		pseudo_lab = loaded_graph.get_tensor_by_name("Input/Placeholder_9:0")
		cluster_ftr = loaded_graph.get_tensor_by_name("Input/Placeholder_10:0")

		P_encode = loaded_graph.get_tensor_by_name("MG/MG/Reshape_45:0")
		B_encode = loaded_graph.get_tensor_by_name("MG/MG/Reshape_46:0")
		H_B_encode = loaded_graph.get_tensor_by_name("MG/MG/Reshape_47:0")
		all_ftr = loaded_graph.get_tensor_by_name("MG/MG/Reshape_48:0")

		contrastive_loss = loaded_graph.get_tensor_by_name("MG/MG/cond/Merge:0")
		cluster_train_op = loaded_graph.get_operation_by_name("MG/MG/Adam")

		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		loader.restore(sess, checkpt_file)
		saver = tf.train.Saver()


		def train_loader(X_train_P, X_train_B, X_train_H_B, y_train):
			tr_step = 0
			tr_size = X_train_P.shape[0]
			train_logits_all = []
			train_labels_all = []
			train_features_all = []
			while tr_step * batch_size < tr_size:
				if (tr_step + 1) * batch_size > tr_size:
					break

				X_input_P = X_train_P[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_P = X_input_P.reshape([-1, 10, 3])
				X_input_B = X_train_B[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_B = X_input_B.reshape([-1, 5, 3])
				X_input_H_B = X_train_H_B[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_H_B = X_input_H_B.reshape([-1, 3, 3])
				labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
				P_en, B_en, all_features = sess.run([P_encode, B_encode, all_ftr],
				                                    feed_dict={
					                                    P_in: X_input_P,
					                                    B_in: X_input_B,
					                                    H_B_in: X_input_H_B,
					                                    P_bias_in: biases_P,
					                                    B_bias_in: biases_B,
					                                    H_B_bias_in: biases_H_B,
					                                    is_train: True,
					                                    attn_drop: 0.0, ffd_drop: 0.0,
					                                    pseudo_lab: np.zeros([batch_size, ]),
					                                    cluster_ftr: np.zeros(
						                                    [batch_size, all_ftr_size])})
				train_features_all.extend(all_features.tolist())
				train_labels_all.extend(labels.tolist())
				tr_step += 1

			train_features_all = np.array(train_features_all).astype(np.float32)
			train_features_all = torch.from_numpy(train_features_all)
			return train_features_all, train_labels_all


		def gal_loader(X_train_P, X_train_B, X_train_H_B, y_train):
			tr_step = 0
			tr_size = X_train_P.shape[0]
			gal_logits_all = []
			gal_labels_all = []
			gal_features_all = []
			while tr_step * batch_size < tr_size:
				if (tr_step + 1) * batch_size > tr_size:
					break

				X_input_P = X_train_P[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_P = X_input_P.reshape([-1, 10, 3])
				X_input_B = X_train_B[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_B = X_input_B.reshape([-1, 5, 3])
				X_input_H_B = X_train_H_B[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_H_B = X_input_H_B.reshape([-1, 3, 3])
				labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]
				P_en, B_en, all_features = sess.run([P_encode, B_encode, all_ftr],
				                                    feed_dict={
					                                    P_in: X_input_P,
					                                    B_in: X_input_B,
					                                    H_B_in: X_input_H_B,
					                                    P_bias_in: biases_P,
					                                    B_bias_in: biases_B,
					                                    H_B_bias_in: biases_H_B,
					                                    is_train: True,
					                                    attn_drop: 0.0, ffd_drop: 0.0,
					                                    pseudo_lab: np.zeros([batch_size, ]),
					                                    cluster_ftr: np.zeros(
						                                    [batch_size, all_ftr_size])})
				gal_features_all.extend(all_features.tolist())
				gal_labels_all.extend(labels.tolist())
				tr_step += 1

			return gal_features_all, gal_labels_all


		def evaluation():
			vl_step = 0
			vl_size = X_test_P.shape[0]
			pro_labels_all = []
			pro_features_all = []
			loaded_graph = tf.get_default_graph()
			while vl_step * batch_size < vl_size:
				if (vl_step + 1) * batch_size > vl_size:
					break

				X_input_P = X_test_P[vl_step * batch_size:(vl_step + 1) * batch_size]
				X_input_P = X_input_P.reshape([-1, 10, 3])
				X_input_B = X_test_B[vl_step * batch_size:(vl_step + 1) * batch_size]
				X_input_B = X_input_B.reshape([-1, 5, 3])
				X_input_H_B = X_test_H_B[vl_step * batch_size:(vl_step + 1) * batch_size]
				X_input_H_B = X_input_H_B.reshape([-1, 3, 3])
				labels = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]
				P_en, B_en, all_features = sess.run([P_encode, B_encode, all_ftr],
				                                    feed_dict={
					                                    P_in: X_input_P,
					                                    B_in: X_input_B,
					                                    H_B_in: X_input_H_B,
					                                    P_bias_in: biases_P,
					                                    B_bias_in: biases_B,
					                                    H_B_bias_in: biases_H_B,
					                                    is_train: False,
					                                    attn_drop: 0.0, ffd_drop: 0.0,
					                                    pseudo_lab: np.zeros([batch_size, ]),
					                                    cluster_ftr: np.zeros(
						                                    [batch_size, all_ftr_size])})
				pro_labels_all.extend(labels.tolist())
				pro_features_all.extend(all_features.tolist())
				vl_step += 1
			X = np.array(gal_features_all)
			y = np.array(gal_labels_all)
			t_X = np.array(pro_features_all)
			t_y = np.array(pro_labels_all)
			# print(X.shape, t_X.shape)
			t_y = np.argmax(t_y, axis=-1)
			y = np.argmax(y, axis=-1)

			def mean_ap(distmat, query_ids=None, gallery_ids=None,
			            query_cams=None, gallery_cams=None):
				# distmat = to_numpy(distmat)
				m, n = distmat.shape
				# Fill up default values
				if query_ids is None:
					query_ids = np.arange(m)
				if gallery_ids is None:
					gallery_ids = np.arange(n)
				if query_cams is None:
					query_cams = np.zeros(m).astype(np.int32)
				if gallery_cams is None:
					gallery_cams = np.ones(n).astype(np.int32)
				# Ensure numpy array
				query_ids = np.asarray(query_ids)
				gallery_ids = np.asarray(gallery_ids)
				query_cams = np.asarray(query_cams)
				gallery_cams = np.asarray(gallery_cams)
				# Sort and find correct matches
				indices = np.argsort(distmat, axis=1)
				matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
				# Compute AP for each query
				aps = []
				if (FLAGS.probe_view != '' and (FLAGS.probe_view == FLAGS.gallery_view or FLAGS.probe_type == 'nm.nm')) or (FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
					for i in range(1, m):
						valid = ((gallery_ids[indices[i]] != query_ids[i]) |
						         (gallery_cams[indices[i]] != query_cams[i]))
						y_true = matches[i, valid]
						y_score = -distmat[i][indices[i]][valid]
						if not np.any(y_true): continue
						aps.append(average_precision_score(y_true, y_score))
				else:
					for i in range(m):
						valid = ((gallery_ids[indices[i]] != query_ids[i]) |
						         (gallery_cams[indices[i]] != query_cams[i]))
						y_true = matches[i, valid]
						y_score = -distmat[i][indices[i]][valid]
						if not np.any(y_true): continue
						aps.append(average_precision_score(y_true, y_score))
				if len(aps) == 0:
					raise RuntimeError("No valid query")
				return np.mean(aps)

			def metrics(X, y, t_X, t_y):
				a, b = torch.from_numpy(t_X), torch.from_numpy(X)
				# compute Euclidean distance
				m, n = a.size(0), b.size(0)
				a = a.view(m, -1)
				b = b.view(n, -1)
				dist_m = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
				         torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
				dist_m.addmm_(1, -2, a, b.t())
				dist_m = dist_m.sqrt()

				mAP = mean_ap(distmat=dist_m.numpy(), query_ids=t_y, gallery_ids=y)
				_, dist_sort = dist_m.sort(1)
				dist_sort = dist_sort.numpy()

				top_1 = top_5 = top_10 = 0
				probe_num = dist_sort.shape[0]
				if (FLAGS.probe_view != '' and (FLAGS.probe_view == FLAGS.gallery_view or FLAGS.probe_type == 'nm.nm')) or (FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
					for i in range(probe_num):
						# print(dist_sort[i, :10])
						if t_y[i] in y[dist_sort[i, 1:2]]:
							top_1 += 1
						if t_y[i] in y[dist_sort[i, 1:6]]:
							top_5 += 1
						if t_y[i] in y[dist_sort[i, 1:11]]:
							top_10 += 1
				else:
					for i in range(probe_num):
						# print(dist_sort[i, :10])
						if t_y[i] in y[dist_sort[i, :1]]:
							top_1 += 1
						if t_y[i] in y[dist_sort[i, :5]]:
							top_5 += 1
						if t_y[i] in y[dist_sort[i, :10]]:
							top_10 += 1
				return mAP, top_1 / probe_num, top_5 / probe_num, top_10 / probe_num

			mAP, top_1, top_5, top_10 = metrics(X, y, t_X, t_y)
			return mAP, top_1, top_5, top_10


		max_acc_1 = 0
		max_acc_2 = 0
		best_cluster_info_1 = [0, 0]
		best_cluster_info_2 = [0, 0]
		cur_patience = 0
		if dataset == 'KGBD' or dataset == 'KS20':
			if FLAGS.gallery_view == '' and FLAGS.probe_view == '':
				_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
				_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
					                       )
			else:
				_, _, _, _, _, _, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
				_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='view_' + FLAGS.gallery_view, time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
					                       batch_size=batch_size,
					                       )
		elif dataset == 'BIWI':
			if probe == 'Walking':
				_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
				_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
					                       )
			else:
				_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
				_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
					                       batch_size=batch_size,
					                       )
		elif dataset == 'IAS':
			if probe == 'A':
				_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
				_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
					                       )
			else:
				_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
				_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
					                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
					                       batch_size=batch_size,
					                       )
		elif dataset == 'CASIA_B':
			_, X_train_P, X_train_B, X_train_H_B, _, y_train, _, X_gal_P, X_gal_B, X_gal_H_B, _, y_gal, \
			_, _, adj_P, biases_P, adj_B, biases_B, adj_H_B, biases_H_B, _, _, nb_classes = \
				process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
				                       nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
				                       PG_type=FLAGS.probe_type.split('.')[1])
		for epoch in range(cluster_epochs):
			train_features_all, train_labels_all = train_loader(X_train_P, X_train_B, X_train_H_B, y_train)
			# train_features_all = train_features_all.numpy()
			gal_features_all, gal_labels_all = gal_loader(X_gal_P, X_gal_B, X_gal_H_B, y_gal)
			mAP, top_1, top_5, top_10 = evaluation()
			cur_patience += 1
			if epoch > 0 and top_1 > max_acc_2:
				max_acc_1 = mAP
				best_cluster_info_1[0] = num_cluster
				best_cluster_info_1[1] = outlier_num
				cur_patience = 0
				if FLAGS.mode == 'DG' and FLAGS.S_dataset != '':
					if FLAGS.probe_view == '' and FLAGS.gallery_view == '':
						# checkpt_file = pre_dir + dataset + '/' + probe + '_' + str(fusion_lambda) + '_' + str(
						# 	nhood) + '_' + str(
						# 	time_step) + '_' + FLAGS.min_samples + '_' + FLAGS.lr + '_' + FLAGS.eps + '_' + \
						#                FLAGS.density_lambda + '_' + change + '_best.ckpt'
						checkpt_file = pre_dir + dataset + '/' + probe + change + '_best.ckpt'
					else:
						checkpt_file = pre_dir + dataset + '/' + probe + '_' + FLAGS.probe_view + 'v' + \
						               FLAGS.gallery_view + change + '_best.ckpt'
					print(checkpt_file)
					saver.save(sess, checkpt_file)
			if epoch > 0 and top_1 > max_acc_2:
				max_acc_2 = top_1
				best_cluster_info_2[0] = num_cluster
				best_cluster_info_2[1] = outlier_num
				cur_patience = 0
			if FLAGS.evaluate == '1':
				print(
						'[Evaluate on %s - %s] | mAP: %.4f | Top-1: %.4f | Top-5: %.4f | Top-10: %.4f' % (
					FLAGS.dataset, FLAGS.probe, mAP,
					top_1, top_5, top_10))
				exit()
			else:
				if FLAGS.probe_view != '' and FLAGS.gallery_view != '':
					print(
					'[DG] View: %s v %s | mAP: %.4f (%.4f) | Top-1: %.4f (%.4f) | Top-5: %.4f | Top-10: %.4f | % d + o: %d |' % (
						FLAGS.probe_view, FLAGS.gallery_view, mAP, max_acc_1,
						top_1, max_acc_2, top_5, top_10,
						best_cluster_info_2[0], best_cluster_info_2[1]))
				else:
					print(
							'[DG] %s - %s | mAP: %.4f (%.4f) | Top-1: %.4f (%.4f) | Top-5: %.4f | Top-10: %.4f | % d + o: %d |' % (
						FLAGS.dataset, FLAGS.probe, mAP, max_acc_1,
						top_1, max_acc_2, top_5, top_10,
						best_cluster_info_2[0], best_cluster_info_2[1]))
			if cur_patience == patience:
				break
			rerank_dist = compute_jaccard_distance(train_features_all, k1=k1, k2=k2)
			if dataset == 'IAS' or dataset == 'KS20':
				cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
			else:
				cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
			pseudo_labels = cluster.fit_predict(rerank_dist)
			# discard outliers
			train_features_all = train_features_all[np.where(pseudo_labels != -1)]

			X_train_P_new = X_train_P[np.where(pseudo_labels != -1)]
			X_train_B_new = X_train_B[np.where(pseudo_labels != -1)]
			X_train_H_B_new = X_train_H_B[np.where(pseudo_labels != -1)]
			outlier_num = np.sum(pseudo_labels == -1)
			pseudo_labels = pseudo_labels[np.where(pseudo_labels != -1)]
			# print(pseudo_labels)
			num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)


			def generate_cluster_features(labels, features):
				centers = collections.defaultdict(list)
				for i, label in enumerate(labels):
					if label == -1:
						continue
					centers[labels[i]].append(features[i])

				centers = [
					torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
				]
				# print(centers)

				centers = torch.stack(centers, dim=0)
				return centers


			cluster_features = generate_cluster_features(pseudo_labels, train_features_all)
			cluster_features = cluster_features.numpy()
			cluster_features = cluster_features.astype(np.float64)

			tr_step = 0
			tr_size = X_train_P_new.shape[0]
			# pro_en_P = []
			# pro_en_B = []
			while tr_step * batch_size < tr_size:
				if (tr_step + 1) * batch_size > tr_size:
					break
				X_input_P = X_train_P_new[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_P = X_input_P.reshape([-1, 10, 3])
				X_input_B = X_train_B_new[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_B = X_input_B.reshape([-1, 5, 3])
				X_input_H_B = X_train_H_B_new[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_H_B = X_input_H_B.reshape([-1, 3, 3])
				labels = pseudo_labels[tr_step * batch_size:(tr_step + 1) * batch_size]
				_, loss, P_en, B_en, all_features = sess.run(
					[cluster_train_op, contrastive_loss, P_encode, B_encode, all_ftr],
					feed_dict={
						P_in: X_input_P,
						B_in: X_input_B,
						H_B_in: X_input_H_B,
						P_bias_in: biases_P,
						B_bias_in: biases_B,
						H_B_bias_in: biases_H_B,
						is_train: True,
						attn_drop: 0.0, ffd_drop: 0.0,
						pseudo_lab: labels,
						cluster_ftr: cluster_features})
				if tr_step % display == 0:
					print('[%s] Batch num: %d | Cluser num: %d | Outlier: %d | Loss: %.5f |' %
					      (str(epoch), tr_step, num_cluster, outlier_num, loss))
				tr_step += 1
		sess.close()


print('----- Model hyperparams -----')
# print('skeleton_nodes: ' + str(nb_nodes))
print('seqence_length: ' + str(time_step))
print('fusion_lambda: ' + str(fusion_lambda))
print('batch_size: ' + str(batch_size))
print('lr: ' + str(FLAGS.lr))
print('temperature: ' + FLAGS.t)
print('eps: ' + FLAGS.eps)
print('min_samples: ' + FLAGS.min_samples)
print('m: ' + FLAGS.m)
print('fusion_lambda: ' + FLAGS.fusion_lambda)
print('patience: ' + FLAGS.patience)

print('Mode: ' + FLAGS.mode)

if FLAGS.mode == 'DG':
	print('----- Mode Information  -----')
	print('Source Dataset: ' + FLAGS.S_dataset)
	print('Target Dataset: ' + FLAGS.dataset)
	print('Target Probe: ' + FLAGS.probe)
elif FLAGS.mode == 'UF':
	print('----- Dataset Information  -----')
	print('Dataset: ' + dataset)
	print('Probe: ' + FLAGS.probe)


