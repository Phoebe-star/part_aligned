#python news.py --data_dir=data --batch_size=1 --mode=cmc
#python news.py --mode=test --image1=data/labeled/val/0046_00.jpg --image2=data/labeled/val/0049_07.jpg
import tensorflow as tf
import numpy as np
import cv2

#import cuhk03_dataset_label2
import big_dataset_label as cuhk03_dataset_label2


import random
import cmc



from triplet_loss import batch_hard_triplet_loss





from importlib import import_module
from tensorflow.contrib import slim
from nets import NET_CHOICES
from heads import HEAD_CHOICES



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt  
from PIL import Image 

print tf.__version__
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', '80', 'batch size for training')
tf.flags.DEFINE_integer('max_steps', '210000', 'max steps for training')
tf.flags.DEFINE_string('logs_dir', 'logs_RES/', 'path to logs directory')
tf.flags.DEFINE_string('data_dir', 'data/', 'path to dataset')
tf.flags.DEFINE_float('learning_rate', '0.01', '')
tf.flags.DEFINE_string('mode', 'train', 'Mode train, val, test')
tf.flags.DEFINE_string('image1', '', 'First image path to compare')
tf.flags.DEFINE_string('image2', '', 'Second image path to compare')

tf.flags.DEFINE_float('global_rate', '1.0', 'global rate')
tf.flags.DEFINE_float('local_rate', '1.0', 'local rate')
tf.flags.DEFINE_float('softmax_rate', '1.0', 'softmax rate')

tf.flags.DEFINE_integer('ID_num', '20', 'id number')
tf.flags.DEFINE_integer('IMG_PER_ID', '4', 'img per id')



tf.flags.DEFINE_integer('embedding_dim', '128', 'Dimensionality of the embedding space.')
tf.flags.DEFINE_string('initial_checkpoint', 'resnet_v1_50.ckpt', 'Path to the checkpoint file of the pretrained network.')





IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224





def triplet_hard_loss(y_pred,id_num,img_per_id):
    with tf.variable_scope('hard_triplet'):

        SN = img_per_id  #img per id
        PN =id_num   #id num
        feat_num = SN*PN # images num
        
        y_pred = tf.nn.l2_normalize(y_pred,dim=1) 
    
        feat1 = tf.tile(tf.expand_dims(y_pred,0),[feat_num,1,1])
        feat2 = tf.tile(tf.expand_dims(y_pred,1),[1,feat_num,1])
        
        delta = tf.subtract(feat1,feat2)
        dis_mat = tf.reduce_sum(tf.square(delta), 2)+ 1e-8

        dis_mat = tf.sqrt(dis_mat)
     
        #dis_mat = tf.reduce_sum(tf.square(tf.subtract(feat1, feat2)), 2)
        #dis_mat = tf.sqrt(dis_mat)
        

    
        positive = dis_mat[0:SN,0:SN]
        negetive = dis_mat[0:SN,SN:]
        
        for i in range(1,PN):
            positive = tf.concat([positive,dis_mat[i*SN:(i+1)*SN,i*SN:(i+1)*SN]],axis = 0)
            if i != PN-1:
                negs = tf.concat([dis_mat[i*SN:(i+1)*SN,0:i*SN],dis_mat[i*SN:(i+1)*SN, (i+1)*SN:]],axis = 1)
            else:
                negs = tf.concat(dis_mat[i*SN:(i+1)*SN, 0:i*SN],axis = 0)
            negetive = tf.concat([negetive,negs],axis = 0)
  
        p=positive
        n=negetive
        positive = tf.reduce_max(positive,1)
        negetive = tf.reduce_min(negetive,axis=1) #acc
        
        #negetive = tf.reduce_mean(negetive,1)
        #negetive = tf.reduce_max(negetive,axis=1) #false

        a1 = 0.3
        
        basic_loss = tf.add(tf.subtract(positive,negetive), a1)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)
        #loss = tf.reduce_mean(tf.maximum(0.0,positive-negetive+a1))
       
        return loss ,tf.reduce_mean(positive) ,tf.reduce_mean(negetive)
        

        






def part_attend(images1, weight_decay):
    with tf.variable_scope('part_attend'):
        
        '''
        input_filter = 512
        global_pool = 14
        dim_split = 128
        '''
        input_filter = 2048
        global_pool = 7
        dim_split = 512
        
        
        
        conv1_branch1 = tf.layers.conv2d(images1, 1, [1, 1], padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), reuse=None, name='conv1_branch1')        
        h1_sigmoid1 = tf.nn.sigmoid(conv1_branch1)
        feat1_tile1 = tf.tile(h1_sigmoid1,[1,1,1,input_filter])
        h1_prod1 = tf.multiply(images1,feat1_tile1)
        feat1_avg_pool1 = tf.nn.avg_pool(h1_prod1, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        reshape_branch1 = tf.reshape(feat1_avg_pool1, [FLAGS.batch_size, -1])
        fc1_branch1 = tf.layers.dense(reshape_branch1, dim_split, reuse=None, name='fc1_branch1')
        
        conv1_branch2 = tf.layers.conv2d(images1, 1, [1, 1], padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), reuse=None, name='conv1_branch2')        
        h1_sigmoid2 = tf.nn.sigmoid(conv1_branch2)
        feat1_tile2 = tf.tile(h1_sigmoid2,[1,1,1,input_filter])
        h1_prod2 = tf.multiply(images1,feat1_tile2)
        feat1_avg_pool2 = tf.nn.avg_pool(h1_prod2, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        reshape_branch2 = tf.reshape(feat1_avg_pool2, [FLAGS.batch_size, -1])
        fc1_branch2 = tf.layers.dense(reshape_branch2, dim_split,reuse=None, name='fc1_branch2')
                    
        
        conv1_branch3 = tf.layers.conv2d(images1, 1, [1, 1], padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), reuse=None, name='conv1_branch3')        
        h1_sigmoid3 = tf.nn.sigmoid(conv1_branch3)
        feat1_tile3 = tf.tile(h1_sigmoid3,[1,1,1,input_filter])
        h1_prod3 = tf.multiply(images1,feat1_tile3)
        feat1_avg_pool3 = tf.nn.avg_pool(h1_prod3, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        reshape_branch3 = tf.reshape(feat1_avg_pool3, [FLAGS.batch_size, -1])
        fc1_branch3 = tf.layers.dense(reshape_branch3, dim_split,reuse=None, name='fc1_branch3')
        
        
        conv1_branch4 = tf.layers.conv2d(images1, 1, [1, 1], padding='same',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), reuse=None, name='conv1_branch4')        
        h1_sigmoid4 = tf.nn.sigmoid(conv1_branch4)
        feat1_tile4 = tf.tile(h1_sigmoid4,[1,1,1,input_filter])
        h1_prod4 = tf.multiply(images1,feat1_tile4)
        feat1_avg_pool4 = tf.nn.avg_pool(h1_prod4, ksize=[1, global_pool, global_pool, 1], strides=[1, 1, 1, 1], padding='VALID')
        reshape_branch4 = tf.reshape(feat1_avg_pool4, [FLAGS.batch_size, -1])
        fc1_branch4 = tf.layers.dense(reshape_branch4, dim_split,reuse=None, name='fc1_branch4')
        
     
        
        
        concat1 = tf.concat([fc1_branch1, fc1_branch2,fc1_branch3,fc1_branch4], axis=1)
        concat1_L2 = tf.nn.l2_normalize(concat1,dim=1)

        

        return concat1_L2




 


def main(argv=None):
  
    if FLAGS.mode == 'test':
        FLAGS.batch_size = 1
    
    if FLAGS.mode == 'cmc':
        FLAGS.batch_size = 1

    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    images = tf.placeholder(tf.float32, [3, FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images')
    
    images_total = tf.placeholder(tf.float32, [FLAGS.batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name='images_total')
    
    labels = tf.placeholder(tf.float32, [FLAGS.batch_size], name='labels')
  
    
    
    

    
    
    
    is_train = tf.placeholder(tf.bool, name='is_train')
    global_step = tf.Variable(0, name='global_step', trainable=False)
    weight_decay = 0.0005
    tarin_num_id = 0
    val_num_id = 0

    if FLAGS.mode == 'train':
        tarin_num_id = cuhk03_dataset_label2.get_num_id(FLAGS.data_dir, 'train')
        print(tarin_num_id, '               11111111111111111111               1111111111111111')
    elif FLAGS.mode == 'val':
        val_num_id = cuhk03_dataset_label2.get_num_id(FLAGS.data_dir, 'val')
  
    
    
    
    
    
    
    # Create the model and an embedding head.
    model = import_module('nets.' + 'resnet_v1_50')
    head = import_module('heads.' + 'fc1024')
    
    
    # Feed the image through the model. The returned `body_prefix` will be used
    # further down to load the pre-trained weights for all variables with this
    # prefix.
    endpoints, body_prefix = model.endpoints(images_total, is_training=True)

    with tf.name_scope('head'):
        endpoints = head.head(endpoints, FLAGS.embedding_dim, is_training=True)
    
    
    '''
    print endpoints['model_output'] # (bt,2048)
    print endpoints['global_pool'] # (bt,2048)
    print endpoints['resnet_v1_50/block4']# (bt,7,7,2048)
    
    print ' 1\n'
    '''

    
    
    
    
    train_mode = tf.placeholder(tf.bool)


    print('Build network')
    
    feat = endpoints['resnet_v1_50/block4']# (bt,7,7,2048)
    
    

    #feat = tf.convert_to_tensor(feat, dtype=tf.float32)

    feat_1x1 = tf.layers.conv2d(feat, 2048, [1, 1],padding='valid',
            kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay), reuse=None, name='conv1x1')     
    
    feature = part_attend(feat_1x1,weight_decay)
   
    #loss_triplet,PP,NN = triplet_hard_loss(feature,FLAGS.ID_num,FLAGS.IMG_PER_ID)
    loss_triplet ,PP,NN = batch_hard_triplet_loss(labels,feature,0.3)

    
    
   
    
    loss = loss_triplet*FLAGS.global_rate
    

    
    
    
    # These are collected here before we add the optimizer, because depending
    # on the optimizer, it might add extra slots, which are also global
    # variables, with the exact same prefix.
    model_variables = tf.get_collection(
    tf.GraphKeys.GLOBAL_VARIABLES, body_prefix)
    
    
    
    
    #optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

    #optimizer = tf.train.AdadeltaOptimizer(learning_rate)
    #train = optimizer.minimize(loss, global_step=global_step)
    
    
    
    
    # Update_ops are used to update batchnorm stats.
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        #train_op = optimizer.minimize(loss_mean, global_step=global_step)

    
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        train = optimizer.minimize(loss, global_step=global_step)
    

    lr = FLAGS.learning_rate

    #config=tf.ConfigProto(log_device_placement=True)
    #config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)) 
    # GPU
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    
    with tf.Session(config=config) as sess:
        
        

        
        
        print '\n'
        #print model_variables
        print '\n'
        #sess.run(tf.global_variables_initializer())
        #saver = tf.train.Saver()
        
        #checkpoint_saver = tf.train.Saver(max_to_keep=0)
        checkpoint_saver = tf.train.Saver()


        ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restore model')
            print ckpt.model_checkpoint_path
            #saver.restore(sess, ckpt.model_checkpoint_path)
            checkpoint_saver.restore(sess, ckpt.model_checkpoint_path)
                    
        #for first , training load imagenet
        else:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(model_variables)
            print FLAGS.initial_checkpoint
            saver.restore(sess, FLAGS.initial_checkpoint)
            
         
            

            
            
        if FLAGS.mode == 'train':
            step = sess.run(global_step)
            for i in xrange(step, FLAGS.max_steps + 1):

                batch_images, batch_labels, batch_images_total = cuhk03_dataset_label2.read_data(FLAGS.data_dir, 'train', tarin_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size,FLAGS.ID_num,FLAGS.IMG_PER_ID)
                
             
              
                feed_dict = {learning_rate: lr,  is_train: True , train_mode: True, images_total: batch_images_total, labels: batch_labels}
                
                
                
                
                _,train_loss = sess.run([train,loss], feed_dict=feed_dict) 
                    
                print('Step: %d, Learning rate: %f, Train loss: %f ' % (i, lr, train_loss))
                
                gtoloss,gp,gn = sess.run([loss_triplet,PP,NN], feed_dict=feed_dict)   
                print 'global hard: ',gtoloss
                print 'global P: ',gp
                print 'global N: ',gn
                
                
            
                
                
                #lr = FLAGS.learning_rate / ((2) ** (i/160000)) * 0.1
                lr = FLAGS.learning_rate * ((0.0001 * i + 1) ** -0.75)
                if i % 100 == 0:
                    #saver.save(sess, FLAGS.logs_dir + 'model.ckpt', i)
                    # test save
                    #vgg.save_npy(sess, './big.npy')
                    
                    checkpoint_saver.save(sess,FLAGS.logs_dir + 'model.ckpt', i)
                
                
                
                
        

        elif FLAGS.mode == 'val':
            total = 0.
            for _ in xrange(10):
                batch_images, batch_labels = cuhk03_dataset_label2.read_data(FLAGS.data_dir, 'val', val_num_id,
                    IMAGE_WIDTH, IMAGE_HEIGHT, FLAGS.batch_size)
                feed_dict = {images: batch_images, labels: batch_labels, is_train: False}
                prediction = sess.run(inference, feed_dict=feed_dict)
                prediction = np.argmax(prediction, axis=1)
                label = np.argmax(batch_labels, axis=1)

                for i in xrange(len(prediction)):
                    if prediction[i] == label[i]:
                        total += 1
            print('Accuracy: %f' % (total / (FLAGS.batch_size * 10)))

            '''
            for i in xrange(len(prediction)):
                print('Prediction: %s, Label: %s' % (prediction[i] == 0, labels[i] == 0))
                image1 = cv2.cvtColor(batch_images[0][i], cv2.COLOR_RGB2BGR)
                image2 = cv2.cvtColor(batch_images[1][i], cv2.COLOR_RGB2BGR)
                image = np.concatenate((image1, image2), axis=1)
                cv2.imshow('image', image)
                key = cv2.waitKey(0)
                if key == 1048603:  # ESC key
                    break
            '''

        
        elif FLAGS.mode == 'test':
            image1 = cv2.imread(FLAGS.image1)
            image1 = cv2.resize(image1, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            image1 = np.reshape(image1, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            image2 = cv2.imread(FLAGS.image2)
            image2 = cv2.resize(image2, (IMAGE_WIDTH, IMAGE_HEIGHT))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
            image2 = np.reshape(image2, (1, IMAGE_HEIGHT, IMAGE_WIDTH, 3)).astype(float)
            test_images = np.array([image1, image2,image2])

            feed_dict = {images: test_images, is_train: False, droup_is_training: False}
            #prediction, prediction2 = sess.run([DD,DD2], feed_dict=feed_dict)
            prediction = sess.run([inference], feed_dict=feed_dict)
            prediction = np.array(prediction)
            print prediction.shape
            print( np.argmax(prediction[0])+1)
            
           
        
            #print(bool(not np.argmax(prediction[0])))

if __name__ == '__main__':
    tf.app.run()