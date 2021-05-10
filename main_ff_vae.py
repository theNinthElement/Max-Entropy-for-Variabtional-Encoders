#import pandas as pd
import numpy as np
#import librosa
#from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
#from mixup_generator import MixupGenerator
#from random_eraser import get_random_eraser
import matplotlib.pyplot as plt
import os as os
#from keras.engine.topology import Layer
from keras import backend as K
#import tensorflow as tf
import scipy.io as scio

DATASET=71
CONV = True
l2_reg = .01


########################################################################################################################
# Data-dependent parameters
########################################################################################################################
if DATASET==5:
   N1=45
   N2=20
   mat=scio.loadmat('tree_cnnannotft2_red.mat')
   width_shift_range=7
   height_shift_range=3
elif DATASET==45:
   N1=45
   N2=20
   mat=scio.loadmat('three_cnnannotft2_red.mat')
   width_shift_range=7
   height_shift_range=3
elif DATASET==21:
   N1=14
   N2=14
   mat=scio.loadmat('mnist_3rg1.mat')
   width_shift_range=0
   height_shift_range=0   
elif DATASET==71:
    N1=28
    N2=28
    mat=scio.loadmat('ds717273.mat')
    width_shift_range=0
    height_shift_range=0   
elif DATASET==72:
    N1=28
    N2=28
    mat=scio.loadmat('ds717273.mat')
    width_shift_range=0
    height_shift_range=0
elif DATASET==73:
    N1=28
    N2=28
    mat=scio.loadmat('ds717273.mat')
    width_shift_range=0
    height_shift_range=0
elif DATASET==81: # gaussian distributed character 3
   N1=28
   N2=28
   mat=scio.loadmat('mnist_3g1.mat')
   width_shift_range=0
   height_shift_range=0
elif DATASET==82: # character 8 
   N1=28
   N2=28
   mat=scio.loadmat('mnist_3g2.mat')
   width_shift_range=0
   height_shift_range=0
elif DATASET==83: # character 9 
   N1=28
   N2=28
   mat=scio.loadmat('mnist_3g3.mat')
   width_shift_range=0
   height_shift_range=0

nceps=N2
feat_length=N1
latent_dim = 4

def cosine_similarity_loss(y_true, y_pred):
    x = K.l2_normalize(y_true, axis=-1)
    y = K.l2_normalize(y_pred, axis=-1)
    return -K.mean(x*y, axis=-1, keepdims=True)


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def vae_def_goog(feat_length, nceps, latent_dim=32):
    """
    Definition of the 2D DCAE used for mel spectrograms.
    :param feat_length: time dimension
    :param nceps: number of mel filters.
    :return: CNN model
    """


    input = keras.layers.Input(shape=(nceps, feat_length, 1), dtype='float32')
    if CONV:
        # DIM=N1*N2 
        x = keras.layers.Reshape((N1, N2, 1), input_shape=(N1*N2, ))(input)  
        M1 = 32
        k1=(17,12)
        x = keras.layers.Conv2D(M1, kernel_size=k1, padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg),\
               bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        # DIM=N1xN2xM1 
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        P1 = (3,2)
        x = keras.layers.AveragePooling2D(pool_size=P1)(x)
        # DIM=N1//P1[0]*N2//P1[1]*M1 
        M2=32
        k2 = (5,5)
        x = keras.layers.Conv2D(M2, kernel_size=k2, padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg),\
               bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        # DIM=N1//P1[0]*N2//P1[1]*M2 
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(latent_dim, kernel_regularizer=keras.regularizers.l2(l2_reg), \
            bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
    else:
        # bottleneck of 32 neurons
        x = keras.layers.Flatten()(input)
#        x = keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
#        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Dense(48, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
#        x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
#        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Dense(latent_dim, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

    # variational layers
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = keras.Model(input, [z_mean, z_log_var, z], name='encoder')
    print(encoder.summary())
    # build decoder model
    latent_input = keras.layers.Input(shape=(latent_dim,), name='z_sampling')
    if CONV:
        x = keras.layers.Dense((N1//P1[0])*(N2//P1[1])*M2, kernel_regularizer=keras.regularizers.l2(l2_reg), \
              bias_regularizer=keras.regularizers.l2(l2_reg))(latent_input) 
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Reshape((N1//P1[0], N2//P1[1], M2), input_shape=(N1//P1[0]*N2//P1[1]*M2, ))(x)
        x = keras.layers.Conv2D(M1, kernel_size=k2, padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg), \
                  bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x) 
        x = keras.layers.Reshape((N1//P1[0], N2//P1[1], M1))(x)
        x = keras.layers.UpSampling2D(size=P1)(x)
        x = keras.layers.Reshape((N1, N2, M1))(x)
        # last one has to have 1 kernel
        x = keras.layers.Conv2D(1, kernel_size=k1, padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg), \
                  bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

    else:
        x=latent_input
#        x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
#        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Dense(48, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
#        x = keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
#        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Dense(nceps*feat_length, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

    model_output = keras.layers.Reshape((nceps,feat_length,1), input_shape=(nceps*feat_length, ))(x)

    # instantiate decoder model
    decoder = keras.Model(latent_input, model_output, name='decoder')
    print(decoder.summary())
    outputs = decoder(encoder(input)[2])
    return input, outputs, z_mean, z_log_var

def vae_def_mnist(feat_length, nceps, latent_dim=32):

    input = keras.layers.Input(shape=(nceps, feat_length, 1), dtype='float32')
    l2_reg = .1
    if CONV: 
        # INPUT DIM = 784
        M1=32
        x = keras.layers.Reshape((N1, N2, 1), input_shape=(N1*N2, ))(input)  
        k1=(16,16)
        x = keras.layers.Conv2D(M1, kernel_size=k1, padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg),\
               bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        DIM=N1*N2*M1  
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        P1 = 2
        x = keras.layers.AveragePooling2D(pool_size=(P1, P1))(x)
        DIM=N1*N2*M1//P1//P1
        M2=32
        x = keras.layers.Conv2D(M2, kernel_size=3, padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg),\
               bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        DIM=N1*N2//P1//P1*M2 
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(latent_dim, kernel_regularizer=keras.regularizers.l2(l2_reg), \
            bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
    else:
        # bottleneck of 32 neurons
        x = keras.layers.Flatten()(input)
        x = keras.layers.Dense(48, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Dense(8, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Dense(latent_dim, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)

    # variational layers
    z_mean = keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = keras.layers.Dense(latent_dim, name='z_log_var')(x)

    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    # instantiate encoder model
    encoder = keras.Model(input, [z_mean, z_log_var, z], name='encoder')
    print(encoder.summary())
    # build decoder model
    latent_input = keras.layers.Input(shape=(latent_dim,), name='z_sampling')
    if CONV:
        DIM=(N1//P1)*(N2//P1)*M2
        x = keras.layers.Dense(DIM, kernel_regularizer=keras.regularizers.l2(l2_reg), \
              bias_regularizer=keras.regularizers.l2(l2_reg))(latent_input) 
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Reshape((N1//P1, N2//P1, M2), input_shape=(DIM, ))(x)
        x = keras.layers.Conv2D(M1, kernel_size=3, padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg), \
                  bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.UpSampling2D(size=(P1, P1))(x)
        # last one has to have 1 kernel
        x = keras.layers.Conv2D(1, kernel_size=3, padding='same', kernel_regularizer=keras.regularizers.l2(l2_reg), \
                  bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        #x = keras.layers.LeakyReLU(alpha=0.1)(x) # sigmoid model 
        #x= keras.layers.Softmax()(x)
        x = keras.activations.sigmoid(x)
    else:
        x = keras.layers.Dense(8, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(latent_input)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Dense(32, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Dense(48, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        x = keras.layers.LeakyReLU(alpha=0.1)(x)
        x = keras.layers.Dense(nceps*feat_length, kernel_regularizer=keras.regularizers.l2(l2_reg), bias_regularizer=keras.regularizers.l2(l2_reg))(x)
        #x = keras.layers.LeakyReLU(alpha=0.1)(x) # use a sigmoid here maybe 
        #x= keras.layers.Softmax()(x)
        x = keras.activations.sigmoid(x)

    model_output = keras.layers.Reshape((nceps,feat_length,1), input_shape=(nceps*feat_length, ))(x)

    # instantiate decoder model
    decoder = keras.Model(latent_input, model_output, name='decoder')
    print(decoder.summary())
    outputs = decoder(encoder(input)[2])
    return input, outputs, z_mean, z_log_var


########################################################################################################################
# Load data and reshape
########################################################################################################################
X = mat['X']
X2 = mat['X2']
Xv = mat['Xv']
#tch = mat['tch']
#tch2 = mat['tch2']
#tchv = mat['tchv']
X = np.expand_dims(np.fliplr(X.reshape((N1, N2, X.shape[1])).transpose()), axis=-1)
X2 = np.expand_dims(np.fliplr(X2.reshape((N1, N2, X2.shape[1])).transpose()), axis=-1)
Xv = np.expand_dims(np.fliplr(Xv.reshape((N1, N2, Xv.shape[1])).transpose()), axis=-1)
#tch = tch.transpose()
#tch2 = tch2.transpose()
#tchv = tchv.transpose()

########################################################################################################################
# Train autoencoder
########################################################################################################################

#num_classes = tch.shape[1]
batch_size = 32
batch_size_test = 32
epochs = 50
aeons = 10
alpha = 1

train_data = np.copy(X)#np.copy(X[tch[:, j]])
val_data = np.copy(Xv)#np.copy(Xv[tchv[:, j]])
tst_data = np.copy(X2)#np.copy(X2[tchv[:, j]])

# create data generator for mixup and random erasing for every batch
datagen = keras.preprocessing.image.ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True,
    width_shift_range=width_shift_range,
    height_shift_range=height_shift_range,
    #preprocessing_function=get_random_eraser(v_l=np.min(train_data), v_h=np.max(train_data))
)
test_datagen = keras.preprocessing.image.ImageDataGenerator(
    #featurewise_center=True,
    #featurewise_std_normalization=True
)
datagen.fit(np.r_[train_data])
test_datagen.fit(np.r_[train_data])
#training_generator = MixupGenerator(train_data, tch, batch_size=batch_size, alpha=alpha, datagen=datagen)

# compile model
if DATASET==81 or DATASET==82 or DATASET==83 :
   input, model_output, z_mean, z_log_var = vae_def_mnist(feat_length=N1, nceps=N2, latent_dim = latent_dim)
elif DATASET==4 or DATASET==5:
    input, model_output, z_mean, z_log_var = vae_def_goog(feat_length=N1, nceps=N2, latent_dim = latent_dim)
elif DATASET==71 or DATASET==72 or DATASET==73 :
    input, model_output, z_mean, z_log_var = vae_def_mnist(feat_length=N1, nceps=N2, latent_dim = latent_dim)

vae = keras.Model(inputs=[input], outputs=[model_output])
# VAE loss
reconstruction_loss = keras.losses.mean_squared_error(input, model_output)
reconstruction_loss *= N1*N2
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5 # multiply that by zero and get auto encoder
vae_loss = K.mean(reconstruction_loss) + K.mean(kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer=keras.optimizers.Adam(decay=0.0001))
print(vae.summary())

# fit model
for k in np.arange(aeons):
    weight_path = 'wts_ff_vae_' + str(k+1) + 'k.h5'
    if not os.path.isfile(weight_path):
        #vae.fit_generator(training_generator(), verbose=2, steps_per_epoch=len(train_data)/batch_size, epochs=epochs)
        vae.fit(train_data, batch_size=batch_size, epochs=epochs, verbose=0)
        vae.save(weight_path)
    else:
        vae = keras.models.load_model(weight_path)

    # compute loss on train data
    #print('loss for training data:')
    #loss_trn = vae.evaluate_generator(test_datagen.flow(train_data, train_data, batch_size=batch_size_test), steps=len(train_data)/batch_size_test)
    #print(loss_trn)
    #loss_trn = vae.evaluate(train_data, train_data, verbose=0)
    #print(loss_trn)
    #trn_recon = vae.predict_generator(test_datagen.flow(train_data, batch_size=batch_size_test), steps=len(train_data)/batch_size_test)
    trn_recon = vae.predict(train_data, batch_size=batch_size_test)
    loss_trn = ((train_data-trn_recon)**2).mean(axis=None)

    # compute loss on validation data
    #print('loss for validation data:')
    #loss_val = vae.evaluate(val_data, val_data, verbose=0)
    #print(loss_val)
    #val_recon = vae.predict_generator(test_datagen.flow(val_data, batch_size=batch_size_test), steps=len(val_data)/batch_size_test)
    val_recon = vae.predict(val_data, batch_size=batch_size_test)
    loss_val = ((val_data-val_recon)**2).mean(axis=None)

    print("aeon %d," % k, "train loss=%f" % loss_trn,  "val loss=%f" % loss_val)

    # compute loss on test data
#    print('loss for test data:')
#    #loss_tst = vae.evaluate(tst_data, tst_data, verbose=0)
#    #print(loss_tst)
#    #tst_recon = vae.predict_generator(test_datagen.flow(tst_data, batch_size=batch_size_test), steps=len(tst_data)/batch_size_test)
#    tst_recon = vae.predict(tst_data, batch_size=batch_size_test)
#    loss_tst = ((tst_data-tst_recon)**2).mean(axis=None)
#    print(loss_tst)
#    print('#####')
