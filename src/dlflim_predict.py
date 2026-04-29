# Relevant libraries and functions
from __future__ import print_function
import numpy as np
import argparse
import os, time, re
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization, Input, Conv2D, add, Conv3D, Reshape
from pathlib import Path
import tifffile as tff
from scipy.signal import savgol_filter
from skimage.filters import gaussian, median
from skimage.morphology import disk
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")

def check_gpu():

    gpus = tf.config.list_physical_devices('GPU')
    print(f'Num GPUs Available: {len(gpus)}')
    if gpus:
        print('TensorFlow GPU check PASSED')

        return True
    else:
        print('TensorFlow GPU test FAILED')

        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Predict Lifetimes with DLFLIM')
    parser.add_argument('--inpath', type=Path, help='path to the parent data folder')
    parser.add_argument('--inpath_irf', type=Path, help='path to the irf data folder')
    parser.add_argument('--inpath_model', type=Path, help='path to the model folder')
    parser.add_argument('--gate_size', nargs='?',default=0.6998,type=float, help='flim gate step')
    parser.add_argument('--nz', nargs='?',default=1,type=int, help='number of z-slices in one volume')
    parser.add_argument('--nt', nargs='?',default=30,type=int, help='number of flim planes in one flim stack')
    parser.add_argument('--version', nargs='?',default=1,type=int, help='single or double decay')
    parser.add_argument('--start_t', nargs='?',default=1,type=int, help='starting time point for processing')
    parser.add_argument('--end_t', nargs='?',default=-1,type=int, help='ending time point for processing')
    parser.add_argument('--channel', nargs='?',default=0, type=int, choices=[0,1], help='channel to process')
    args = parser.parse_args()
    return args

def cubify(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    # newshape must divide oldshape evenly or else ValueError will be raised
    return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

def uncubify(arr, oldshape):
    N, newshape = arr.shape[0], arr.shape[1:]
    oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
    return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

def tfcubify(arr, newshape):
    oldshape = np.array(arr.shape)
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.column_stack([repeats, newshape]).ravel()
    order = np.arange(len(tmpshape))
    order = np.concatenate([order[::2], order[1::2]])
    return tf.reshape(tf.transpose(tf.reshape(arr,tmpshape),order),np.hstack([-1,np.array(newshape)]))

def tfuncubify(arr, oldshape):
    N, newshape = tf.shape(arr).numpy()[0], tf.shape(arr).numpy()[1:]
    # oldshape = np.array(oldshape)    
    repeats = (oldshape / newshape).astype(int)
    tmpshape = np.concatenate([repeats, newshape])
    order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')

    return tf.reshape(tf.transpose(tf.reshape(arr,tmpshape),order),oldshape)

def input_flim_vol(file_path,nt,nz,sz_x=512,sz_y=512):
    A = np.fromfile(file_path,dtype='int16',sep="")
    temp = np.zeros((nt,nz,sz_x,sz_y))
    for j in range(nz):
        for n in range(nt):
            temp[n,j,:,:] = np.reshape(A[(j*nt+n)*(512*512):(j*nt+n+1)*(512*512)],[512,512])
    
    return temp

def resblock_2D(num_filters, size_filter, x):
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = Activation('relu')(Fx)
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(Fx)
    output = add([Fx, x])
    output = Activation('relu')(output)
    return output

def resblock_2D_BN(num_filters, size_filter, x):
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = BatchNormalization()(Fx)
    Fx = Activation('relu')(Fx)
    Fx = Conv2D(num_filters, size_filter, padding='same', activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    output = add([Fx, x])
    #output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output

def resblock_3D_BN(num_filters, size_filter, x):
    Fx = Conv3D(num_filters, size_filter, padding='same', activation=None)(x)
    Fx = BatchNormalization()(Fx)
    Fx = Activation('relu')(Fx)
    Fx = Conv3D(num_filters, size_filter, padding='same', activation=None)(Fx)
    Fx = BatchNormalization()(Fx)
    output = add([Fx, x])
    #output = BatchNormalization()(output)
    output = Activation('relu')(output)
    return output

def load_model(version,model_dir,xX=32,yY=32,nTG=100):

    model = None

    if version == 1:
        
        t_data = Input(shape=(xX, yY, nTG,2))
        tpsf = t_data

        # tpsf = Masking(mask_value=0.0)(tpsf)
        tpsf = Conv3D(50,kernel_size=(1,1,10),strides=(1,1,5), padding='same', activation=None, data_format="channels_last")(tpsf)
        tpsf = BatchNormalization()(tpsf)
        tpsf = Activation('relu')(tpsf)
        tpsf = resblock_3D_BN(50, (1,1,5), tpsf)
        tpsf = resblock_3D_BN(50, (1,1,5), tpsf)
        sz = tpsf.shape
        # print(sz)
        tpsf = Reshape((xX,yY,sz[3]*sz[4]))(tpsf)
        # tpsf = Reshape((xX,yY,1600))(tpsf)
        tpsf = Conv2D(256, 1, padding='same', activation=None, data_format="channels_last")(tpsf)
        tpsf = BatchNormalization()(tpsf)
        tpsf = Activation('relu')(tpsf)
        tpsf = Conv2D(256, 1, padding='same', activation=None, data_format="channels_last")(tpsf)
        tpsf = BatchNormalization()(tpsf)
        tpsf = Activation('relu')(tpsf)
        tpsf = resblock_2D_BN(256, 1, tpsf)
        tpsf = resblock_2D_BN(256, 1, tpsf)

        # Long-lifetime branch
        imgT2 = Conv2D(64, 1, padding='same', activation=None)(tpsf)
        imgT2 = BatchNormalization()(imgT2)
        imgT2 = Activation('relu')(imgT2)
        imgT2 = Conv2D(32, 1, padding='same', activation=None)(imgT2)
        imgT2 = BatchNormalization()(imgT2)
        imgT2 = Activation('relu')(imgT2)
        imgT2 = Conv2D(1, 1, padding='same', activation=None)(imgT2)
        imgT2 = Activation('relu')(imgT2)

        adamprop = Adam(learning_rate=1e-4)

        modelD = Model(inputs=[t_data], outputs={'imgT2':imgT2})
        modelD.compile(loss='mse',optimizer=adamprop,metrics={'imgT2':'mae'})

        fN = 'single_decay_hafldecay_real'
        print('Loading model... {}'.format(os.path.join(model_dir,fN+'.keras')))

        modelD.load_weights(os.path.join(model_dir,fN+'.keras'))

        return modelD
    
    elif version == 2:

        t_data = Input(shape=(xX, yY, nTG, 2))
        tpsf = t_data

        # tpsf = Masking(mask_value=0.0)(tpsf)
        tpsf = Conv3D(50,kernel_size=(1,1,10),strides=(1,1,5), padding='same', activation=None, data_format="channels_last")(tpsf)
        tpsf = BatchNormalization()(tpsf)
        tpsf = Activation('relu')(tpsf)
        tpsf = resblock_3D_BN(50, (1,1,5), tpsf)
        tpsf = resblock_3D_BN(50, (1,1,5), tpsf)
        sz = tpsf.shape
        # print(sz)
        tpsf = Reshape((xX,yY,sz[3]*sz[4]))(tpsf)
        # tpsf = Reshape((xX,yY,1600))(tpsf)
        tpsf = Conv2D(256, 1, padding='same', activation=None, data_format="channels_last")(tpsf)
        tpsf = BatchNormalization()(tpsf)
        tpsf = Activation('relu')(tpsf)
        tpsf = Conv2D(256, 1, padding='same', activation=None, data_format="channels_last")(tpsf)
        tpsf = BatchNormalization()(tpsf)
        tpsf = Activation('relu')(tpsf)
        tpsf = resblock_2D_BN(256, 1, tpsf)
        tpsf = resblock_2D_BN(256, 1, tpsf)

        # Short-lifetime branch
        imgT1 = Conv2D(64, 1, padding='same', activation=None)(tpsf)
        imgT1 = BatchNormalization()(imgT1)
        imgT1 = Activation('relu')(imgT1)
        imgT1 = Conv2D(32, 1, padding='same', activation=None)(imgT1)
        imgT1 = BatchNormalization()(imgT1)
        imgT1 = Activation('relu')(imgT1)
        imgT1 = Conv2D(1, 1, padding='same', activation=None)(imgT1)
        imgT1 = Activation('relu')(imgT1)

        # Long-lifetime branch
        imgT2 = Conv2D(64, 1, padding='same', activation=None)(tpsf)
        imgT2 = BatchNormalization()(imgT2)
        imgT2 = Activation('relu')(imgT2)
        imgT2 = Conv2D(32, 1, padding='same', activation=None)(imgT2)
        imgT2 = BatchNormalization()(imgT2)
        imgT2 = Activation('relu')(imgT2)
        imgT2 = Conv2D(1, 1, padding='same', activation=None)(imgT2)
        imgT2 = Activation('relu')(imgT2)

        # Amplitude-Ratio branch
        imgTR = Conv2D(64, 1, padding='same', activation=None)(tpsf)
        imgTR = BatchNormalization()(imgTR)
        imgTR = Activation('relu')(imgTR)
        imgTR = Conv2D(32, 1, padding='same', activation=None)(imgTR)
        imgTR = BatchNormalization()(imgTR)
        imgTR = Activation('relu')(imgTR)
        imgTR = Conv2D(1, 1, padding='same', activation=None)(imgTR)
        imgTR = Activation('relu')(imgTR)

        adamprop = Adam(learning_rate=1e-4)

        modelD = Model(inputs=[t_data], outputs={'imgT1':imgT1,'imgT2':imgT2, 'imgTR':imgTR})

        modelD.compile(loss={'imgT1':'mse','imgT2':'mse','imgTR':'mse'},
                    optimizer=adamprop,
                    metrics={'imgT1':'mae','imgT2':'mae','imgTR':'mae'})

        fN = 'double_decay_halfdecay_real'
        print('Loading model... {}'.format(os.path.join(model_dir,fN+'.keras')))

        modelD.load_weights(os.path.join(model_dir,fN+'.keras'))

        return modelD

    else:
        exit(f'error: version should be either 1 or 2')


if __name__ == '__main__':

    if check_gpu():

        args = parse_args()
        model_dir = args.inpath_model

        version = args.version
        if version == 1:
            nTG = 100
        elif version == 2:
            nTG = 200

        model = load_model(version,model_dir,nTG=nTG)

        step_size = args.gate_size
        nt = args.nt
        nz = args.nz

        orig_step_size = 0.6998
        orig_nt = 30

        nx = 512
        ny = 512
        xX = 32
        yY = 32

        dt_max = 0.2046
        dt_irf = 0.0186

        ddir = f'/nrs/aic/Owen/I2/FLIM_Data/ML_Models/irf/IRF_561nm_700steps_18p6psStepSize_20nsWidth_20251014_135341'

        print('Loading IRFs....')

        nT = 700
        ch1 = f'SPC00_TM00000_ANG000_CHN00_PH0.stack'
        irfs = input_flim_vol(os.path.join(ddir,ch1),nT,21)
        mean_irf = np.mean(irfs,axis=1)
        avg_irf = mean_irf[:,:int(mean_irf.shape[1]/2),:]
        irf_end = np.mean(avg_irf[-3:,...],axis=0)
        avg_irf = np.clip(avg_irf - irf_end,a_min=0.0,a_max=np.inf)

        avg_irf1 = savgol_filter(avg_irf,51,3,axis=0)
        id_start = 250
        grad = np.argmin(np.gradient(np.gradient(avg_irf1,axis=0),axis=0)[id_start:,...],axis=0)+id_start
        grad_test_old = np.median(grad.flatten())

        ddir = args.inpath_irf

        ch1 = f'SPC00_TM00000_ANG000_CHN00_PH0.stack'
        gates = 800
        irfs = input_flim_vol(os.path.join(ddir,ch1),gates,21)
        avg_irf = np.mean(irfs[:,1:,...],axis=1)

        avg_irf2 = savgol_filter(avg_irf,51,3,axis=0)
        id_start = 250
        grad_new = np.argmin(np.gradient(np.gradient(avg_irf2,axis=0),axis=0)[id_start:,...],axis=0)+id_start
        grad_test_new = np.median(grad_new.flatten())

        shift = np.round(grad_test_new - grad_test_old).astype(np.int64)

        irf_test = np.copy(avg_irf)
        irf_test1 = np.clip(irf_test,a_min=1e-6,a_max=None)
        irf_test = np.pad(irf_test,pad_width=((0,np.round((orig_step_size*orig_nt-dt_irf*avg_irf.shape[0])/dt_irf).astype(np.int64)),(0,0),(0,0)),mode='edge')
        avg_irf2 = savgol_filter(irf_test,51,3,axis=0)
        id_start = 250
        grad_irf_test = np.argmin(np.gradient(np.gradient(avg_irf2,axis=0),axis=0)[id_start:,...],axis=0)+id_start
        idx = np.round(median(grad_irf_test,disk(2)) - np.amin(median(grad_irf_test,disk(2)))).astype(np.int64)

        irf_test_reshaped = np.reshape(irf_test,(irf_test.shape[0],irf_test.shape[1]*irf_test.shape[2]))

        idx_reshaped = np.reshape(idx,(idx.shape[0]*idx.shape[1])) + shift

        ids = np.clip(idx_reshaped[:, np.newaxis] + np.arange(gates+np.round((orig_step_size*orig_nt-dt_irf*avg_irf.shape[0])/dt_irf).astype(np.int64)),a_min=0,a_max=irf_test.shape[0]-1)
        x = np.moveaxis(irf_test_reshaped,0,1)
        x = np.take_along_axis(x,ids.astype(np.int64),axis=1)
        x = np.moveaxis(x,1,0)

        irf_test = np.reshape(x,(x.shape[0],irf_test.shape[1],irf_test.shape[2]))

        interpolator = interp1d(np.linspace(0,irf_test.shape[0]-1,irf_test.shape[0]),irf_test,kind='slinear', fill_value="extrapolate",axis=0)
        irf_test = interpolator(np.linspace(0,irf_test.shape[0]-1,nTG))
        irf_test = np.divide(irf_test,np.amax(irf_test,axis=0),np.zeros_like(irf_test),where=np.amax(irf_test,axis=0)>0)

        irf_test = tf.cast(tf.convert_to_tensor(irf_test),tf.float32)
        tpsf_irf = tfcubify(irf_test,np.array((nTG,xX,yY)))

        ddirs = [args.inpath]

        for ii, ddir in zip(range(len(ddirs)),ddirs):
            print('Processing {} ....'.format(ddir))

            if args.channel:
                times = [f for f in os.listdir(ddir) if re.search(r'ANG000_CHN01',f)]
            else:
                times = [f for f in os.listdir(ddir) if re.search(r'ANG000_CHN00',f)]
            times = times[args.start_t:args.end_t]
            print(len(times))
            os.makedirs(os.path.join(ddir,'taus_cnn'),exist_ok=True)
            save_dir = os.path.join(ddir,'taus_cnn')

            for i, time in zip(range(len(times)),times):

                print(time)

                t = re.split('_',re.split(r'TM',time)[-1])[0]

                im = input_flim_vol(os.path.join(ddir,time),nt=nt,nz=nz)

                if args.version == 1:
                    taus = np.zeros((len(times),im.shape[1],im.shape[2],im.shape[3]))
                elif args.version == 2:
                    taus_1P = np.zeros((len(times),im.shape[1],im.shape[2],im.shape[3]))
                    taus_2P = np.zeros((len(times),im.shape[1],im.shape[2],im.shape[3]))
                    taus_RP = np.zeros((len(times),im.shape[1],im.shape[2],im.shape[3]))
                    taus_M = np.zeros((len(times),im.shape[1],im.shape[2],im.shape[3]))

                for j in range(im.shape[1]):

                    print(j)

                    im_slice = im[:,int(j),...]

                    im2 = np.clip(savgol_filter(im_slice,5,3,axis=0),a_min=5e-5,a_max=np.inf)

                    mask = np.sum(im2,axis=0) < 10

                    im2[:,mask] = 0.0

                    im4 = np.divide(im2,np.amax(im2,axis=0),np.zeros(im2.shape,float),where=np.amax(im2,axis=0)>0.0)

                    x = tf.convert_to_tensor(im4)
                    x = tf.reshape(x,[im4.shape[0],im4.shape[1]*im4.shape[2]])
                    
                    t = np.arange(gates+np.round((orig_step_size*orig_nt-dt_irf*avg_irf.shape[0])/dt_irf).astype(np.int64))*dt_irf
                    new_step = step_size
                    if tf.shape(x).numpy()[0]*new_step < orig_nt*orig_step_size:
                        for k in range(np.ceil((orig_step_size*orig_nt-step_size*tf.shape(x).numpy()[0])/step_size).astype(int)):
                            x = tf.pad(x,tf.constant([[0,1],[0,0]]),mode='CONSTANT', constant_values=0.0)
                    x = x[...,tf.newaxis]
                    x = tf.image.resize(x,[np.round((step_size*x.shape[0])/dt_irf).astype(np.int64),x.shape[1]])
                    if x.shape[0] < irf_test_reshaped.shape[0]:
                        x = tf.pad(x,tf.constant([[0,irf_test_reshaped.shape[0]-x.shape[0]-1],[0,0],[0,0]]),mode='CONSTANT', constant_values=0.0)

                    new_step = dt_irf
                    if new_step*tf.shape(x).numpy()[0] > np.amax(t):
                        ids = np.clip(idx_reshaped[:, np.newaxis] + np.arange(gates+np.floor((orig_step_size*orig_nt-dt_irf*avg_irf.shape[0])/dt_irf)),a_min=0, a_max=x.shape[0]-1)
                    x = tf.keras.ops.moveaxis(x,0,1)
                    x = tf.keras.ops.take_along_axis(x,ids[...,np.newaxis],axis=1)
                    x = tf.keras.ops.moveaxis(x,1,0)
                    
                    if new_step*tf.shape(x).numpy()[0] > np.amax(t):
                        idd = np.where(np.arange(tf.shape(x).numpy()[0])*new_step <= t[-1])[0][-1]
                        x = x[:idd,...]

                    x = tf.image.resize(x,[nTG,im4.shape[1]*im4.shape[2]],antialias=True)

                    x = tf.reshape(x,[nTG,im4.shape[1],im4.shape[2]])

                    tpsfT_test_new = tfcubify(x,np.array((nTG,xX,yY)))

                    test_arr = tf.stack([tf.keras.ops.moveaxis(tpsfT_test_new,1,-1),tf.keras.ops.moveaxis(tpsf_irf,1,-1)],axis=-1)

                    if args.version == 1:
                        t2P = model.predict(tf.convert_to_tensor(test_arr))

                        t2P = t2P['imgT2']

                        t2P_image = tfuncubify(t2P,(im.shape[2],im.shape[3],1))

                        taus[int(i),int(j),...] = t2P_image[:,:,0].numpy()

                    elif args.version == 2:
                        
                        testV = model.predict(tf.convert_to_tensor(test_arr))
                        t1P = testV['imgT1'] # Predicted t1 values
                        t1P_image = tfuncubify(t1P,(im.shape[2],im.shape[3],1))
                        t2P = testV['imgT2'] # Predicted t2 values
                        t2P_image = tfuncubify(t2P,(im.shape[2],im.shape[3],1))
                        tRP = testV['imgTR'] # Predicted AR values
                        tRP_image = tfuncubify(tRP,(im.shape[2],im.shape[3],1))

                        taus_1P[int(i),int(j),...] = t1P_image[:,:,0].numpy()
                        taus_2P[int(i),int(j),...] = t2P_image[:,:,0].numpy()
                        taus_RP[int(i),int(j),...] = tRP_image[:,:,0].numpy()
                        taus_M[int(i),int(j),...] = (tRP_image*t1P_image + (1.0 - tRP_image)*t2P_image)[:,:,0]


                    else:
                        exit(f'error: version should be either 1 or 2')

                if args.version == 1:            
                    tff.imwrite(os.path.join(save_dir,'taus.tif'),taus,metadata={'axes': 'TZYX'})

                elif args.version == 2:            
                    tff.imwrite(os.path.join(save_dir,'taus_1P.tif'),taus_1P,metadata={'axes': 'TZYX'})
                    tff.imwrite(os.path.join(save_dir,'taus_2P.tif'),taus_2P,metadata={'axes': 'TZYX'})
                    tff.imwrite(os.path.join(save_dir,'taus_RP.tif'),taus_RP,metadata={'axes': 'TZYX'})
                    tff.imwrite(os.path.join(save_dir,'taus_M.tif'),taus_M,metadata={'axes': 'TZYX'})
                else:
                    exit(f'error: version should be either 1 or 2')

    

