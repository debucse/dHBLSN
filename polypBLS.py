"""
Created on Tue May  3 18:55:01 2022

@author: kaushiki and debapriyo
"""

from __future__ import print_function, division
import os

from tensorflow.keras.utils import to_categorical

from BLS_functions import *

import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
import time
import numpy as np



# conv
conv_c = 0.005
conv_s = 0.85
conv_e = 10000
kernel_number_para = 0.25



dtcwt_c = 0.005
dtcwt_s = 0.75
dtcwt_e = 100




fusion_c = .001  

def polypBLS(x_train, train_y, x_val, val_y, x_test, test_y):
    print('conv_c:', conv_c, 'conv_s:', conv_s, 'conv_e:', conv_e)
    x_train_conv = np.load("x_train_conv.npy")  # -train
    x_val_conv = np.load("x_val_conv.npy")  # -val
    x_test_conv = np.load("x_test_conv.npy")  # -testures

    x_train_dtcwt = np.load("x_train_dtcwt.npy")
    x_val_dtcwt = np.load("x_val_dtcwt.npy")
    x_test_dtcwt = np.load("x_test_dtcwt.npy")
    conv_shape = x_train_conv.shape[1]
    
    dtcwt_shape = x_train_dtcwt.shape[1]
    InOfEnhLayer1WithBias = np.hstack([x_train_conv, 0.1 * np.ones((x_train_conv.shape[0], 1))])
    if conv_shape >= conv_e:
        random.seed(67797325)
        weiOfEnhLayer1 = LA.orth(2 * random.randn(conv_shape + 1, conv_e)) - 1
    else:
        random.seed(67797325)
        weiOfEnhLayer1 = LA.orth(2 * random.randn(conv_shape + 1, conv_e).T - 1).T
    tempOfOutOfEnhLayer1 = np.dot(InOfEnhLayer1WithBias, weiOfEnhLayer1)
    parameterOfShrink1 = conv_s / np.max(tempOfOutOfEnhLayer1)
    OutOfEnhLayer1 = tansig(tempOfOutOfEnhLayer1 * parameterOfShrink1)

   
    InputOfC1Layer = np.hstack([x_train_conv, OutOfEnhLayer1])
    pinvOfInputC1 = pinv(InputOfC1Layer, conv_c)
    C1Weight = np.dot(pinvOfInputC1, train_y)
    OutC1 = np.dot(InputOfC1Layer, C1Weight)


    InOfEnhLayer3WithBias = np.hstack([x_train_dtcwt, 0.1 * np.ones((x_train_dtcwt.shape[0], 1))])
    
    if dtcwt_shape >= dtcwt_shape:
        random.seed(67797325)
        weiOfEnhLayer3 = LA.orth(2 * random.randn(dtcwt_shape + 1, dtcwt_e)) - 1
    else:
        random.seed(67797325)
        weiOfEnhLayer3 = LA.orth(2 * random.randn(dtcwt_shape + 1, dtcwt_e).T - 1).T
    tempOfOutOfEnhLayer3 = np.dot(InOfEnhLayer3WithBias, weiOfEnhLayer3)
    parameterOfShrink3 = dtcwt_s / np.max(tempOfOutOfEnhLayer3)
    OutOfEnhLayer3 = tansig(tempOfOutOfEnhLayer3 * parameterOfShrink3)

    
    InputOfC3Layer = np.hstack([x_train_dtcwt, OutOfEnhLayer3])
    pinvOfInputC3 = pinv(InputOfC3Layer, dtcwt_c)
    C3Weight = np.dot(pinvOfInputC3, train_y)
    OutC3 = np.dot(InputOfC3Layer, C3Weight)
    
    OutC1_N = sigmoid(OutC1)
    
    OutC3_N = sigmoid(OutC3)
  

    
    InputOfOutputLayer = np.hstack([OutC1_N, OutC3_N])  #
    pinvOfInput = pinv(InputOfOutputLayer, fusion_c)
    OutputWeight = np.dot(pinvOfInput, train_y)  # 全局违逆
    time_end = time.time()
    trainTime = time_end - time_start


    OutputOfTrain = np.dot(InputOfOutputLayer, OutputWeight)
    trainAcc = show_accuracy(OutputOfTrain, train_y)
    print('Training accurate is', trainAcc * 100, '%')
    print('Training time is ', trainTime, 's')

    InOfEnhLayer1WithBiasVal = np.hstack([x_val_conv, 0.1 * np.ones((x_val_conv.shape[0], 1))])
    tempOfOutOfEnhLayer1Val = np.dot(InOfEnhLayer1WithBiasVal, weiOfEnhLayer1)
    OutOfEnhLayer1Val = tansig(tempOfOutOfEnhLayer1Val * parameterOfShrink1)


    InOfEnhLayer3WithBiasVal = np.hstack([x_val_dtcwt, 0.1 * np.ones((x_val_dtcwt.shape[0], 1))])
    tempOfOutOfEnhLayer3Val = np.dot(InOfEnhLayer3WithBiasVal, weiOfEnhLayer3)
    OutOfEnhLayer3Val = tansig(tempOfOutOfEnhLayer3Val * parameterOfShrink3)

    
    InputOfC1LayerVal = np.hstack([x_val_conv, OutOfEnhLayer1Val])
    OutC1Val = np.dot(InputOfC1LayerVal, C1Weight)

    

    InputOfC3LayerVal = np.hstack([x_val_dtcwt, OutOfEnhLayer3Val])
    OutC3Val = np.dot(InputOfC3LayerVal, C3Weight)

   
   
    OutC1Val_N = sigmoid(OutC1Val)
    
    OutC3Val_N = sigmoid(OutC3Val)
   

  
    InputOfOutputLayerVal = np.hstack([OutC1Val_N, OutC3Val_N])  
    OutputOfVal = np.dot(InputOfOutputLayerVal, OutputWeight)
    time_end = time.time()
   
    valAcc = show_accuracy(OutputOfVal, val_y)
    print('Val accurate is', valAcc * 100, '%')
    

    
    InOfEnhLayer1WithBiasTest = np.hstack([x_test_conv, 0.1 * np.ones((x_test_conv.shape[0], 1))])
    tempOfOutOfEnhLayer1Test = np.dot(InOfEnhLayer1WithBiasTest, weiOfEnhLayer1)
    OutOfEnhLayer1Test = tansig(tempOfOutOfEnhLayer1Test * parameterOfShrink1)

    

    InOfEnhLayer3WithBiasTest = np.hstack([x_test_dtcwt, 0.1 * np.ones((x_test_dtcwt.shape[0], 1))])
    tempOfOutOfEnhLayer3Test = np.dot(InOfEnhLayer3WithBiasTest, weiOfEnhLayer3)
    OutOfEnhLayer3Test = tansig(tempOfOutOfEnhLayer3Test * parameterOfShrink3)

    InputOfC1LayerTest = np.hstack([x_test_conv, OutOfEnhLayer1Test])
    OutC1Test = np.dot(InputOfC1LayerTest, C1Weight)

   

    InputOfC3LayerTest = np.hstack([x_test_dtcwt, OutOfEnhLayer3Test])
    OutC3Test = np.dot(InputOfC3LayerTest, C3Weight)

   

    
    OutC1Test_N = sigmoid(OutC1Test)
   
    OutC3Test_N = sigmoid(OutC3Test)
   

    
    InputOfOutputLayerTest = np.hstack([OutC1Test_N, OutC3Test_N])  #
    
    OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)
    
    
    testAcc = show_accuracy(OutputOfTest, test_y)
    
    
    
    print(test_y.shape)
    vb=[]
    import numpy
    mj = numpy.empty(shape=(1,75264),dtype='object')
    print(OutputOfTest.shape[0])
    print(OutputOfTest.shape[1])
    preds_train_t = (OutputOfTest > 0.05)
        #print(preds_train_t)
        #for i in range(OutputOfTest.shape[0]):
         #   for j in range(OutputOfTest.shape[1]):
                
          #      if OutputOfTest[i][j].any()>0.01 :
    vb.append(OutputOfTest)
    vb=np.array(vb)
        #print(vb[:,5,3])
    import csv 
    a = [[1,2,3,4],[5,6,7,8]] 
    with open("new_file_nbb80_pred.csv",'w', newline='') as my_csv: 
           csvWriter = csv.writer(my_csv,delimiter=',')
           for i in range(0, 25600):   
            for j in range(0, 1): 
              #  if vb[:,i,j]>0.5:  
                    csvWriter.writerows(map(lambda x: [x], vb[:,i,j]))
                    #mj[:,i]=(vb[:,i,j])  
       # if vb.any()>0.2 :
        #
       
        
    vb1=[]
    vb1.append(test_y)
    vb1=np.array(vb1)
        
    import csv 
    a = [[1,2,3,4],[5,6,7,8]] 
    with open("new_file_nbb81_actual.csv",'w', newline='') as my_csv: 
           csvWriter = csv.writer(my_csv,delimiter=',')
           for i in range(0,25600):   
            for j in range(0, 1): 
              #  if vb[:,i,j]>0.5:  
                    csvWriter.writerows(map(lambda x: [x], vb1[:,i,j]))
                    #mj[:,i]=(vb1[:,i,j])  
       # if vb.any()>0.2 :
        #
    #print(mj)
        
    
    
    
    
    
    
    
    print(OutputOfTest.shape)
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    return trainAcc,valAcc,testAcc
    
    
    
    
    print('Testing accurate is', testAcc * 100, '%')
    print('Testing time is ', testTime, 's')
    return trainAcc,valAcc,testAcc



    
    
    
    
    
    
def polyp_AddEnhanceNodes(x_train, train_y, x_val, val_y, x_test, test_y,s,c,N1,N2,N3,L,M):
    vb=[]
    
    print('conv_c:', conv_c, 'conv_s:', conv_s, 'conv_e:', conv_e)
    x_train_conv = np.load("x_train_conv.npy")  # -train
    x_val_conv = np.load("x_val_conv.npy")  # -val
    x_test_conv = np.load("x_test_conv.npy")  # -testures

    x_train_dtcwt = np.load("x_train_dtcwt.npy")
    x_val_dtcwt = np.load("x_val_dtcwt.npy")
    x_test_dtcwt = np.load("x_test_dtcwt.npy")
    conv_shape = x_train_conv.shape[1]
    
    dtcwt_shape = x_train_dtcwt.shape[1]
    InOfEnhLayer1WithBias = np.hstack([x_train_conv, 0.1 * np.ones((x_train_conv.shape[0], 1))])
    
    if N1*N2>=N3:
        random.seed(67797325)
        weiOfEnhLayer1 = LA.orth(2 * random.randn(N2*N1+1,N3))-1
        #print(" weightOfEnhanceLayer")
        #print( weiOfEnhLayer1.shape)
    else:
        random.seed(67797325)
       # print(N1)
       # print(N2)
       # print(N3)
        weiOfEnhLayer1 = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
        #print(" weightOfEnhanceLayer")
       # print( weiOfEnhLayer1.shape)
    
    
    
    weightOfEnhanceLayer=weiOfEnhLayer1
    tempOfOutputOfEnhanceLayer = np.dot(InOfEnhLayer1WithBias,weightOfEnhanceLayer)
    parameterOfShrink = conv_s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
  
    InputOfOutputLayer = np.hstack([x_train_conv,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer,conv_c)
    OutputWeight = pinvOfInput.dot(train_y) 

  
    time_end=time.time() 
    trainTime = time_end - time_start
    
    
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    
    
    
    
    
    
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    
    
  
    time_start = time.time()  

 

    
    InputOfEnhanceLayerWithBiasTest = np.hstack([x_test_conv, 0.1 * np.ones((x_test_conv.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    

    InputOfOutputLayerTest = np.hstack([x_test_conv,OutputOfEnhanceLayerTest])
 
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    time_end=time.time() 
    testTime = time_end - time_start
    testAcc = show_accuracy(OutputOfTest,test_y)
    #print('Testing accurate is' ,testAcc*100,'%')
    #print('Testing time is ',testTime,'s')

    

    
    parameterOfShrinkAdd = []
    #print("hey")
    L=5
   # print(L)              



    B=0
    B_prev=0

    
    
    for e in range(20):
        #print("hii")
        #time_start=time.time()
       # print(conv_shape)
       # print(M)

       
        if N1*N2>=M:
          random.seed(e)
          weiOfEnhLayer1 = LA.orth(2 * random.randn(N2*N1+1,M))-1
     
        else:
          random.seed(e)
          weiOfEnhLayer1 = LA.orth(2 * random.randn(N2*N1+1,M).T-1).T
  
        weightOfEnhanceLayerAdd=weiOfEnhLayer1
        
        B_prev=B;

        tempOfOutputOfEnhanceLayerAdd = np.dot(InOfEnhLayer1WithBias,weiOfEnhLayer1)
        parameterOfShrinkAdd.append(conv_s/np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd*parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer,OutputOfEnhanceLayerAdd])
        
        
        
        #tempOfLastLayerInput = np.hstack([InputOfOutputLayer,0.1*OutputOfEnhanceLayerAdd])
        import numpy
        print(InputOfOutputLayer.shape)
        import csv
        
        
        
        
        
       
                    #print(OutputOfEnhanceLayerAdd.shape)
        D = (pinvOfInput.dot(OutputOfEnhanceLayerAdd))
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T,D)).I.dot(np.dot(D.T,pinvOfInput))
#
        else:
            B = pinv(C,conv_c)


        B=B+0.1*B_prev
       
        pinvOfInput = np.vstack([(pinvOfInput - (D.dot(B))),B])
        
        OutputWeightEnd = pinvOfInput.dot(train_y)
        result = np.zeros([200,2])
        result=np.append(OutputWeight, result, axis=0)
        
      
        


        OutputWeightEnd=OutputWeightEnd+ result
        
        InputOfOutputLayer = tempOfLastLayerInput
        
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)
        OutputOfTrain1= sigmoid(OutputOfTrain1)
        TrainingAccuracy = show_accuracy(OutputOfTrain1,train_y)
        
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %' )
        

        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])
        InputOfOutputLayerTest=np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest1 = InputOfOutputLayerTest.dot(OutputWeightEnd)
        OutputOfTest1= sigmoid(OutputOfTest1)
      



#2D-DTCWT LAYER
      

    InOfEnhLayer1WithBias = np.hstack([x_train_dtcwt, 0.1 * np.ones((x_train_dtcwt.shape[0], 1))])
    
    if N1*N2>=N3:
        random.seed(67797325)
        weiOfEnhLayer1 = LA.orth(2 * random.randn(N2*N1+1,N3))-1
        #print(" weightOfEnhanceLayer")
        #print( weiOfEnhLayer1.shape)
    else:
        random.seed(67797325)
       
        weiOfEnhLayer1 = LA.orth(2 * random.randn(N2*N1+1,N3).T-1).T
        #print(" weightOfEnhanceLayer")
       # print( weiOfEnhLayer1.shape)
    
    
    
    weightOfEnhanceLayer=weiOfEnhLayer1
    tempOfOutputOfEnhanceLayer = np.dot(InOfEnhLayer1WithBias,weightOfEnhanceLayer)
    parameterOfShrink = conv_s/np.max(tempOfOutputOfEnhanceLayer)
    OutputOfEnhanceLayer = tansig(tempOfOutputOfEnhanceLayer * parameterOfShrink)
    
    InputOfOutputLayer = np.hstack([x_train_dtcwt,OutputOfEnhanceLayer])
    pinvOfInput = pinv(InputOfOutputLayer,dtcwt_c)
    OutputWeight = pinvOfInput.dot(train_y) 

    
    OutputOfTrain = np.dot(InputOfOutputLayer,OutputWeight)
    
    
    
    
    
    
    trainAcc = show_accuracy(OutputOfTrain,train_y)
    
  

    
    InputOfEnhanceLayerWithBiasTest = np.hstack([x_test_dtcwt, 0.1 * np.ones((x_test_dtcwt.shape[0], 1))])
    tempOfOutputOfEnhanceLayerTest = np.dot(InputOfEnhanceLayerWithBiasTest,weightOfEnhanceLayer)

    OutputOfEnhanceLayerTest = tansig(tempOfOutputOfEnhanceLayerTest * parameterOfShrink)    

    InputOfOutputLayerTest = np.hstack([x_test_dtcwt,OutputOfEnhanceLayerTest])
 
    OutputOfTest = np.dot(InputOfOutputLayerTest,OutputWeight)
    
    testAcc = show_accuracy(OutputOfTest,test_y)
   
    
    parameterOfShrinkAdd = []
    
    L=5
                



    B=0
    B_prev=0

    
    
    for e in range(20):
       
       
        if N1*N2>=M:
          random.seed(e)
          weiOfEnhLayer1 = LA.orth(2 * random.randn(N2*N1+1,M))-1
     
        else:
          random.seed(e)
          weiOfEnhLayer1 = LA.orth(2 * random.randn(N2*N1+1,M).T-1).T
  
        weightOfEnhanceLayerAdd=weiOfEnhLayer1
        
        B_prev=B;

        tempOfOutputOfEnhanceLayerAdd = np.dot(InOfEnhLayer1WithBias,weiOfEnhLayer1)
        parameterOfShrinkAdd.append(conv_s/np.max(tempOfOutputOfEnhanceLayerAdd))
        OutputOfEnhanceLayerAdd = tansig(tempOfOutputOfEnhanceLayerAdd*parameterOfShrinkAdd[e])
        tempOfLastLayerInput = np.hstack([InputOfOutputLayer,OutputOfEnhanceLayerAdd])
        
        
        
        #tempOfLastLayerInput = np.hstack([InputOfOutputLayer,0.1*OutputOfEnhanceLayerAdd])
        import numpy
        print(InputOfOutputLayer.shape)
        import csv
        
        
        
        
        
       
                    #print(OutputOfEnhanceLayerAdd.shape)
        D = (pinvOfInput.dot(OutputOfEnhanceLayerAdd))
        C = OutputOfEnhanceLayerAdd - InputOfOutputLayer.dot(D)
        if C.all() == 0:
            w = D.shape[1]
            B = np.mat(np.eye(w) - np.dot(D.T,D)).I.dot(np.dot(D.T,pinvOfInput))
#
        else:
            B = pinv(C,dtcwt_c)


        B=B+0.1*B_prev
       
        pinvOfInput = np.vstack([(pinvOfInput - (D.dot(B))),B])
        
        OutputWeightEnd = pinvOfInput.dot(train_y)
        result = np.zeros([200,2])
        result=np.append(OutputWeight, result, axis=0)
        
        
        


        OutputWeightEnd=OutputWeightEnd+ result
        
        InputOfOutputLayer = tempOfLastLayerInput
        
        OutputOfTrain1 = InputOfOutputLayer.dot(OutputWeightEnd)
        OutputOfTrain1= sigmoid(OutputOfTrain1)
        TrainingAccuracy = show_accuracy(OutputOfTrain1,train_y)
        
        print('Incremental Training Accuracy is :', TrainingAccuracy * 100, ' %' )
        

        time_start = time.time()
        OutputOfEnhanceLayerAddTest = tansig(InputOfEnhanceLayerWithBiasTest.dot(weightOfEnhanceLayerAdd) * parameterOfShrinkAdd[e])
        InputOfOutputLayerTest=np.hstack([InputOfOutputLayerTest, OutputOfEnhanceLayerAddTest])

        OutputOfTest2 = InputOfOutputLayerTest.dot(OutputWeightEnd)
        OutputOfTest2= sigmoid(OutputOfTest2)


        

        InputOfOutputLayerTest = np.hstack([OutputOfTest1, OutputOfTest2])  #
    
        OutputOfTest = np.dot(InputOfOutputLayerTest, OutputWeight)



        TestingAcc = show_accuracy(OutputOfTest1,test_y)
        
        
        print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %' )

        OutputWeight=OutputWeightEnd

        vb=[]
        import numpy
        #mj = numpy.empty(shape=(1,75264),dtype='object')
        
        #print(preds_train_t)
        #for i in range(predictLabel.shape[0]):
         #   for j in range(predictLabel.shape[1]):
                
          #      if predictLabel[i][j].any()>0.01 :
        vb.append(pinvOfInput)
        vb=np.array(vb)
        #print(vb[:,5,3])
        import csv 
        a = [[1,2,3,4],[5,6,7,8]] 
        with open("add_enhance_pinvinput%d.csv"%e,'w', newline='') as my_csv: 
           csvWriter = csv.writer(my_csv,delimiter=',')
           for i in range(0, 1):   
            for j in range(0, 1): 
              #  if vb[:,i,j]>0.5:  
                    csvWriter.writerows(map(lambda x: [x], vb[:,i,j]))
                    
       # if vb.any()>0.2 :
        #
        
        
        vb1=[]
        vb1.append(test_y)
        vb1=np.array(vb1)
        
        import csv 
        a = [[1,2,3,4],[5,6,7,8]] 
        with open("new_file_nbb81_actual.csv",'w', newline='') as my_csv: 
           csvWriter = csv.writer(my_csv,delimiter=',')
           for i in range(0,94080):   
            for j in range(0, 1): 
              #  if vb[:,i,j]>0.5:  
                    csvWriter.writerows(map(lambda x: [x], vb1[:,i,j]))
                      
       # if vb.any()>0.2 :
        #
        
        
        
        #print('Incremental Testing Accuracy is : ', TestingAcc * 100, ' %' )
       
    return trainAcc





from skimage.io import imread, imshow, imread_collection, concatenate_images
from tqdm import tqdm
from tqdm import tqdm_notebook, tnrange
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
IMG_HEIGHT=32
IMG_WIDTH=32
IMG_CHANNELS=3
from skimage.io import imread, imshow, imread_collection, concatenate_images
from tqdm import tqdm
from tqdm import tqdm_notebook, tnrange
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import scipy.io as scio
from sklearn.model_selection import train_test_split
if __name__ == '__main__':

  
    from sklearn.model_selection import train_test_split
   
    from sklearn.model_selection import train_test_split
    num_train_samples = 76800
    x_train = np.empty((num_train_samples, 32, 32,3), dtype='uint8')
    Y_train = np.empty((num_train_samples,), dtype='uint8')
    
    dataFile = './kvasirdata.mat'
    data = scio.loadmat(dataFile)
    X_train = np.double(data['train_x'])
    X_train = X_train.reshape(76800, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    
    print("h")
    print(X_train.shape)
    print("h")
    Y_train= np.double(data['train_y'])
    x_test = np.double(data['test_x'])
    x_test = x_test.reshape(25600, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
    print(x_test.shape)
    y_test = np.double(data['test_y'])
    
    

    
    
    Y_train = np.reshape(Y_train, (len(Y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))
    
  
    
    print(X_train.shape)
    print(x_test.shape)
    print(Y_train.shape)
    print(y_test.shape)
    #
    x_train = X_train[0:50000, ]
    x_val = X_train[50000:76800, ]

   
    y_train=Y_train[0:50000, ]
    y_train = np.reshape(y_train, (len(y_train), 1))
    y_val = Y_train[50000:76800, ]
    y_val = np.reshape(y_val, (len(y_val), 1))
    y_test = y_test
    
    
    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)
    print('x_test shape:', x_test.shape)
    
    print('y_train shape:', y_train.shape)
    print('y_val shape:', y_val.shape)
    print('ytest shape:', y_test.shape)

    print('x_train shape:', x_train.shape[0])
    print('x_val shape:', x_val.shape[0])
    print('x_test shape:', x_test.shape[0])

    
    N1 = 10 #  # of nodes belong to each window
    N2 = 2  #  # of windows -------Feature mapping layer
    N3 = 100 #  # of enhancement nodes -----Enhance layer
    L = 5 #  # of incremental steps 
    M = 200 #  # of adding enhance nodes
    s = 0.8  #  shrink coefficient
    C = .005 # Regularization coefficient
    c=0.00
     

    print('-------------------polyp_BLS_BASE---------------------------')

    polypBLS(x_train, y_train, x_val, y_val, x_test, y_test)



    print('-------------------polyp_BLS_ENHANCE------------------------')
    polyp_AddEnhanceNodes(x_train, train_y, x_val, val_y, x_test, test_y, s, c, N1, N2, N3, L, M)
