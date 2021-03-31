import os, h5py, glob, re, argparse
from PIL import Image
import numpy as  np

def IoU(rect1, rect2):
    #Calculate intersection-over-union between two rectangles
    
    ileft = np.maximum(rect1[0],rect2[0])
    iright = np.minimum(rect1[0]+rect1[2],rect2[0]+rect2[2])
    itop = np.maximum(rect1[1],rect2[1])
    ibottom = np.minimum(rect1[1]+rect1[3],rect2[1]+rect2[3])
    intersection = (iright-ileft)*(ibottom-itop)
    union = rect1[2]*rect1[3] + rect2[2]*rect2[3] - intersection
    return(intersection/union)

def rect_regression(rect, anchor):
    #Convert rectangle into a regression target
   return(np.array([(rect[0]-anchor[0])/anchor[2],(rect[1]-anchor[1])/anchor[3],
                     np.log(rect[2]/anchor[2]),np.log(rect[3]/anchor[3])]))

def initialize_weights():
    #Randomly initialize weight tensors
    W1 = 0.001*np.random.randn(4608,512)
    W2 = 0.001*np.random.randn(9,512,5)
    return(W1,W2)

def load_weights(filename):
    #Load pre-trained weights 
    with h5py.File(filename,'r')  as f:
        W1 = f['W1'][:]
        W2 = f['W2'][:]
    return(W1,W2)

def save_weights(filename,W1,W2):
    #Save trained weights
    with h5py.File(filename,'w') as f:
        f.create_dataset('W1',data=W1)
        f.create_dataset('W2',data=W2)


class Face_Dataset(object):
    def __init__(self, datadir):
        self.imagefiles = glob.glob(os.path.join(datadir, 'images/*/*.jpg'))
        sizes = np.array([128,256,512])*(224/1024) # image had 1024 rows; resized has 224
        w = np.outer(sizes, np.sqrt([0.5,1,2])).flatten()
        h = np.outer(sizes, np.sqrt([2,1,0.5])).flatten()
        # anchor rectangles: [x,y,w,h], where x, y, w, h are each 196x9 matrices (xy by a)
        self.anchors = np.array([
            [[x for a in range(9)] for y in np.arange(8,16*14,16) for x in np.arange(8,16*14,16)],
            [[y for a in range(9)] for y in np.arange(8,16*14,16) for x in np.arange(8,16*14,16)],
            [[w[a] for a in range(9)] for y in np.arange(8,16*14,16) for x in np.arange(8,16*14,16)],
            [[h[a] for a in range(9)] for y in np.arange(8,16*14,16) for x in np.arange(8,16*14,16)]])
        
    def __len__(self):
        return(len(self.imagefiles))
    
    def target_tensor(self, rects):
        target = np.zeros((9,196,5))
        for rect in rects:
            similarities = IoU(rect, self.anchors)
            xylist,alist = (similarities > 0.7).nonzero()
            for xy,a in zip(xylist,alist):  
                target[a,xy,0:4] = rect_regression(rect,self.anchors[:,xy,a])
                target[a,xy,4] = 1
            else:
                xy,a  = np.unravel_index(np.argmax(similarities), similarities.shape)
                target[a,xy,0:4] = rect_regression(rect,self.anchors[:,xy,a])
                target[a,xy,4] = 1
        return(target)
    
    def __getitem__(self, n):

        imagepath = self.imagefiles[n]
        image = np.asarray(Image.open(imagepath)).astype('float64')
        image = (image - np.amin(image)) / (1e-6+np.amax(image)-np.amin(image))
        with h5py.File(re.sub(r'images','features',re.sub(r'.jpg','.hdf5',imagepath)),'r') as f:
            features = f['features'][:] / np.sum(np.abs(f['features'][:]))
        with open(re.sub(r'images','rects',re.sub(r'.jpg','.txt',imagepath))) as f:
            rects=np.array([[float(w) for w in line.strip().split() ] for line in f.readlines()])
        for rect in rects:
            rect[0] += 0.5*rect[2]
            rect[1] += 0.5*rect[3]
        target = self.target_tensor(rects)
        return({'image':image, 'features':features, 'rects':rects, 'target':target})

###############################################################################


def concatenate(features):
    concatenation = np.zeros((196,4608))
    for i in range(0,196):
        twoD_index = np.unravel_index(i,(14,14))
        y_img_idx = twoD_index[0]
        x_img_idx = twoD_index[1]

        y_Neg = y_img_idx - 1
        y = y_img_idx
        y_Pos = y_img_idx + 1

        x_Neg = x_img_idx - 1
        x = x_img_idx
        x_Pos = x_img_idx + 1 
        
        if((y_img_idx - 1) < 0):
            y_Neg = 0;
        elif((y_img_idx + 1) > 13):
            y_Pos = 13
        
        if((x_img_idx - 1) < 0):
            x_Neg = 0;
        elif((x_img_idx + 1) > 13):
            x_Pos = 13

        batch_3_1 = np.concatenate((features[0,:,y_Neg,x_Neg], features[0,:,y_Neg,x], features[0,:,y_Neg,x_Pos]), axis = None)
        batch_3_2 = np.concatenate((features[0,:,y,x_Neg], features[0,:,y,x],features[0,:,y,x_Pos]), axis = None)
        batch_3_3 = np.concatenate((features[0,:,y_Pos,x_Neg], features[0,:,y_Pos,x],features[0,:,y_Pos,x_Pos]), axis = None)

        concatenation[i,:] = np.concatenate((batch_3_1, batch_3_2, batch_3_3), axis = None)
        
    return(concatenation)

def sigmoid(excitation):
    activation = np.zeros(excitation.shape)
    activation[excitation > -100] = 1/(1+np.exp(-excitation[excitation > -100]))
    return(activation)

def forward(concatenation, W1, W2):
   
    # concatenation * W1 -> [196,512] (excitation)
    # relu(exictation) -> get rid of all neg #'s in excitation
    # 
    #[196x512] [512x5] -> [196 x 5] for each anchor so [9x196x5]
    hypothesis = np.zeros((9,196,5))
    hidden = np.zeros((196,512))

    #excitation = x * W1 
    excitation = np.matmul(concatenation,W1)

    # hidden = relu(excitation) 
    hidden = abs(excitation * (excitation > 0))

        
   # linear activation to first 4 hypothesis
    for i in range(0,9):
        hypothesis[i,:,:] = np.matmul(hidden, W2[i,:,:])

    #apply sigmoid to last hypothesis
    for a in range(0,9):
        for i in range(0,196):
            hypothesis[a,i,4] = sigmoid(hypothesis[a,i,4]) 
    
    

    return(hypothesis, hidden)

def reverse_reg(regression_targets, anchors):
        rectangles = np.zeros(4)
        rectangles[0] = regression_targets[0] * anchors[2] + anchors[0];
        rectangles[1] = regression_targets[1] * anchors[3] + anchors[1]; 
        reg_min2 = min(regression_targets[2],np.log(2))
        rectangles[2] = np.exp(reg_min2) * anchors[2]
        reg_min3 = min(regression_targets[3],np.log(2))
        rectangles[3] = np.exp(reg_min3) * anchors[3]
        return rectangles


def detect_rectangles(hypothesis, number_to_return, Face_Dataset):
    # hypothesis = (9x196x5)
    # Face_Dataset.anchors.shape = (4x169x9)
    # print(Face_Dataset.anchors.shape)
    # print(best_rects.shape)
    ###############################
    best_rects = np.zeros((number_to_return,4))
    highest_hyp = []
    j = 0
    for a in range(0,9):
        for i in range(0,196):
            highest_hyp.append((hypothesis[a,i,4],a,i))
    highest_hyp = sorted(highest_hyp, key = lambda h : h[0], reverse = True)[:number_to_return]
    for h, a, i in highest_hyp:
        best_rects[j,:] = reverse_reg(hypothesis[a,i,0:4], Face_Dataset.anchors[:,i,a])
        j = j + 1
    return(best_rects)



# hypothesis passed into function is excitation of output layer 
def outputgrad(hypothesis, target):

    outputgrad = np.zeros((9,196,5))
    for a in range(0,9):
        for i in range(0,196):
            outputgrad[a,i,:] = (hypothesis[a,i,:] - target[a,i,:]) / (196 * 9)
    return(outputgrad)

def backprop(outputgrad, hidden, W2):
    #outputgrad -> dL / d(e) 

    #[196x5] * 
    backprop = np.zeros((196,512))
    output_W2_sum = np.zeros((196,512))
    output_W2_sum = np.sum(np.matmul(outputgrad[a], np.transpose(W2[a])) for a in range(9))
    
    
    Relu_derivative = np.heaviside(hidden,0) #0 if < 0 & 1 if > 0
    for i in range(0,196):
        for j in range(0,512):
            backprop[i,j] = output_W2_sum[i,j] * Relu_derivative[i,j]
    return(backprop)

def weightgrad(outputgrad, backprop, hidden, concatenation):
    dW1 = np.zeros((4608,512))
    dW2 = np.zeros((9,512,5))
    hiddenT = np.transpose(hidden)
    concatT = np.transpose(concatenation)
    for i in range(0,9):
        dW2[i] = np.matmul(hiddenT, outputgrad[i,:,:])
    dW1 = np.matmul(concatT, backprop) 
   
    return(dW1, dW2)

def update_weights(W1, W2, dW1, dW2, learning_rate):
    new_W2 = np.zeros(W2.shape)
    for i in range(0,4608):
        for j in range (0,512):
            new_W1[i,j] = W1[i,j] - learning_rate * dW1[i,j]

    for i in range(0,9):
        for j in range (0,512):
            for k in range (0,5):
                new_W2[i,j,k] = W2[i,j,k] - learning_rate*dW2[i,j,k]
   
    return(new_W1, new_W2)


###############################################################################
if __name__=="__main__":
    parser.add_argument('--datadir',default='data',help='''Set datadir.  Default: "data"''')
    parser.add_argument('-w','--weights',
                        help='''Name of HDF5 file containing initial weights.
                        Default: weights_trained.hdf5''')
    parser.add_argument('-i','--iters',metavar='N',type=int,default=1,
                        help='''# of training iterations, with batch size=1 image''')
    args = parser.parse_args()


    test_unravel = np.unravel_index(14,(14,14))
    test_arr = np.arange(4).reshape(2,2)
    sliced = test_arr[0,:]



    # Load the weights
    if args.weights == None:
        args.weights = 'weights_trained.hdf5'
    with h5py.File(args.weights,'r') as f:
        W1 = f['W1'][:]
        W2 = f['W2'][:]

    # Load the data
    face_data = Face_Dataset(args.datadir)
    
    # Perform the training iterations
    for i in range(args.iters):
        if i % len(face_data) == 0:
            print(i)
        else:
            print('.',end='')
        datum = face_data[i % len(face_data)]
        concatenation = concatenate(datum['features'])
        hypothesis, hidden = forward(concatenation, W1, W2)
        outputgrad = outputgrad(hypothesis, datum['target'])
        backprop = backprop(outputgrad, hidden, W2)
        dW1, dW2 = weightgrad(outputgrad, backprop, hidden, concatenation)
        W1, W2 = update_weights(W1, W2, dW1, dW2, 0.01)

    # Test
    best_rects = detect_rectangles(hypothesis, 10, face_data)

    # Save results
    datadir_name = args.datadir
    weights_name = os.path.splitext(args.weights)[0]
    experiment_name = '%s_%s_%d'%(datadir_name,weights_name,args.iters)
    with h5py.File('results_'+experiment_name+'.hdf5','w')  as f:
        f.create_dataset('features',data=datum['features'])
        f.create_dataset('target',data=datum['target'])
        f.create_dataset('concatenation',data=concatenation)
        f.create_dataset('hypothesis',data=hypothesis)
        f.create_dataset('hidden',data=hidden)
        f.create_dataset('outputgrad',data=outputgrad)
        f.create_dataset('backprop',data=backprop)
        f.create_dataset('dW1',data=dW1)
        f.create_dataset('dW2',data=dW2)
        f.create_dataset('best_rects',data=best_rects)
        f.create_dataset('W1',data=W1)
        f.create_dataset('W2',data=W2)
    with h5py.File('weights_'+experiment_name+'.hdf5','w') as f:
        f.create_dataset('W1',data=W1)
        f.create_dataset('W2',data=W2)