#import sys

import matplotlib
#matplotlib.use('Agg') 

#%matplotlib inline
# In[2]:
import pickle
import time, os
import shutil
#import tensorflow as tf
#import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.axes_grid1 import make_axes_locatable

from BasicFunc import mySaveFig, univAprox
from BasicFunc import my_fft, SelectPeakIndex

import dgl
import networkx as nx
import numpy as np
#plt.ion()

#tf.compat.v1.disable_eager_execution()
# In[3]:
isShowPic=0
Leftp=0.18
Bottomp=0.18
Widthp=0.88-Leftp
Heightp=0.9-Bottomp
poS=[Leftp,Bottomp,Widthp,Heightp]
ComputeStepFFT=30
LowFreqDrawAdd=1e-5 # for plot. plot y+this number, in case of 0 in log-log

SD=0.0  ### noise standard deviation in sample data, used in the fitting function

# y_name='|x|'
#y_name='sigmoid'
y_name='sinx'
# y_name='inv x'
def sigmoid(xx):
    return (1 / (1 + np.exp(-xx)))
def func0(xx,SD):
    y_sin=np.sin(xx)+np.sin(4*xx)+np.sin(6*xx)
    return y_sin

### discretized the func0 by sin_div
def func_to_approx(xx,sin_div,SD):
    y_sin=func0(xx,SD)
    if sin_div==0:
        return y_sin
    out_y = np.round(y_sin/sin_div)
    out_y2 = out_y * sin_div
    return out_y2

R_variable={}  ### used for saved all parameters and data

### mkdir a folder to save all output
R_variable['iscontinue']=0
if R_variable['iscontinue']:
    FolderName='Errordata/%s/'%('50129')
else:
    BaseDir = 'Errordata/'
    subFolderName = '%s'%(int(np.absolute(np.random.normal([1])*100000))//int(1)) 
    FolderName = '%s%s/'%(BaseDir,subFolderName)
    if not os.path.isdir(BaseDir):
        os.mkdir(BaseDir)
    os.mkdir(FolderName)
    os.mkdir('%smodel/'%(FolderName))
R_variable['FolderName']=FolderName 
 

### initialization standard deviation
R_variable['astddev']=0.05 # for weight
R_variable['bstddev']=0.05 # for bias terms2

### the length to discretized the continuous function
R_variable['sin_div']=0

### noise standard deviation in sample data, used in the fitting function
R_variable['SD']=SD

### hidden layer structure
# R_variable['hidden_units']=[20,10]
# R_variable['hidden_units']=[40,20]
R_variable['hidden_units']=[200,200,200,100]
#R_variable['hidden_units']=[1500,1500,500,500]
#R_variable['hidden_units']=[1]

# R_variable['hidden_units']=[800,800,400,400]

R_variable['learning_rate']= 0.001 #1e-5
R_variable['learning_rateDecay']=0
R_variable['rateDecayStep']=2000 

### setup for activation function
R_variable['seed']=0
R_variable['ActFuc']=1  ### 0: ReLU; 1: Tanh; 3:sin;4: x**5,, 5: sigmoid  6 sigmoid derivate
R_variable['train_size']=61;  ### training size
R_variable['batch_size']=int(np.floor(R_variable['train_size'])) ### batch size
R_variable['test_size']=int(201)  ### test size
R_variable['x_start']=-10 #math.pi*3 ### start point of input
R_variable['x_end']=10 #math.pi*3  ### end point of input

R_variable['isRec_y_test']=0 
R_variable['isFFT']=1 #compute FFT or not   
R_variable['ismovie']=0 # make a training movie

R_variable['tol']=2e-3
R_variable['Total_Step']=600000  ### the training step. Set a big number, if it converges, can manually stop training
R_variable['Record_Step']=1    ### every R_step compute Entropy or other values
R_variable['id']=0         ### index for how many step recorded

R_variable['y_name']=y_name     ### the target fitting function
R_variable['FolderName']=FolderName   ### folder for save images
 

# initialization for variables
x_end=R_variable['x_end'] 
R_variable['test_inputs'] =np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['test_size'],
                                                  endpoint=True),[R_variable['test_size'],1])


R_variable['train_inputs']=np.reshape(np.linspace(R_variable['x_start'], R_variable['x_end'], num=R_variable['train_size'],
                                                  endpoint=True),[R_variable['train_size'],1])
# ###randomly select training set from test set
# indperm = np.random.permutation(test_size)
# ind = indperm[0:Size_in]
# R_variable['train_inputs']=R_variable['test_inputs'][ind]
# ###randomly select training set from test set

R_variable['loss_test']=[]
R_variable['loss_train']=[]
R_variable['fft_fit_test']=[] 
R_variable['fft_true_test']=[]
R_variable['fft_0_train']=[]
R_variable['fft_fit_train']=[]
R_variable['fft_true_train']=[]
R_variable['fft_0_test']=[]
R_variable['y_test_all']=[]


R_variable['y_train_all']=[]

R_variable['y_test']=[]
R_variable['y_true_test']=[]  

sin_div=R_variable['sin_div']
test_inputs=R_variable['test_inputs']
train_inputs=R_variable['train_inputs']


t0=time.time() 


# In[16]:


y_0_test = func0(test_inputs,0)
y_0_train = func0(train_inputs,0)
y_true_test = func_to_approx(test_inputs,R_variable['sin_div'],R_variable['SD'])
y_true_train = func_to_approx(train_inputs,R_variable['sin_div'],R_variable['SD'])

R_variable['y_0_train']=y_0_train
R_variable['y_0_test']=y_0_test
R_variable['y_true_test']= y_true_test
R_variable['y_true_train']=y_true_train

if R_variable['isFFT']:   
    fft_0_test=my_fft(y_0_test,ComputeStepFFT)
    R_variable['fft_0_test']=fft_0_test
    fft_0_train=my_fft(y_0_train,ComputeStepFFT)
    R_variable['fft_0_train']=fft_0_train
    fft_true_test=my_fft(y_true_test,ComputeStepFFT)    
    R_variable['fft_true_test']=fft_true_test
    fft_true_train=my_fft(y_true_train,ComputeStepFFT)
    R_variable['fft_true_train']=fft_true_train

### compute a FFT for FFT Training.
fft_y_true_train=np.fft.fft(np.squeeze(y_true_train))

 
if R_variable['isFFT']:
    Sel_fre_true=np.array(R_variable['fft_true_train'][0:ComputeStepFFT])
    Peak_ind1=SelectPeakIndex(Sel_fre_true, endpoint=False)
    Peak_ind=Peak_ind1
    Hand_Add_peak=[1]  # sometimes, first few points are also very important. such as x^2
    Hand_Add_peak=[] 
    Peak_ind=np.concatenate([Peak_ind,Hand_Add_peak],axis=0)
    Peak_ind=np.sort(Peak_ind)
    Peak_len=len(Peak_ind)
    Peak_ind=np.int32(Peak_ind)
    Peak_id=0  # count peak for peak training

#train_inputs = np.round(train_inputs,3)
#test_inputs = np.round(test_inputs,3)
pos = np.sort(np.unique(np.concatenate((train_inputs,test_inputs),axis=0)))
y = func_to_approx(pos,R_variable['sin_div'],R_variable['SD'])
ypos = func_to_approx(pos,R_variable['sin_div'],R_variable['SD'])
train_mask = np.zeros(np.shape(pos),dtype=bool)
test_mask = np.zeros(np.shape(pos),dtype=bool)
G = nx.Graph()
for s in range(len(pos)):
    if pos[s] in train_inputs:
        train_mask[s] = True
        G.add_node(s, color = 'blue',weight=6)
    else:
        train_mask[s] = False
        G.add_node(s, color = 'red',weight=6)
    if pos[s] in test_inputs:
        test_mask[s] = True
    else:
        test_mask[s] = False
#pos = set(pos)
print(pos)
print(len(pos))
print(sum(test_mask==True))
print(sum(train_mask==True))
import dgl
import torch
#g = dgl.DGLGraph()
nnode = len(pos)
print(nnode)
nnbr = 3
tpl = []
for i in range(nnode):
    #g.add_nodes(i)
    tpl.extend([(i,i - j) for j in range(1,nnbr) if i-j >= 0])
    tpl.extend([(i,i + j) for j in range(1,nnbr) if i+j < nnode])
src, dst = tuple(zip(*tpl))
print(list(src))
print(list(dst))
src = torch.tensor(list(src))
dst = torch.tensor(list(dst))
g = dgl.graph((src,dst),num_nodes=nnode)
#g.add_edges(src, dst)
print(np.shape(np.reshape(pos,(len(pos),1))))
g.ndata['feat'] = torch.tensor(np.reshape(y,(len(pos),1)),dtype=torch.float)
g.ndata['trm'] = torch.tensor(np.reshape(train_mask,(len(pos),1)))
g.ndata['tem'] = torch.tensor(np.reshape(test_mask,(len(pos),1)))
g.ndata['y'] = torch.tensor(np.reshape(ypos,(len(pos),1)),dtype=torch.float)
print(tpl)

print(g)
G.add_edges_from([(s,d) for s,d in tpl])
colors = [node[1]['color'] for node in G.nodes(data=True)]

from dgl.nn import GraphConv
class gcnmodel(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out, dp):
        super(gcnmodel, self).__init__()
        self.conv1 = GraphConv(in_feats,h_feats[0],weight=True,bias=True, allow_zero_in_degree=True)
        self.dropout = torch.nn.Dropout(dp)
        self.conv2 = GraphConv(h_feats[0],h_feats[1],weight=True,bias=True, allow_zero_in_degree=True)
        self.conv3 = GraphConv(h_feats[1],h_feats[2],weight=True,bias=True, allow_zero_in_degree=True)
        self.conv4 = GraphConv(h_feats[2],h_feats[3],weight=True,bias=True, allow_zero_in_degree=True)
        self.conv5 = GraphConv(h_feats[3],out,weight=True,bias=True, allow_zero_in_degree=True)
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = torch.tanh(h)
        #h = self.dropout(h)

        h = self.conv2(g, h)
        h = torch.tanh(h)
        #h = self.dropout(h)

        h = self.conv3(g, h)
        h = torch.tanh(h)
        #h = self.dropout(h)

        h = self.conv4(g, h)
        h = torch.tanh(h)
        #h = self.dropout(h)
                
        h = self.conv5(g, h)
        return h

model = gcnmodel(g.ndata['feat'].shape[1], [200,200,200,100], 1, 0.01)
#model = model.float()
#train(g, model)
'''     
#tf.reset_default_graph() 
with tf.variable_scope('Graph',reuse=tf.AUTO_REUSE) as scope:
        # Our inputs will be a batch of values taken by our functions
        x = tf.placeholder(tf.float32, shape=[None, 1], name="x")
        y_true = tf.placeholder_with_default(input=[[0.0]], shape=[None, 1], name="y")
        
        y,w_Sess,b_Sess,L2w_all_out = univAprox(x, R_variable['hidden_units'],
                                                astddev=R_variable['astddev'],bstddev=R_variable['bstddev'],
                                                ActFuc=R_variable['ActFuc'],seed=R_variable['seed'])
        
        with tf.variable_scope('Loss',reuse=tf.AUTO_REUSE):
            loss=tf.reduce_mean(tf.square(y - y_true))
        
        # We define our train operation using the Adam optimizer
        learning_rate=R_variable['learning_rate']
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = adam.minimize(loss)
        # train_op = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
'''
optimizer = torch.optim.Adam(model.parameters(), lr=R_variable['learning_rate'])
all_logits = []
best_val_acc = 0
best_test_acc = 0

features = g.ndata['feat']
labels = g.ndata['y']
train_mask = g.ndata['trm']
val_mask = g.ndata['tem']
test_mask = g.ndata['tem']

#saver = tf.train.Saver() 
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True  
#sess = tf.Session(config=config)
#if R_variable['iscontinue']:
#    saver.restore(sess, "%smodel/model.ckpt"%(FolderName))
#else:
#    sess.run(tf.global_variables_initializer())
for i in range(R_variable['Total_Step']):
    #indperm = np.random.permutation(R_variable['train_size'])
    #ind = indperm[0:R_variable['batch_size']]
    #print(len(ind))
    logits = model(g, features)

    # Compute prediction
    pred = logits # logits.argmax(1)
    #print(pred)
    # Compute loss
    # Note that we should only compute the losses of the nodes in the training set,
    # i.e. with train_mask 1.
    loss = torch.nn.functional.mse_loss(logits[train_mask], labels[train_mask])
    #print(loss)
    # Compute accuracy on training/validation/test
    #train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
    #val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
    #test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

    # Save the best validation accuracy and the corresponding test accuracy.
    #if best_val_acc < val_acc:
    #    best_val_acc = val_acc
    #    best_test_acc = test_acc

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #all_logits.append(logits.detach())
    #if e % 5 == 0:
    #    print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'.format(
    #        e, loss, val_acc, best_val_acc, test_acc, best_test_acc))
    val_loss = torch.nn.functional.mse_loss(logits[test_mask], labels[test_mask])


    if (i) % R_variable['Record_Step'] == 0:
        #y_test, loss_test_tmp,w_tmp,b_tmp= sess.run([y,loss,w_Sess,b_Sess], 
        #                                               feed_dict={x: test_inputs, y_true: y_true_test})
        #y_train,loss_train_tmp = sess.run([y,loss],feed_dict={x: train_inputs, y_true: y_true_train})
    
        if i==0:
            y_test_ini=pred[test_mask].detach().numpy()
            R_variable['y_test_ini']=y_test_ini
        R_variable['loss_test'].append(val_loss.detach().numpy())
        R_variable['loss_train'].append(loss.detach().numpy())  
        if R_variable['isRec_y_test']:
            R_variable['y_test_all'].append(np.squeeze(pred[test_mask]))
        
        if R_variable['isFFT']:
            R_variable['fft_fit_test'].append(my_fft(pred[test_mask].detach().numpy(),ComputeStepFFT))
            R_variable['fft_fit_train'].append(my_fft(pred[train_mask].detach().numpy(),ComputeStepFFT)) 
        
        
        if loss<R_variable['tol']:
            print('total step:%s; total error:%s'%(i,loss))
            break

    #indperm = np.random.permutation(R_variable['train_size'])
    #ind = indperm[0:R_variable['batch_size']]
    #_ = sess.run(train_op, feed_dict={x: train_inputs[ind], y_true: y_true_train[ind]})
    # Backward
    #optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()
        
   
        
    if (i%250==0 and i<2000000):
        print('batch: %d, test loss: %f' % (i + 1, R_variable['loss_test'][-1]))
        print('batch: %d, train loss: %f' % (i + 1, R_variable['loss_train'][-1]))
        R_variable['y_test']=pred[test_mask].detach().numpy()
        R_variable['y_train']=pred[train_mask].detach().numpy()
        t1=time.time()
        print('time cost:%s'%(t1-t0))
        shutil.rmtree('%smodel/'%(FolderName))
        os.mkdir('%smodel/'%(FolderName))
        #save_path = saver.save(sess, "%smodel/model.ckpt"%(FolderName))
        with open('%s/objs.pkl'%(FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(R_variable, f, protocol=4)
           
        text_file = open("%s/Output.txt"%(FolderName), "w")
        for para in R_variable:
            #print(np.size(R_variable[para]))
            #print(para)
            if np.size(R_variable[para])>20:
                continue
            text_file.write('%s: %s\n'%(para,R_variable[para]))
        
        text_file.close()
        
        plt.ion()
        fig = plt.figure(1,figsize=(15,5)),plt.clf()
        ax=plt.gca() 
        plt.plot(test_inputs, R_variable['y_true_test'],'g', label='Test')
        nx.draw(G,pos=np.concatenate((np.reshape(pos,(len(pos),1)),pred.detach().numpy()),axis=1), node_color=colors,node_size=50,ax=ax)
        plt.draw()
        plt.pause(0.001)
        plt.show(block=False)
        plt.ioff()

        plt.figure() 
        ax=plt.gca() 
        plt.plot(train_inputs, R_variable['y_train'],'m.', label='Train_fit')
        plt.plot(train_inputs, R_variable['y_true_train'],'b-', label='Train_true')
        plt.legend(fontsize=16)
        plt.xlabel('x',fontsize=18)
        plt.ylabel('y',fontsize=18)
        plt.rc('xtick',labelsize=18)
        plt.rc('ytick',labelsize=18)
        plt.title('epoch: %s'%(i),fontsize=18)
        ax.set_position(poS, which='both')
        fntmp = '%sytrain%s'%(R_variable['FolderName'],i)
        mySaveFig(plt,fntmp,ax=ax,iseps=0)
        
        plt.figure() 
        ax=plt.gca() 
        plt.plot(test_inputs, R_variable['y_test'],'r.', label='Test_fit')
        plt.plot(test_inputs, R_variable['y_true_test'],'g-', label='Test_true')
        plt.legend(fontsize=16)
        plt.xlabel('x',fontsize=18)
        plt.ylabel('y',fontsize=18)
        plt.rc('xtick',labelsize=18)
        plt.rc('ytick',labelsize=18)
        plt.title('epoch: %s'%(i),fontsize=18)
        ax.set_position(poS, which='both')
        fntmp = '%sytest%s'%(R_variable['FolderName'],i)
        mySaveFig(plt,fntmp,ax=ax,iseps=0)
        
        plt.figure()
        ax = plt.gca()
        y1 = R_variable['loss_test']
        y2 = R_variable['loss_train']
        plt.plot(y1,'ro',label='Test')
        plt.plot(y2,'g*',label='Train')
        #ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlabel('epoch',fontsize=18)
        plt.ylabel('loss',fontsize=18)
        plt.rc('xtick',labelsize=18)
        plt.rc('ytick',labelsize=18)                
        plt.legend(fontsize=18)
        ax.set_position(poS, which='both')
        fntmp = '%sloss'%(FolderName)
        mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)
        
        if R_variable['isFFT']:
            plt.figure()
            ax=plt.gca()
            y1 = R_variable['fft_true_train'] / R_variable['train_size']
            y2 = R_variable['fft_fit_train'][-1] / R_variable['train_size']
            plt.semilogy(y1+LowFreqDrawAdd,'ro-',label='Trn_true')
            plt.semilogy(y2+LowFreqDrawAdd,'g*-',label='Trn_fit')
            plt.xlabel('freq index',fontsize=18)
            plt.ylabel('|FFT|',fontsize=18)
            plt.rc('xtick',labelsize=18)
            plt.rc('ytick',labelsize=18)                
            plt.legend(fontsize=18)
            plt.title('Train, epoch: %s'%(i),fontsize=18)
            ax.set_position(poS, which='both')
            fntmp = '%strainfft%s'%(FolderName,i)
            mySaveFig(plt, fntmp,ax=ax,iseps=0) 
            
            plt.figure()
            ax=plt.gca()
            y1 = R_variable['fft_true_test'] / R_variable['test_size']
            y2 = R_variable['fft_fit_test'][-1] / R_variable['test_size']
            plt.semilogy(y1+LowFreqDrawAdd,'ro-',label='Trn_true')
            plt.semilogy(y2+LowFreqDrawAdd,'g*-',label='Trn_fit')
            plt.xlabel('freq index',fontsize=18)
            plt.ylabel('|FFT|',fontsize=18)
            plt.rc('xtick',labelsize=18)
            plt.rc('ytick',labelsize=18)                
            plt.legend(fontsize=18)
            plt.title('Test, epoch: %s'%(i),fontsize=18)
            ax.set_position(poS, which='both')
            fntmp = '%stestfft%s'%(FolderName,i)
            mySaveFig(plt, fntmp,ax=ax,iseps=0) 
                    
        
print("for over")    
R_variable['traintime']=time.time()-t0
print(R_variable['traintime'])
#quit()
#save data

with open('%s/objs.pkl'%(FolderName), 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump(R_variable, f, protocol=4)
   
text_file = open("%s/Output.txt"%(FolderName), "w")
for para in R_variable:
    if np.size(R_variable[para])>20:
#        print(para)
        continue
    text_file.write('%s: %s\n'%(para,R_variable[para]))

text_file.close()



#with open('objs.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#    R_variable  = pickle.load(f)
#FolderName='' 


test_inputs=R_variable['test_inputs']
plt.figure() 
ax=plt.gca()
plt.plot(test_inputs, R_variable['y_test_ini'],'g-', label='initial')
plt.legend(fontsize=18)
ax.set_xlabel('x',fontsize=18)
ax.set_ylabel('y',fontsize=18)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
ax.set_position(poS, which='both')
fntmp = '%sy_ini'%(FolderName)
mySaveFig(plt, fntmp,ax=ax)


plt.figure() 
ax=plt.gca() 
plt.plot(train_inputs, pred[train_mask].detach().numpy(),'m.', label='Train_fit')
plt.plot(train_inputs, R_variable['y_true_train'],'b-', label='Train_true')
plt.legend(fontsize=16)
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.title('epoch: %s'%(i),fontsize=18)
ax.set_position(poS, which='both')
fntmp = '%sytrain'%(R_variable['FolderName'])
mySaveFig(plt,fntmp,ax=ax,iseps=0)

plt.figure() 
ax=plt.gca() 
plt.plot(test_inputs, pred[test_mask].detach().numpy(),'r.', label='Test_fit')
plt.plot(test_inputs, R_variable['y_true_test'],'g-', label='Test_true')
plt.legend(fontsize=16)
plt.xlabel('x',fontsize=18)
plt.ylabel('y',fontsize=18)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
plt.title('epoch: %s'%(i),fontsize=18)
ax.set_position(poS, which='both')
fntmp = '%sytest'%(R_variable['FolderName'])
mySaveFig(plt,fntmp,ax=ax,iseps=0)

plt.figure()
ax = plt.gca()
y1 = R_variable['loss_test']
y2 = R_variable['loss_train']
plt.plot(y1,'ro',label='Test')
plt.plot(y2,'g*',label='Train')
#ax.set_xscale('log')
ax.set_yscale('log')
plt.xlabel('epoch',fontsize=18)
plt.ylabel('loss',fontsize=18)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)                
plt.legend(fontsize=18)
ax.set_position(poS, which='both')
fntmp = '%sloss'%(FolderName)
mySaveFig(plt,fntmp,ax=ax,isax=1,iseps=0)


plt.figure() 
ax=plt.gca()
plt.plot(test_inputs, R_variable['y_true_test'],'b-', label='true')
plt.legend(fontsize=16)
ax.set_xlabel('x',fontsize=18)
ax.set_ylabel('y',fontsize=18)
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
ax.set_position(poS, which='both')
fntmp = '%sy_true'%(FolderName)
mySaveFig(plt, fntmp,ax=ax,iseps=0)



if R_variable['isFFT']: 
    UsedPeak=np.arange(len(Peak_ind))
    # plot fft with important peak     
    plt.figure()
    ax=plt.gca()
    ind = np.arange(ComputeStepFFT)
    y1 = R_variable['fft_true_train']/R_variable['train_size']
    plt.semilogy( y1[0:ComputeStepFFT],'r-',label='Trn_true')
    UsedPeaktmp=np.arange(len(Peak_ind))
    plt.semilogy(Peak_ind[UsedPeak],y1[Peak_ind[UsedPeak]]+1e-6,'ks')
    plt.ylim([1e-5,10])
    ax.set_xlabel('freq index',fontsize=18)
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    ax.set_position(poS, which='both')
    fntmp = '%sfftPeak'%(FolderName)
    mySaveFig(plt, fntmp,ax=ax,iseps=0)
    

    d_step_fft_true_train_all=np.zeros([ComputeStepFFT,len(R_variable['fft_fit_train'])])
    abs_err_all=np.zeros([ComputeStepFFT,len(R_variable['fft_fit_train'])])
    fft_train_fitAll=np.asarray(np.abs(R_variable['fft_fit_train']))
    for itfes in range(ComputeStepFFT):
        tmp1=fft_train_fitAll[:,itfes]
        tmp2=R_variable['fft_true_train'][itfes]
        d_step_fft_true_train_all[itfes,:] = (np.absolute(tmp1-tmp2))/(1e-5+tmp2)
        abs_err_all[itfes,:] = np.absolute(tmp1-tmp2)

    DrawDis=1
    DrawLastStep=len(R_variable['fft_fit_train'])
    z_min=0.1
    z_max=1  
    plt.figure()
    ax=plt.gca()
    im=plt.pcolor(d_step_fft_true_train_all[Peak_ind[UsedPeak],:DrawLastStep:DrawDis], cmap='RdBu', vmin=z_min, vmax=z_max)
    ax.set_xlabel('epoch',fontsize=18)
    ax.set_ylabel('freq peak index',fontsize=18)
    #ax.set_xscale('log')
    #ax.set_xlim([1,10])
    plt.rc('xtick',labelsize=18)
    plt.rc('ytick',labelsize=18)
    ax.set_yticks([0,1,2]) 
    ax.set_position(poS, which='both')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax,ticks=[0,0.2,0.4,0.6,0.8,1])
    fntmp = '%speakerror'%(FolderName)
    mySaveFig(plt, fntmp,ax=ax,iseps=0)


