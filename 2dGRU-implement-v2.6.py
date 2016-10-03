# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:13:34 2016

@author: Yang
这一版本将求导功能写成函数
"""
import random
import numpy as np
import math

def sigmoid(x):
    return 1/(1+np.exp(-x))

#calculate reset gate from different directions
def cal_r(parameter,i,j,w_r,b_r):
    h=parameter['h']
    s=parameter['s']
    if i==0 and j==0:
        h_l=np.zeros_like(h[0][0])
        h_d=np.zeros_like(h[0][0])
        h_t=np.zeros_like(h[0][0])
    elif i==0:
        h_l=np.zeros_like(h[0][0])
        h_d=np.zeros_like(h[0][0])
        h_t=h[i][j-1]
    elif j==0:
        h_l=h[i-1][j]
        h_d=np.zeros_like(h[0][0])
        h_t=np.zeros_like(h[0][0])
    else:
        h_t=h[i][j-1]
        h_l=h[i-1][j]
        h_d=h[i-1][j-1]
    q=np.concatenate((h_l,h_t,h_d,s[i][j]),0)
    r=sigmoid(w_r.dot(q)+b_r)
    return r

#concatenate different r from direction
def cal_r_con(parameter,i,j):
    r_l=cal_r(parameter,i,j,parameter['w_r_l'],parameter['b_r_l'])
    r_t=cal_r(parameter,i,j,parameter['w_r_t'],parameter['b_r_t'])
    r_d=cal_r(parameter,i,j,parameter['w_r_d'],parameter['b_r_d'])
    return np.concatenate((r_l,r_t,r_d))
    
#compute 'inner' update gate
def cal_z_pie(parameter,i,j,w_z,b_z):
    h=parameter['h']
    s=parameter['s']
    if i==0 and j==0:
        h_l=np.zeros_like(h[0][0])
        h_d=np.zeros_like(h[0][0])
        h_t=np.zeros_like(h[0][0])
    elif i==0:
        h_l=np.zeros_like(h[0][0])
        h_d=np.zeros_like(h[0][0])
        h_t=h[i][j-1]
    elif j==0:
        h_l=h[i-1][j]
        h_d=np.zeros_like(h[0][0])
        h_t=np.zeros_like(h[0][0])
    else:
        h_t=h[i][j-1]
        h_l=h[i-1][j]
        h_d=h[i-1][j-1]
    q=np.concatenate((h_l,h_t,h_d,s[i][j]),0)
    z=w_z.dot(q)+b_z
    return z

def softmax(K):
    summation=sum(np.exp(K))
    for i in range(np.shape(K)[0]):
        K[i]=np.exp(K[i])/summation
    return K

#compute updata from 4 direction
def cal_z(parameter,i,j,direction):
    if direction=='l':
        z=cal_z_pie(parameter,i,j,parameter['w_z_l'],parameter['b_z_l'])
    elif direction=='t':
        z=cal_z_pie(parameter,i,j,parameter['w_z_t'],parameter['b_z_t'])
    elif direction=='d':
        z=cal_z_pie(parameter,i,j,parameter['w_z_d'],parameter['b_z_d'])
    elif direction=='i':
        z=cal_z_pie(parameter,i,j,parameter['w_z_i'],parameter['b_z_i'])
    return softmax(z)
    
#def cal_z(parameter,i,j,w_z,b_z):
#    z=cal_z_pie(parameter,i,j,w_z,b_z)
#    return softmax(z)
# 
#compute hidden layer outout h
def cal_h(parameter,i,j):
    w=parameter['w']
    U=parameter['U']
    s=parameter['s']
    h=parameter['h'] 
    b=parameter['b']
#    print (U[0][0])
#    r_l=cal_r(parameter,i,j,parameter['w_r_l'],parameter['b_r_l'])
#    r_t=cal_r(parameter,i,j,parameter['w_r_t'],parameter['b_r_t'])
#    r_d=cal_r(parameter,i,j,parameter['w_r_d'],parameter['b_r_d'])
#    z_l=cal_z(parameter,i,j,parameter['w_z_l'],parameter['b_z_l'])
#    z_t=cal_z(parameter,i,j,parameter['w_z_t'],parameter['b_z_t'])
#    z_d=cal_z(parameter,i,j,parameter['w_z_d'],parameter['b_z_d'])
#    z_i=cal_z(parameter,i,j,parameter['w_z_i'],parameter['b_z_i'])
    z_l=cal_z(parameter,i,j,'l')
    z_t=cal_z(parameter,i,j,'t')
    z_d=cal_z(parameter,i,j,'d')
    z_i=cal_z(parameter,i,j,'i')
#    r=np.concatenate((r_l,r_t,r_d))  
    r=cal_r_con(parameter,i,j)
    if i==0 and j==0:
        a_h_pie=w.dot(s[i][j])+b
        h_pie=np.tanh(a_h_pie)
        h_ij=z_i*h_pie
    elif i>0 and j>0:
        h_con=np.concatenate((h[i-1][j],h[i][j-1],h[i-1][j-1]))
        tmp=r*h_con
        a_h_pie=w.dot(s[i][j])+U.dot(tmp)+b
        h_pie=np.tanh(a_h_pie)
        h_ij=z_i*h_pie+z_l*h[i-1][j]+z_t*h[i][j-1]+z_d*h[i-1][j-1]
    elif i==0:
        h_con=np.concatenate((np.zeros_like(h[i][j-1]),h[i][j-1],np.zeros_like(h[i][j-1])))
        tmp=r*h_con
        a_h_pie=w.dot(s[i][j])+U.dot(tmp)+b
        h_pie=np.tanh(a_h_pie)
        h_ij=z_i*h_pie+z_t*h[i][j-1]
        
    elif j==0:
        h_con=np.concatenate((h[i-1][j],np.zeros_like(h[i-1][j]),np.zeros_like(h[i-1][j])))
        tmp=r*h_con
        a_h_pie=w.dot(s[i][j])+U.dot(tmp)+b
        h_pie=np.tanh(a_h_pie)
        h_ij=z_i*h_pie+z_l*h[i-1][j]
        
    return h_ij,h_pie




#后向传播计算导数 
'''checked'''     
def delta_z_pie(parameter,i,j,direction):
    z=cal_z(parameter,i,j,direction)
    if direction=='l':
#        return parameter['h'][i-1][j]*z_l    
        delta_z=parameter['e'][i][j]*(parameter['h'][i-1][j])
        delta_z_pie_value=np.zeros_like(delta_z)
        delta_z_pie_value=delta_z*z*(1-z)                
                
        return delta_z_pie_value
    elif direction=='t':
        delta_z=parameter['e'][i][j]*(parameter['h'][i][j-1])
        delta_z_pie_value=np.zeros_like(delta_z)
        delta_z_pie_value=delta_z*z*(1-z)
        return delta_z_pie_value
        
    elif direction=='d':
        delta_z=parameter['e'][i][j]*(parameter['h'][i-1][j-1])
        delta_z_pie_value=np.zeros_like(delta_z)
        delta_z_pie_value=delta_z*z*(1-z)
        return delta_z_pie_value
        
    elif direction=='i':
        delta_z=parameter['e'][i][j]*(parameter['h_pie'][i][j])
        delta_z_pie_value=np.zeros_like(delta_z)
        delta_z_pie_value=delta_z*z*(1-z) 
        return delta_z_pie_value
#        return parameter['h'][i-1][j-1].dot(parameter['e'][i][j].T)

#eq (1)
'''checked'''
def delta_pie(parameter,i,j):
#    if direction=='l':
##        z_l=cal_z(parameter,i,j,parameter['w_z_l'],parameter['b_z_l'])
#        z_l=cal_z(parameter,i,j,'l')
#        delta_pie=parameter['e'][i][j].dot((z_l*(1-parameter['a_h_pie'][i][j]*parameter['a_h_pie'][i][j])).T)
#    elif direction=='t':
##        z_t=cal_z(parameter,i,j,parameter['w_z_t'],parameter['b_z_t'])
#        z_t=cal_z(parameter,i,j,'t')
#        delta_pie=parameter['e'][i][j].dot((z_t*(1-parameter['a_h_pie'][i][j]*parameter['a_h_pie'][i][j])).T)
#    elif direction=='d':
##        z_d=cal_z(parameter,i,j,parameter['w_z_d'],parameter['b_z_d'])
#        z_d=cal_z(parameter,i,j,'d')
#        delta_pie=parameter['e'][i][j].dot((z_d*(1-parameter['a_h_pie'][i][j]*parameter['a_h_pie'][i][j])).T)

    z_d=cal_z(parameter,i,j,'i')
#    a,hpie=cal_h(parameter,i,j)
#    delta_pie_val=parameter['e'][i][j]*z_d*(1-hpie*hpie)
    delta_pie_val=parameter['e'][i][j]*z_d*(1-parameter['h_pie'][i][j]*parameter['h_pie'][i][j])
    return delta_pie_val

'''checked'''
def delta_r(parameter,i,j,direction):
    '''chongfu'''
    #计算delta_pie,r
    s=parameter['s']
    U=parameter['U']
    h=parameter['h']
    if i==0 and j==0:
        h_l=np.zeros_like(h[0][0])
        h_d=np.zeros_like(h[0][0])
        h_t=np.zeros_like(h[0][0])
    elif i==0:
        h_l=np.zeros_like(h[0][0])
        h_d=np.zeros_like(h[0][0])
        h_t=h[i][j-1]
    elif j==0:
        h_l=h[i-1][j]
        h_d=np.zeros_like(h[0][0])
        h_t=np.zeros_like(h[0][0])
    else:
        h_t=h[i][j-1]
        h_l=h[i-1][j]
        h_d=h[i-1][j-1]
        
        
    q=np.concatenate((h_l,h_t,h_d,s[i][j]),0)
    
    rh=np.concatenate((h_l,h_t,h_d),0)
    
    
    length=len(h_l)
    delta_pie_val=delta_pie(parameter,i,j)
    
    
    epsilon=(delta_pie_val.T.dot(U)).T*rh
    #计算a_r eq(4),并构造和r相乘的矩阵rh
    if direction=='l':
        epsilon_r=epsilon[:length]
        a_r=parameter['w_r_l'].dot(q)+parameter['b_r_l']
    elif direction=='t':
        epsilon_r=epsilon[length:2*length]
        a_r=parameter['w_r_t'].dot(q)+parameter['b_r_t']
    elif direction=='d':
        epsilon_r=epsilon[2*length:3*length]
        a_r=parameter['w_r_d'].dot(q)+parameter['b_r_d']
        
        
    #计算delta_pie
    r=np.tanh(a_r)
    
#    return a_r
    return epsilon_r*(r*(1-r))
#print (delta_r(parameter,1,1,'l'))
def epsilon(parameter,i,j,m,n):
    length=len(parameter['h'][0][0])
    w_rl=parameter['w_r_l'];w_rt=parameter['w_r_t'];w_rd=parameter['w_r_d']
    w_zl=parameter['w_z_l'];w_zt=parameter['w_z_t'];w_zd=parameter['w_z_d']
    w_zi=parameter['w_z_i']
    if i==m-1 and j==n-1:
        return -2*(1-parameter['w_s'].dot(parameter['h'][m-1][n-1])-parameter['b_s'])*parameter['w_s'].T
    elif i==m-1:

        e=parameter['e'];U=parameter['U']
   
        
        firstline=cal_z(parameter,i,j+1,'t')*e[i][j+1]
        r_line_2=delta_r(parameter,i,j+1,'l').T.dot(w_rl).T+delta_r(parameter,i,j+1,'t').T.dot(w_rt).T+delta_r(parameter,i,j+1,'d').T.dot(w_rd).T
        
        z_line_2=delta_z_pie(parameter,i,j+1,'l').T.dot(w_zl).T+delta_z_pie(parameter,i,j+1,'t').T.dot(w_zt).T+delta_z_pie(parameter,i,j+1,'d').T.dot(w_zd).T+delta_z_pie(parameter,i,j+1,'i').T.dot(w_zi).T
    
        lastline=delta_pie(parameter,i,j+1).T.dot(U).T*cal_r_con(parameter,i,j+1)
        
        return lastline[length:2*length]+r_line_2[length:2*length]+z_line_2[length:2*length]+firstline
    elif j==n-1:

        e=parameter['e'];U=parameter['U']

        
        firstline=cal_z(parameter,i+1,j,'l')*e[i+1][j]
        r_line_1=delta_r(parameter,i+1,j,'l').T.dot(w_rl).T+delta_r(parameter,i+1,j,'t').T.dot(w_rt).T+delta_r(parameter,i+1,j,'d').T.dot(w_rd).T
        
        z_line_1=delta_z_pie(parameter,i+1,j,'l').T.dot(w_zl).T+delta_z_pie(parameter,i+1,j,'t').T.dot(w_zt).T+delta_z_pie(parameter,i+1,j,'d').T.dot(w_zd).T+delta_z_pie(parameter,i+1,j,'i').T.dot(w_zi).T
    
        lastline=delta_pie(parameter,i+1,j).T.dot(U).T*cal_r_con(parameter,i+1,j)
        return lastline[:length]+r_line_1[:length]+z_line_1[:length]+firstline
    else:

        e=parameter['e'];U=parameter['U']
 
        
        firstline=cal_z(parameter,i+1,j,'l')*e[i+1][j]+cal_z(parameter,i,j+1,'t')*e[i][j+1]+cal_z(parameter,i+1,j+1,'d')*e[i+1][j+1]
        r_line_1=delta_r(parameter,i+1,j,'l').T.dot(w_rl).T+delta_r(parameter,i+1,j,'t').T.dot(w_rt).T+delta_r(parameter,i+1,j,'d').T.dot(w_rd).T
        r_line_2=delta_r(parameter,i,j+1,'l').T.dot(w_rl).T+delta_r(parameter,i,j+1,'t').T.dot(w_rt).T+delta_r(parameter,i,j+1,'d').T.dot(w_rd).T
        r_line_3=delta_r(parameter,i+1,j+1,'l').T.dot(w_rl).T+delta_r(parameter,i+1,j+1,'t').T.dot(w_rt).T+delta_r(parameter,i+1,j+1,'d').T.dot(w_rd).T
        
        z_line_1=delta_z_pie(parameter,i+1,j,'l').T.dot(w_zl).T+delta_z_pie(parameter,i+1,j,'t').T.dot(w_zt).T+delta_z_pie(parameter,i+1,j,'d').T.dot(w_zd).T+delta_z_pie(parameter,i+1,j,'i').T.dot(w_zi).T
        z_line_2=delta_z_pie(parameter,i,j+1,'l').T.dot(w_zl).T+delta_z_pie(parameter,i,j+1,'t').T.dot(w_zt).T+delta_z_pie(parameter,i,j+1,'d').T.dot(w_zd).T+delta_z_pie(parameter,i,j+1,'i').T.dot(w_zi).T
        z_line_3=delta_z_pie(parameter,i+1,j+1,'l').T.dot(w_zl).T+delta_z_pie(parameter,i+1,j+1,'t').T.dot(w_zt).T+delta_z_pie(parameter,i+1,j+1,'d').T.dot(w_zd).T+delta_z_pie(parameter,i+1,j+1,'i').T.dot(w_zi).T
    
        lastline=(delta_pie(parameter,i+1,j).T.dot(U).T*cal_r_con(parameter,i+1,j))[:length]+(delta_pie(parameter,i,j+1).T.dot(U).T*cal_r_con(parameter,i,j+1))[length:2*length]+(delta_pie(parameter,i+1,j+1).T.dot(U).T*cal_r_con(parameter,i+1,j+1))[2*length:3*length]
        
        return lastline+r_line_1[:length]+r_line_2[length:2*length]+r_line_3[2*length:3*length]+z_line_1[:length]+z_line_2[length:2*length]+z_line_3[2*length:3*length]+firstline

def out(parameter):
    return parameter['w_s'].dot(parameter['h'][-1][-1])+parameter['b_s']
def testout(h,w_s,b_s):
    return (1-w_s.dot(h[-1][-1])-b_s)*(1-w_s.dot(h[-1][-1])-b_s)
#delta_r(parameter,i,j,direction)
#def delta_r(parameter,i,j,direction)
#def delta_z_pie(parameter,i,j,direction)
#def cal_z(parameter,i,j,w_z,b_z)
#def cal_h(parameter,i,j)
#def cal_r(parameter,i,j,w_r,b_r)
#def cal_z_pie(parameter,i,j,w_z,b_z)
#def delta_pie(parameter,i,j,direction)
#前向计算一遍h
#initialize parameter'
#m=6;n=7
    
    
    
d_h=5;d_s=1
np.random.seed(0)

with open(u'testData.txt','r') as f:
    for line in f:
        line=line.strip()
        tmp=line.split('\t')
        break
container=[]
for i in tmp:
    container.append(i.split(' '))
worddic={}
with open(u'1000词向量.txt','r') as f:
    for line in f:
        line=line.strip()
        tmp=line.split(' ')
        for i in range(1,len(tmp)):
            tmp[i]=float(tmp[i])
        worddic[tmp[0].lower()]=tmp[1:]
        
vectors=[]
for i in container:
    vector=[]
    for word in i:
        if word.lower() not in worddic:
            vector.append([0]*len(worddic['i']))
        else:           
            vector.append(worddic[word.lower()])
    vector=np.array(vector)
    vectors.append(vector)
m=np.shape(vectors[0])[0]
n=np.shape(vectors[1])[0]
parameter=dict()
parameter['s']=np.zeros((m,n,d_s,1))

for i in range(len(vectors[0])):
    for j in range(len(vectors[1])):
        
        tmp=sum(vectors[0][i]*vectors[1][j])/(np.sqrt(vectors[0][i].dot(vectors[0][i]))*np.sqrt(vectors[1][j].dot(vectors[1][j])))
        if math.isnan(tmp):
            tmp=0
        parameter['s'][i][j][0][0]=tmp

#x1=np.array([[1,2],[3,4],[5,6]])
#x2=np.array([[1,2],[3,4],[5,6]])
#
#y=sum(x1*x2)/(np.sqrt(x1.dot(x1))*np.sqrt(x2.dot(x2)))






#s=np.random.random((m,n,d_s,1))



# use dictionary to store parameter
con=100
parameter['w_r_l']=np.random.random((d_h,d_s+3*d_h))/con
parameter['w_r_t']=np.random.random((d_h,d_s+3*d_h))/con
parameter['w_r_d']=np.random.random((d_h,d_s+3*d_h))/con
parameter['w_z_l']=np.random.random((d_h,d_s+3*d_h))/con
parameter['w_z_t']=np.random.random((d_h,d_s+3*d_h))/con
parameter['w_z_d']=np.random.random((d_h,d_s+3*d_h))/con
parameter['w_z_i']=np.random.random((d_h,d_s+3*d_h))/con
parameter['b_r_l']=np.random.random((d_h,1))/con
parameter['b_r_t']=np.random.random((d_h,1))/con
parameter['b_r_d']=np.random.random((d_h,1))/con
parameter['b_z_l']=np.random.random((d_h,1))/con
parameter['b_z_t']=np.random.random((d_h,1))/con
parameter['b_z_d']=np.random.random((d_h,1))/con
parameter['b_z_i']=np.random.random((d_h,1))/con

parameter['w']=np.random.random((d_h,d_s))/con
parameter['U']=np.random.random((d_h,3*d_h))/con
parameter['b']=np.random.random((d_h,1))/con
parameter['w_s']=np.random.random((1,d_h))
parameter['b_s']=np.random.rand(1)

parameter['h']=np.zeros((m,n,d_h,1))
parameter['e']=np.zeros((m,n,d_h,1))

parameter['h_pie']=np.zeros((m,n,d_h,1))

parameter['dw_r_l']=np.zeros((m,n,d_h,d_s+3*d_h))
parameter['dw_r_t']=np.zeros((m,n,d_h,d_s+3*d_h))
parameter['dw_r_d']=np.zeros((m,n,d_h,d_s+3*d_h))
parameter['dw_z_l']=np.zeros((m,n,d_h,d_s+3*d_h))
parameter['dw_z_t']=np.zeros((m,n,d_h,d_s+3*d_h))
parameter['dw_z_d']=np.zeros((m,n,d_h,d_s+3*d_h))
parameter['dw_z_i']=np.zeros((m,n,d_h,d_s+3*d_h))
parameter['db_r_l']=np.zeros((m,n,d_h,1))
parameter['db_r_t']=np.zeros((m,n,d_h,1))
parameter['db_r_d']=np.zeros((m,n,d_h,1))
parameter['db_z_l']=np.zeros((m,n,d_h,1))
parameter['db_z_t']=np.zeros((m,n,d_h,1))
parameter['db_z_d']=np.zeros((m,n,d_h,1))
parameter['db_z_i']=np.zeros((m,n,d_h,1))

parameter['dw']=np.zeros((m,n,d_h,d_s))
parameter['dU']=np.zeros((m,n,d_h,3*d_h))
parameter['db']=np.zeros((m,n,d_h,1))
parameter['dw_s']=np.zeros((m,n,1,d_h))
parameter['db_s']=np.zeros((m,n,1))  
    
    
def forward(m,n,parameter,d_h):    
    for i in range(m):
        for j in range(n):
            parameter['h'][i][j],parameter['h_pie'][i][j]=cal_h(parameter,i,j)
    return parameter['h'] ,parameter['h_pie']
def backword(m,n,parameter): 
    for i in range(m-1,-1,-1):
        for j in range(n-1,-1,-1):
            parameter['e'][i][j]=epsilon(parameter,i,j,m,n)
    
            
            if i==0 and j==0:
                h_l=np.zeros_like(parameter['h'][0][0])
                h_d=np.zeros_like(parameter['h'][0][0])
                h_t=np.zeros_like(parameter['h'][0][0])
            elif i==0:
                h_l=np.zeros_like(parameter['h'][0][0])
                h_d=np.zeros_like(parameter['h'][0][0])
                h_t=parameter['h'][i][j-1]
            elif j==0:
                h_l=parameter['h'][i-1][j]
                h_d=np.zeros_like(parameter['h'][0][0])
                h_t=np.zeros_like(parameter['h'][0][0])
            else:
                h_t=parameter['h'][i][j-1]
                h_l=parameter['h'][i-1][j]
                h_d=parameter['h'][i-1][j-1]
            q=np.concatenate((h_l,h_t,h_d,parameter['s'][i][j]),0)
            qr=np.concatenate((h_l,h_t,h_d),0)
            
            parameter['dw_r_l'][i][j]=delta_r(parameter,i,j,'l').dot(q.T)
            parameter['dw_r_t'][i][j]=delta_r(parameter,i,j,'t').dot(q.T)
            parameter['dw_r_d'][i][j]=delta_r(parameter,i,j,'d').dot(q.T)
            
            parameter['dw_z_l'][i][j]=delta_z_pie(parameter,i,j,'l').dot(q.T)
            parameter['dw_z_t'][i][j]=delta_z_pie(parameter,i,j,'t').dot(q.T)
            parameter['dw_z_d'][i][j]=delta_z_pie(parameter,i,j,'d').dot(q.T)
            parameter['dw_z_i'][i][j]=delta_z_pie(parameter,i,j,'i').dot(q.T)
            
            parameter['db_r_l'][i][j]=delta_r(parameter,i,j,'l')
            parameter['db_r_t'][i][j]=delta_r(parameter,i,j,'t')
            parameter['db_r_d'][i][j]=delta_r(parameter,i,j,'d')
            
            parameter['db_z_l'][i][j]=delta_z_pie(parameter,i,j,'l')
            parameter['db_z_t'][i][j]=delta_z_pie(parameter,i,j,'t')
            parameter['db_z_d'][i][j]=delta_z_pie(parameter,i,j,'d')
            parameter['db_z_i'][i][j]=delta_z_pie(parameter,i,j,'i')
    ##        
            parameter['dU'][i][j]=delta_pie(parameter,i,j).dot((cal_r_con(parameter,i,j)*qr).T)  
            parameter['dw'][i][j]=delta_pie(parameter,i,j).dot(parameter['s'][i][j].T)
            parameter['db'][i][j]=delta_pie(parameter,i,j)
    return parameter
    



def gradcheck(para,parameter):       
    left=(sum(sum(parameter['d'+para])))[-1][-1]
    temp=parameter[para]
    parameter['h']=np.zeros((m,n,d_h,1))
    parameter[para][-1][-1]+=1e-6
    parameter['h'],parameter['h_pie']=forward(m,n,parameter,d_h)   
    
    Jplus=testout(parameter['h'],parameter['w_s'],parameter['b_s'])   
    parameter[para][-1][-1]=temp[-1][-1]-1e-6
    
    parameter['h']=np.zeros((m,n,d_h,1))
    parameter['h_pie']=np.zeros((m,n,d_h,1))
    parameter['h'],parameter['h_pie']=forward(m,n,parameter,d_h)  
    
    Jminus=testout(parameter['h'],parameter['w_s'],parameter['b_s']) 
    print('our calculation of the derivative of the function',left)
    print('The derivative of the function calculate by the defination',((Jplus-Jminus)/(2e-6))[0][0]) 

parameter['h'],parameter['h_pie']=forward(m,n,parameter,d_h)    
parameter=backword(m,n,parameter)      
gradcheck('w',parameter)
