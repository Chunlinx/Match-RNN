# -*- coding: utf-8 -*-
"""

@author: Yang
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
        h_con=np.concatenate((h[i][j-1],h[i-1][j],h[i-1][j-1]))
        tmp=r*h_con
        a_h_pie=w.dot(s[i][j])+U.dot(tmp)+b
        h_pie=np.tanh(a_h_pie)
        h_ij=z_i*h_pie+z_l*h[i-1][j]+z_t*h[i][j-1]+z_d*h[i-1][j-1]
    elif i==0:
        h_con=np.concatenate((h[i][j-1],np.zeros_like(h[i][j-1]),np.zeros_like(h[i][j-1])))
        tmp=r*h_con
        a_h_pie=w.dot(s[i][j])+U.dot(tmp)+b
        h_pie=np.tanh(a_h_pie)
        h_ij=z_i*h_pie+z_t*h[i][j-1]
        
    elif j==0:
        h_con=np.concatenate((np.zeros_like(h[i-1][j]),h[i-1][j],np.zeros_like(h[i-1][j])))
        tmp=r*h_con
        a_h_pie=w.dot(s[i][j])+U.dot(tmp)+b
        h_pie=np.tanh(a_h_pie)
        h_ij=z_i*h_pie+z_l*h[i-1][j]
        
    return h_ij,h_pie




#后向传播计算导数      
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
    delta_pie=parameter['e'][i][j]*z_d*(1-parameter['h_pie'][i][j]*parameter['h_pie'][i][j])
    return delta_pie


def delta_r(parameter,i,j,direction):
    #计算delta_pie,r
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
    
    #计算a_r eq(4),并构造和r相乘的矩阵rh
    if direction=='l':
        
        rh_l=h_l
        rh_t=np.zeros_like(h_l)
        rh_d=np.zeros_like(h_l)       
        rh=np.concatenate((rh_l,rh_t,rh_d),0)
        
        a_r=parameter['w_r_l'].dot(q)+parameter['b_r_l']
    elif direction=='t':
        
        rh_t=h_t
        rh_l=np.zeros_like(h_l) 
        rh_d=np.zeros_like(h_l)     
        rh=np.concatenate((rh_l,rh_t,rh_d),0) 
        
        a_r=parameter['w_r_t'].dot(q)+parameter['b_r_t']
    elif direction=='d':
        
        rh_d=h_d      
        rh_t=np.zeros_like(h_l) 
        rh_l=np.zeros_like(h_l) 
        rh=np.concatenate((rh_l,rh_t,rh_d),0)
        
        a_r=parameter['w_r_d'].dot(q)+parameter['b_r_d']
    #计算delta_pie
    r=np.tanh(a_r)
    delta_pie_val=delta_pie(parameter,i,j)
#    return a_r
    return delta_pie_val*(U.dot(rh))*(r*(1-r))
#print (delta_r(parameter,1,1,'l'))
def epsilon(parameter,i,j,m,n):
    if i==m-1 and j==n-1:
        return -2*(1-parameter['w_s'].dot(parameter['h'][m-1][n-1])-parameter['b_s'])*parameter['w_s'].T
    elif i==m-1:
        w_rt=parameter['w_r_t']
        w_zt=parameter['w_z_t']
        w_zi=parameter['w_z_i']
        e=parameter['e'];s=parameter['s'];U=parameter['U']
        
        e_qt=np.concatenate((np.zeros_like(e[i][j]),np.ones_like(e[i][j]),np.zeros_like(e[i][j]),np.zeros_like(s[i][j])),0)
        e_t=np.concatenate((np.zeros_like(e[i][j]),np.ones_like(e[i][j]),np.zeros_like(e[i][j])),0)
        
        cal_r_con(parameter,i,j)    
        
        firstline=cal_z(parameter,i,j,'t')*e[i][j+1]
        r_line_2=delta_r(parameter,i,j+1,'t')*(w_rt.dot(e_qt))
        
        z_line_2=delta_z_pie(parameter,i,j+1,'t')*(w_zt.dot(e_qt))+delta_z_pie(parameter,i,j+1,'i')*(w_zi.dot(e_qt))
    
        lastline=delta_pie(parameter,i,j+1)*(U.dot(cal_r_con(parameter,i,j+1)*e_t))
        
        return lastline+r_line_2+z_line_2+firstline
    elif j==n-1:
        w_rl=parameter['w_r_l']
        w_zl=parameter['w_z_l']
        w_zi=parameter['w_z_i']
        e=parameter['e'];s=parameter['s'];U=parameter['U']
        e_ql=np.concatenate((np.ones_like(e[i][j]),np.zeros_like(e[i][j]),np.zeros_like(e[i][j]),np.zeros_like(s[i][j])),0)
        e_l=np.concatenate((np.ones_like(e[i][j]),np.zeros_like(e[i][j]),np.zeros_like(e[i][j])),0)
        
        cal_r_con(parameter,i,j)
        
        firstline=cal_z(parameter,i,j,'l')*e[i+1][j]
        r_line_1=delta_r(parameter,i+1,j,'l')*(w_rl.dot(e_ql))
        
        z_line_1=delta_z_pie(parameter,i+1,j,'l')*(w_zl.dot(e_ql))+delta_z_pie(parameter,i+1,j,'i')*(w_zi.dot(e_ql))
    
        lastline=delta_pie(parameter,i+1,j)*(U.dot(cal_r_con(parameter,i+1,j)*e_l))
        return lastline+r_line_1+z_line_1+firstline
    else:
        w_rl=parameter['w_r_l'];w_rt=parameter['w_r_t'];w_rd=parameter['w_r_d']
        w_zl=parameter['w_z_l'];w_zt=parameter['w_z_t'];w_zd=parameter['w_z_d']
        w_zi=parameter['w_z_i']
        e=parameter['e'];s=parameter['s'];U=parameter['U']
        e_ql=np.concatenate((np.ones_like(e[i][j]),np.zeros_like(e[i][j]),np.zeros_like(e[i][j]),np.zeros_like(s[i][j])),0)
        e_qt=np.concatenate((np.zeros_like(e[i][j]),np.ones_like(e[i][j]),np.zeros_like(e[i][j]),np.zeros_like(s[i][j])),0)
        e_qd=np.concatenate((np.zeros_like(e[i][j]),np.zeros_like(e[i][j]),np.ones_like(e[i][j]),np.zeros_like(s[i][j])),0)
        e_l=np.concatenate((np.ones_like(e[i][j]),np.zeros_like(e[i][j]),np.zeros_like(e[i][j])),0)
        e_t=np.concatenate((np.zeros_like(e[i][j]),np.ones_like(e[i][j]),np.zeros_like(e[i][j])),0)
        e_d=np.concatenate((np.zeros_like(e[i][j]),np.zeros_like(e[i][j]),np.ones_like(e[i][j])),0)
        
        cal_r_con(parameter,i,j)    
        
        firstline=cal_z(parameter,i,j,'l')*e[i+1][j]+cal_z(parameter,i,j,'t')*e[i][j+1]+cal_z(parameter,i,j,'d')*e[i+1][j+1]
        r_line_1=delta_r(parameter,i+1,j,'l')*(w_rl.dot(e_ql))+delta_r(parameter,i+1,j,'t')*(w_rt.dot(e_ql))+delta_r(parameter,i+1,j,'d')*(w_rd.dot(e_ql))
        r_line_2=delta_r(parameter,i,j+1,'l')*(w_rl.dot(e_qt))+delta_r(parameter,i,j+1,'t')*(w_rt.dot(e_qt))+delta_r(parameter,i,j+1,'d')*(w_rd.dot(e_qt))
        r_line_3=delta_r(parameter,i+1,j+1,'l')*(w_rl.dot(e_qd))+delta_r(parameter,i+1,j+1,'t')*(w_rt.dot(e_qd))+delta_r(parameter,i+1,j+1,'d')*(w_rd.dot(e_qd))
        
        z_line_1=delta_z_pie(parameter,i+1,j,'l')*(w_zl.dot(e_ql))+delta_z_pie(parameter,i+1,j,'t')*(w_zt.dot(e_ql))+delta_z_pie(parameter,i+1,j,'d')*(w_zd.dot(e_ql))+delta_z_pie(parameter,i+1,j,'i')*(w_zi.dot(e_ql))
        z_line_2=delta_z_pie(parameter,i,j+1,'l')*(w_zl.dot(e_qt))+delta_z_pie(parameter,i,j+1,'t')*(w_zt.dot(e_qt))+delta_z_pie(parameter,i,j+1,'d')*(w_zd.dot(e_qt))+delta_z_pie(parameter,i,j+1,'i')*(w_zi.dot(e_qt))
        z_line_3=delta_z_pie(parameter,i+1,j+1,'l')*(w_zl.dot(e_qd))+delta_z_pie(parameter,i+1,j+1,'t')*(w_zt.dot(e_qd))+delta_z_pie(parameter,i+1,j+1,'d')*(w_zd.dot(e_qd))+delta_z_pie(parameter,i+1,j+1,'i')*(w_zi.dot(e_qd))
    
        lastline=delta_pie(parameter,i+1,j)*(U.dot(cal_r_con(parameter,i+1,j)*e_l))+delta_pie(parameter,i,j+1)*(U.dot(cal_r_con(parameter,i,j+1)*e_t))+delta_pie(parameter,i+1,j+1)*(U.dot(cal_r_con(parameter,i+1,j+1)*e_d))
        return lastline+r_line_1+r_line_2+r_line_3+z_line_1+z_line_2+z_line_3+firstline

def out(parameter):
    return parameter['w_s'].dot(parameter['h'][-1][-1])+parameter['b_s']
def testout(h,w_s,b_s):
    return w_s.dot(h[-1][-1])+b_s
#delta_r(parameter,i,j,direction)
#def delta_r(parameter,i,j,direction)
#def delta_z_pie(parameter,i,j,direction)
#def cal_z(parameter,i,j,w_z,b_z)
#def cal_h(parameter,i,j)
#def cal_r(parameter,i,j,w_r,b_r)
#def cal_z_pie(parameter,i,j,w_z,b_z)
#def delta_pie(parameter,i,j,direction)
#前向计算一遍h
#initialize parameter
#m=6;n=7
d_h=5;d_s=1
np.random.seed(0)

with open(u'H:\\testData.txt','r') as f:
    for line in f:
        line=line.strip()
        tmp=line.split('\t')
        break
container=[]
for i in tmp:
    container.append(i.split(' '))
worddic={}
with open(u'H:\\1000词向量.txt','r') as f:
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
s=np.zeros((m,n,d_s,1))


for i in range(len(vectors[0])):
    for j in range(len(vectors[1])):
        
        tmp=sum(vectors[0][i]*vectors[1][j])/(np.sqrt(vectors[0][i].dot(vectors[0][i]))*np.sqrt(vectors[1][j].dot(vectors[1][j])))
        if math.isnan(tmp):
            tmp=0
        s[i][j][0][0]=tmp

#x1=np.array([[1,2],[3,4],[5,6]])
#x2=np.array([[1,2],[3,4],[5,6]])
#
#y=sum(x1*x2)/(np.sqrt(x1.dot(x1))*np.sqrt(x2.dot(x2)))
w_r_l=np.random.random((d_h,d_s+3*d_h))/1;w_r_t=np.random.random((d_h,d_s+3*d_h))/1
w_r_d=np.random.random((d_h,d_s+3*d_h))/1

w_z_l=np.random.random((d_h,d_s+3*d_h))/1;w_z_t=np.random.random((d_h,d_s+3*d_h))/1
w_z_d=np.random.random((d_h,d_s+3*d_h))/1;w_z_i=np.random.random((d_h,d_s+3*d_h))/1

b_r_l=np.random.random((d_h,1))/1;b_r_t=np.random.random((d_h,1))/1
b_r_d=np.random.random((d_h,1))/1
b_z_l=np.random.random((d_h,1))/1;b_z_t=np.random.random((d_h,1))/1
b_z_d=np.random.random((d_h,1))/1;b_z_i=np.random.random((d_h,1))/1

w=np.random.random((d_h,d_s))/1;U=np.random.random((d_h,3*d_h))/1
b=np.random.random((d_h,1))/1;

w_s=np.random.random((1,d_h));
b_s=np.random.rand(1);

dw_r_l=np.zeros((m,n,d_h,d_s+3*d_h));dw_r_t=np.zeros((m,n,d_h,d_s+3*d_h))
dw_r_d=np.zeros((m,n,d_h,d_s+3*d_h))

dw_z_l=np.zeros((m,n,d_h,d_s+3*d_h));dw_z_t=np.zeros((m,n,d_h,d_s+3*d_h))
dw_z_d=np.zeros((m,n,d_h,d_s+3*d_h));dw_z_i=np.zeros((m,n,d_h,d_s+3*d_h))

db_r_l=np.zeros((m,n,d_h,1));db_r_t=np.zeros((m,n,d_h,1))
db_r_d=np.zeros((m,n,d_h,1))
db_z_l=np.zeros((m,n,d_h,1));db_z_t=np.zeros((m,n,d_h,1))
db_z_d=np.zeros((m,n,d_h,1));db_z_i=np.zeros((m,n,d_h,1))

dw=np.zeros((m,n,d_h,d_s));dU=np.zeros((m,n,d_h,3*d_h))
db=np.zeros((m,n,d_h,1))

dw_s=np.zeros((m,n,1,d_h));
db_s=np.zeros((m,n,1));

h=np.zeros((m,n,d_h,1))
h_pie=np.zeros((m,n,d_h,1))
e=np.zeros((m,n,d_h,1))
#s=np.random.random((m,n,d_s,1))



# use dictionary to store parameter
parameter=dict()
parameter['w_r_l']=w_r_l;parameter['w_r_t']=w_r_t;parameter['w_r_d']=w_r_d
parameter['w_z_l']=w_z_l;parameter['w_z_t']=w_z_t;parameter['w_z_d']=w_z_d
parameter['w_z_i']=w_z_i;parameter['b_r_l']=b_r_l;parameter['b_r_t']=b_r_t
parameter['b_r_d']=b_r_d;parameter['b_z_l']=b_z_l;parameter['b_z_t']=b_z_t
parameter['b_z_d']=b_z_d;parameter['b_z_i']=b_z_i

parameter['w']=w;parameter['U']=U;parameter['b']=b
parameter['w_s']=w_s;parameter['b_s']=b_s;parameter['h']=h
parameter['e']=e;parameter['s']=s
parameter['h_pie']=h_pie

parameter['dw_r_l']=dw_r_l;parameter['dw_r_t']=dw_r_t;parameter['dw_r_d']=dw_r_d
parameter['dw_z_l']=dw_z_l;parameter['dw_z_t']=dw_z_t;parameter['dw_z_d']=dw_z_d
parameter['dw_z_i']=dw_z_i;parameter['db_r_l']=db_r_l;parameter['db_r_t']=db_r_t
parameter['db_r_d']=db_r_d;parameter['db_z_l']=db_z_l;parameter['db_z_t']=db_z_t
parameter['db_z_d']=db_z_d;parameter['db_z_i']=db_z_i

parameter['dw']=dw;parameter['dU']=dU;parameter['db']=db
parameter['dw_s']=dw_s;parameter['db_s']=db_s    
    
    
    

e=parameter['e']
for i in range(m):
    for j in range(n):
        parameter['h'][i][j],parameter['h_pie'][i][j]=cal_h(parameter,i,j)

for i in range(m-1,-1,-1):
    for j in range(n-1,-1,-1):
        e[i][j]=epsilon(parameter,i,j,m,n)
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
        qr=np.concatenate((h_l,h_t,h_d),0)
        parameter['dw_r_l'][i][j]=delta_z_pie(parameter,i,j,'l').dot(q.T)
        parameter['dw_r_t'][i][j]=delta_z_pie(parameter,i,j,'t').dot(q.T)
        parameter['dw_r_d'][i][j]=delta_z_pie(parameter,i,j,'d').dot(q.T)
        
#        parameter['dw_z_l']=dw_z_l
#        parameter['dw_z_t']=dw_z_t
#        parameter['dw_z_d']=dw_z_d
#        parameter['dw_z_i']=dw_z_i
        
        parameter['db_r_l'][i][j]=delta_z_pie(parameter,i,j,'l')
        parameter['db_r_t'][i][j]=delta_z_pie(parameter,i,j,'t')
        parameter['db_r_d'][i][j]=delta_z_pie(parameter,i,j,'d')
        
#        parameter['db_z_l']=db_z_l
#        parameter['db_z_t']=db_z_t
#        parameter['db_z_d']=db_z_d
#        parameter['db_z_i']=db_z_i
##        
#        parameter['dw']=dw
#        parameter['dU']=dU
#        parameter['db']=db
#        parameter['dw_s']=dw_s
#        parameter['db_s']=db_s
      
        parameter['dU'][i][j]=delta_pie(parameter,i,j).dot((cal_r_con(parameter,i,j)*qr).T)  
        parameter['dw'][i][j]=delta_pie(parameter,i,j).dot(parameter['s'][i][j])


print(parameter['dw'][-1][-1][-1][-1])       
left=sum(parameter['dw'])
print(testout(parameter['h'],parameter['w_s'],parameter['b_s']) )



parameter['s'][-1][-1]+=1e-6
for i in range(m):
    for j in range(n):
        parameter['h'][i][j],parameter['h_pie'][i][j]=cal_h(parameter,i,j)
Jplus=testout(parameter['h'],parameter['w_s'],parameter['b_s'])   
parameter['s'][-1][-1]-=2e-6
for i in range(m):
    for j in range(n):
        parameter['h'][i][j],parameter['h_pie'][i][j]=cal_h(parameter,i,j)
Jminus=testout(parameter['h'],parameter['w_s'],parameter['b_s']) 
print(left)
print((Jplus-Jminus)/(2e-6)) 



    
#print(parameter['dU'][-1][-1][-1][-1])       
#left=(sum(sum(parameter['dU'])))[-1][-1]
#print(testout(parameter['h'],parameter['w_s'],parameter['b_s']) )
#
#
#
#parameter['U'][-1][-1]+=1e-6
#for i in range(m):
#    for j in range(n):
#        parameter['h'][i][j],parameter['h_pie'][i][j]=cal_h(parameter,i,j)
#Jplus=testout(parameter['h'],parameter['w_s'],parameter['b_s'])   
#parameter['U'][-1][-1]-=2e-6
#for i in range(m):
#    for j in range(n):
#        parameter['h'][i][j],parameter['h_pie'][i][j]=cal_h(parameter,i,j)
#Jminus=testout(parameter['h'],parameter['w_s'],parameter['b_s']) 
#print(left)
#print((Jplus-Jminus)/(2e-6)) 
      
      
#pw=(parameter['h'])[-1][-1]
#print(testout(parameter['h'],parameter['w_s'],parameter['b_s']) )
#
#
#
#
#
#
#parameter['w_s'][0][0]+=1e-6
#for i in range(m):
#    for j in range(n):
#        parameter['h'][i][j],parameter['h_pie'][i][j]=cal_h(parameter,i,j)
#Jplus=testout(parameter['h'],parameter['w_s'],parameter['b_s'])   
#parameter['w_s'][0][0]-=2e-6
#for i in range(m):
#    for j in range(n):
#        parameter['h'][i][j],parameter['h_pie'][i][j]=cal_h(parameter,i,j)
#Jminus=testout(parameter['h'],parameter['w_s'],parameter['b_s']) 
#print(pw)
#print((Jplus-Jminus)/(2e-6))  









#pw=sum(sum(parameter['dw_r_l']))[0][0]
#pw=parameter['dw_r_l'][-1][-1]
#print(testout(parameter['h'],parameter['w_s'],parameter['b_s']) )
#
#parameter['w_r_l'][0][0]+=1e-3
#for i in range(m):
#    for j in range(n):
#        parameter['h'][i][j],parameter['h_pie'][i][j]=cal_h(parameter,i,j)
#Jplus=testout(parameter['h'],parameter['w_s'],parameter['b_s'])   
#parameter['w_r_l'][0][0]-=2e-3
#for i in range(m):
#    for j in range(n):
#        parameter['h'][i][j],parameter['h_pie'][i][j]=cal_h(parameter,i,j)
#Jminus=testout(parameter['h'],parameter['w_s'],parameter['b_s']) 
#print(pw)
#print((Jplus-Jminus)/(2e-3))  





#parameter['w_r_l'][0][0]-=20
#for i in range(m):
#    for j in range(n):
#        parameter['htestminus'][i][j],parameter['h_pie'][i][j]=cal_h(parameter,i,j)
#print(testout(parameter['htestminus'],parameter['w_s'],parameter['b_s']) )
       
#print (sum(sum(e)))
#print(delta_pie(parameter,1,1))
#       delta_r(parameter,i,j,direction)
#def epsilon(m,n,i,j,parameter):
#        #第（m，n）个        
#    if j==n-1 and i==m-1:
#        return parameter['w_s'].T
#    #第（i，n）个，即最后一列
#    elif j==n-1:
#        return -1
#    #第（m，j）个，最后一行
#    elif i==m-1:
#        return parameter['z_t']*epsilon(m,n,i,j+1,parameter)+delta_pie(h,e,i,j)
#    else:
#        return i+j
#def delta_pie(h,e,i,j):
#    return h[i,j].dot(e[i,j].T)
#print epsilon(m,n,5,6,parameter)
