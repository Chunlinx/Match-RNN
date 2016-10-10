# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 13:13:34 2016

@author: Yang
3.7: 添加添加loss记录,添加drop out
3.5: 修改bug
3.4：去掉重复计算的变量,效率提高300%
3.3：前向计算返回parameter
3.2：添加minibath的外层循环，遮罩层
3.1:调整参数初始化的数值设置，对每个变量分配独自的rho权重
3.0:这一版本不包含gradient check,完成了minibatch
2.8:去掉之前构造的二次loss函数
2.7:添加minbatch
2.6:这一版本将求导功能写成函数
"""
import random
import numpy as np
import math
import copy

def sigmoid(x):
    return 1/(1+np.exp(-x))

#calculate reset gate from different directions
def cal_r(parameter,i,j,w_r,b_r):
    #只与前向有关
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
    #只与前向有关
    r_l=cal_r(parameter,i,j,parameter['w_r_l'],parameter['b_r_l'])
    r_t=cal_r(parameter,i,j,parameter['w_r_t'],parameter['b_r_t'])
    r_d=cal_r(parameter,i,j,parameter['w_r_d'],parameter['b_r_d'])
    return np.concatenate((r_l,r_t,r_d))
    
#compute 'inner' update gate
def cal_z_pie(parameter,i,j,w_z,b_z):
    #只与前向有关
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
    q=np.concatenate((h_l,h_t,h_d,parameter['s'][i][j]),0)
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
    h=parameter['h'] 
    b=parameter['b']

    z_l=cal_z(parameter,i,j,'l')
    z_t=cal_z(parameter,i,j,'t')
    z_d=cal_z(parameter,i,j,'d')
    z_i=cal_z(parameter,i,j,'i')

    r=cal_r_con(parameter,i,j)
    if i==0 and j==0:
        a_h_pie=w.dot(parameter['s'][i][j])+b
        h_pie=np.tanh(a_h_pie)
        h_ij=z_i*h_pie
    elif i>0 and j>0:
        h_con=np.concatenate((h[i-1][j],h[i][j-1],h[i-1][j-1]))
        tmp=r*h_con
        a_h_pie=w.dot(parameter['s'][i][j])+U.dot(tmp)+b
        h_pie=np.tanh(a_h_pie)
        h_ij=z_i*h_pie+z_l*h[i-1][j]+z_t*h[i][j-1]+z_d*h[i-1][j-1]
    elif i==0:
        h_con=np.concatenate((np.zeros_like(h[i][j-1]),h[i][j-1],np.zeros_like(h[i][j-1])))
        tmp=r*h_con
        a_h_pie=w.dot(parameter['s'][i][j])+U.dot(tmp)+b
        h_pie=np.tanh(a_h_pie)
        h_ij=z_i*h_pie+z_t*h[i][j-1]
        
    elif j==0:
        h_con=np.concatenate((h[i-1][j],np.zeros_like(h[i-1][j]),np.zeros_like(h[i-1][j])))
        tmp=r*h_con
        a_h_pie=w.dot(parameter['s'][i][j])+U.dot(tmp)+b
        h_pie=np.tanh(a_h_pie)
        h_ij=z_i*h_pie+z_l*h[i-1][j]
        
    return h_ij,h_pie,z_l,z_t,z_d,z_i




#后向传播计算导数 
'''checked and opted'''     
def delta_z_pie(parameter,i,j,direction):
    z=parameter['z'][direction][i][j]
    if direction=='l': 
        delta_z=parameter['e'][i][j]*(parameter['h'][i-1][j])
        delta_z_pie_value=delta_z*z*(1-z)                
                
        return delta_z_pie_value
    elif direction=='t':
        delta_z=parameter['e'][i][j]*(parameter['h'][i][j-1])
        delta_z_pie_value=delta_z*z*(1-z)
        return delta_z_pie_value
        
    elif direction=='d':
        delta_z=parameter['e'][i][j]*(parameter['h'][i-1][j-1])
        delta_z_pie_value=delta_z*z*(1-z)
        return delta_z_pie_value
        
    elif direction=='i':
        delta_z=parameter['e'][i][j]*(parameter['h_pie'][i][j])
        delta_z_pie_value=delta_z*z*(1-z) 
        return delta_z_pie_value


#eq (1)
'''checked and opted'''
def delta_pie(parameter,i,j):
    z_d=parameter['z']['i'][i][j]
    delta_pie_val=parameter['e'][i][j]*z_d*(1-parameter['h_pie'][i][j]*parameter['h_pie'][i][j])
    return delta_pie_val

'''checked'''
def delta_r(parameter,i,j,direction):
    '''chongfu'''
    '''只和前向计算有关，后向计算使用时可以直接调用'''
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
    return epsilon_r*(r*(1-r))
#print (delta_r(parameter,1,1,'l'))
def epsilon(parameter,i,j,m,n):
    length=len(parameter['h'][0][0])
    w_rl=parameter['w_r_l'];w_rt=parameter['w_r_t'];w_rd=parameter['w_r_d']
    w_zl=parameter['w_z_l'];w_zt=parameter['w_z_t'];w_zd=parameter['w_z_d']
    w_zi=parameter['w_z_i']
    if i==m-1 and j==n-1:
        return parameter['w_s'].T
    elif i==m-1:

        e=parameter['e'];U=parameter['U']
   
        
        firstline=parameter['z']['t'][i][j+1]*e[i][j+1]
        
        r_line_2=parameter['delta_r']['l'][i][j+1].T.dot(w_rl).T+parameter['delta_r']['t'][i][j+1].T.dot(w_rt).T+parameter['delta_r']['d'][i][j+1].T.dot(w_rd).T
        
        z_line_2=parameter['delta_z_pie']['l'][i][j+1].T.dot(w_zl).T+parameter['delta_z_pie']['t'][i][j+1].T.dot(w_zt).T+parameter['delta_z_pie']['d'][i][j+1].T.dot(w_zd).T+parameter['delta_z_pie']['i'][i][j+1].T.dot(w_zi).T
    
        lastline=parameter['delta_pie'][i][j+1].T.dot(U).T*cal_r_con(parameter,i,j+1)
        
        return lastline[length:2*length]+r_line_2[length:2*length]+z_line_2[length:2*length]+firstline
    elif j==n-1:

        e=parameter['e'];U=parameter['U']

        firstline=parameter['z']['l'][i+1][j]*e[i+1][j]
        r_line_1=parameter['delta_r']['l'][i+1][j].T.dot(w_rl).T+parameter['delta_r']['t'][i+1][j].T.dot(w_rt).T+parameter['delta_r']['d'][i+1][j].T.dot(w_rd).T
        
        z_line_1=parameter['delta_z_pie']['l'][i+1][j].T.dot(w_zl).T+parameter['delta_z_pie']['t'][i+1][j].T.dot(w_zt).T+parameter['delta_z_pie']['d'][i+1][j].T.dot(w_zd).T+parameter['delta_z_pie']['i'][i+1][j].T.dot(w_zi).T
    
        lastline=parameter['delta_pie'][i+1][j].T.dot(U).T*cal_r_con(parameter,i+1,j)
        return lastline[:length]+r_line_1[:length]+z_line_1[:length]+firstline
    else:

        e=parameter['e'];U=parameter['U']
 
        
        firstline=parameter['z']['l'][i+1][j]*e[i+1][j]+parameter['z']['t'][i][j+1]*e[i][j+1]+parameter['z']['d'][i+1][j+1]*e[i+1][j+1]
        r_line_1=parameter['delta_r']['l'][i+1][j].T.dot(w_rl).T+parameter['delta_r']['t'][i+1][j].T.dot(w_rt).T+parameter['delta_r']['d'][i+1][j].T.dot(w_rd).T
        r_line_2=parameter['delta_r']['l'][i][j+1].T.dot(w_rl).T+parameter['delta_r']['t'][i][j+1].T.dot(w_rt).T+parameter['delta_r']['d'][i][j+1].T.dot(w_rd).T
        r_line_3=parameter['delta_r']['l'][i+1][j+1].T.dot(w_rl).T+parameter['delta_r']['t'][i+1][j+1].T.dot(w_rt).T+parameter['delta_r']['d'][i+1][j+1].T.dot(w_rd).T
        
        z_line_1=parameter['delta_z_pie']['l'][i+1][j].T.dot(w_zl).T+parameter['delta_z_pie']['t'][i+1][j].T.dot(w_zt).T+parameter['delta_z_pie']['d'][i+1][j].T.dot(w_zd).T+parameter['delta_z_pie']['i'][i+1][j].T.dot(w_zi).T
        z_line_2=parameter['delta_z_pie']['l'][i][j+1].T.dot(w_zl).T+parameter['delta_z_pie']['t'][i][j+1].T.dot(w_zt).T+parameter['delta_z_pie']['d'][i][j+1].T.dot(w_zd).T+parameter['delta_z_pie']['i'][i][j+1].T.dot(w_zi).T
        z_line_3=parameter['delta_z_pie']['l'][i+1][j+1].T.dot(w_zl).T+parameter['delta_z_pie']['t'][i+1][j+1].T.dot(w_zt).T+parameter['delta_z_pie']['d'][i+1][j+1].T.dot(w_zd).T+parameter['delta_z_pie']['i'][i+1][j+1].T.dot(w_zi).T
    
        lastline=(parameter['delta_pie'][i+1][j].T.dot(U).T*cal_r_con(parameter,i+1,j))[:length]+(parameter['delta_pie'][i][j+1].T.dot(U).T*cal_r_con(parameter,i,j+1))[length:2*length]+(parameter['delta_pie'][i+1][j+1].T.dot(U).T*cal_r_con(parameter,i+1,j+1))[2*length:3*length]
        
        return lastline+r_line_1[:length]+r_line_2[length:2*length]+r_line_3[2*length:3*length]+z_line_1[:length]+z_line_2[length:2*length]+z_line_3[2*length:3*length]+firstline

def out(parameter):
    return parameter['w_s'].dot(parameter['h'][-1][-1])+parameter['b_s']



    
    
 
    
    
def forward(m,n,parameter,d_h):    
    for i in range(m):
        for j in range(n):
            parameter['h'][i][j],parameter['h_pie'][i][j],parameter['z']['l'][i][j],parameter['z']['t'][i][j],parameter['z']['d'][i][j],parameter['z']['i'][i][j]=cal_h(parameter,i,j)
    return parameter
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
            
            #其他导数的存储
            parameter['delta_r']['l'][i][j]=delta_r(parameter,i,j,'l')
            parameter['delta_r']['t'][i][j]=delta_r(parameter,i,j,'t')
            parameter['delta_r']['d'][i][j]=delta_r(parameter,i,j,'d')

            parameter['delta_z_pie']['l'][i][j]=delta_z_pie(parameter,i,j,'l')
            parameter['delta_z_pie']['t'][i][j]=delta_z_pie(parameter,i,j,'t')
            parameter['delta_z_pie']['d'][i][j]=delta_z_pie(parameter,i,j,'d')
            parameter['delta_z_pie']['i'][i][j]=delta_z_pie(parameter,i,j,'i')

            parameter['delta_pie'][i][j]=delta_pie(parameter,i,j)
            #参数导数的叠加
            parameter['dw_r_l']+=parameter['delta_r']['l'][i][j].dot(q.T)
            parameter['dw_r_t']+=parameter['delta_r']['t'][i][j].dot(q.T)
            parameter['dw_r_d']+=parameter['delta_r']['d'][i][j].dot(q.T)
            
            parameter['dw_z_l']+=parameter['delta_z_pie']['l'][i][j].dot(q.T)
            parameter['dw_z_t']+=parameter['delta_z_pie']['t'][i][j].dot(q.T)
            parameter['dw_z_d']+=parameter['delta_z_pie']['d'][i][j].dot(q.T)
            parameter['dw_z_i']+=parameter['delta_z_pie']['i'][i][j].dot(q.T)
            
            parameter['db_r_l']+=parameter['delta_r']['l'][i][j]
            parameter['db_r_t']+=parameter['delta_r']['t'][i][j]
            parameter['db_r_d']+=parameter['delta_r']['d'][i][j]
            
            parameter['db_z_l']+=parameter['delta_z_pie']['l'][i][j]
            parameter['db_z_t']+=parameter['delta_z_pie']['t'][i][j]
            parameter['db_z_d']+=parameter['delta_z_pie']['d'][i][j]
            parameter['db_z_i']+=parameter['delta_z_pie']['i'][i][j]
    ##        
            parameter['dU']+=parameter['delta_pie'][i][j].dot((cal_r_con(parameter,i,j)*qr).T)  
            parameter['dw']+=parameter['delta_pie'][i][j].dot(parameter['s'][i][j].T)
            parameter['db']+=parameter['delta_pie'][i][j]
    parameter['dw_s']=parameter['h'][-1][-1].T
    return parameter
    
def result(h,w_s):
    return w_s.dot(h[-1][-1])
def calhmn(parameter,datasets,index):
    m=np.shape(datasets[index])[0]
    n=np.shape(datasets[index])[1]
    parameter['s']=copy.deepcopy(datasets[index])
    #reset forward para
    parameter=initiallize_bin(parameter,m,n,d_h,d_s)
    #forward compute
    parameter=forward(m,n,parameter,d_h)
    hval=copy.deepcopy(parameter['h'])
    return hval 
    
def gradcheckfornewdata(dpara,para,sminus,splus,parameter,index=0,d_h=5):   
    left=dpara[-1][-1]
    
    parameter[para][-1][-1]+=1e-6
    hplus=calhmn(parameter,splus,index)
    hminus=calhmn(parameter,sminus,index)
 
    Jplus=testoutfornewdata(hminus,hplus,parameter['w_s'])

    parameter[para][-1][-1]-=2e-6
    
    hplus=calhmn(parameter,splus,index)
    hminus=calhmn(parameter,sminus,index)
    
    Jminus=testoutfornewdata(hminus,hplus,parameter['w_s'])
    
 
    print('our calculation of the derivative of the function',left)
    print('The derivative of the function calculate by the defination',((Jplus-Jminus)/(2e-6))[-1][-1])

def minbatch(parameter,sminus,splus,d_h,d_s,batch,rho,maskval,maxiter):
    loop=0
    while 1:
        innertime1=clock()
        
        loop+=1
        print('Loop:'+str(loop) )
        if loop==maxiter:
            break
        counter=0
        parameter=initiallize_grad(parameter,d_h,d_s)
        #初始化，设置为0 
        dw_r_l=np.zeros_like(parameter['dw_r_l'])
        dw_r_d=np.zeros_like(parameter['dw_r_d'])
        dw_r_t=np.zeros_like(parameter['dw_r_t'])
        dw_z_l=np.zeros_like(parameter['dw_z_l'])
        dw_z_t=np.zeros_like(parameter['dw_z_t'])
        dw_z_d=np.zeros_like(parameter['dw_z_d'])
        dw_z_i=np.zeros_like(parameter['dw_z_i'])
        db_r_l=np.zeros_like(parameter['db_r_l'])
        db_r_t=np.zeros_like(parameter['db_r_t'])
        db_r_d=np.zeros_like(parameter['db_r_d'])
        db_z_l=np.zeros_like(parameter['db_z_l'])
        db_z_t=np.zeros_like(parameter['db_z_t'])
        db_z_d=np.zeros_like(parameter['db_z_d'])
        db_z_i=np.zeros_like(parameter['db_z_i'])
        dU=np.zeros_like(parameter['dU'])
        dw=np.zeros_like(parameter['dw'])
        db=np.zeros_like(parameter['db'])
        dw_s=np.zeros_like(parameter['dw_s'])  
        #初始化adagrad方法的分母
        ada_dw_r_l=np.zeros_like(parameter['dw_r_l'])
        ada_dw_r_d=np.zeros_like(parameter['dw_r_d'])
        ada_dw_r_t=np.zeros_like(parameter['dw_r_t'])
        ada_dw_z_l=np.zeros_like(parameter['dw_z_l'])
        ada_dw_z_t=np.zeros_like(parameter['dw_z_t'])
        ada_dw_z_d=np.zeros_like(parameter['dw_z_d'])
        ada_dw_z_i=np.zeros_like(parameter['dw_z_i'])
        ada_db_r_l=np.zeros_like(parameter['db_r_l'])
        ada_db_r_t=np.zeros_like(parameter['db_r_t'])
        ada_db_r_d=np.zeros_like(parameter['db_r_d'])
        ada_db_z_l=np.zeros_like(parameter['db_z_l'])
        ada_db_z_t=np.zeros_like(parameter['db_z_t'])
        ada_db_z_d=np.zeros_like(parameter['db_z_d'])
        ada_db_z_i=np.zeros_like(parameter['db_z_i'])
        ada_dU=np.zeros_like(parameter['dU'])
        ada_dw=np.zeros_like(parameter['dw'])
        ada_db=np.zeros_like(parameter['db'])
        ada_dw_s=np.zeros_like(parameter['dw_s'])  
        for index in range(len(sminus)):
            #计算M(s+)
            m=np.shape(splus[index])[0]
            n=np.shape(splus[index])[1]
            #对于每一条输入数据，都初始化和参数无关的变量，包括h，e，h‘
            parameter=initiallize_grad(parameter,d_h,d_s)
            parameter=initiallize_bin(parameter,m,n,d_h,d_s)
            #传入输入
            parameter['s']=copy.deepcopy(splus[index])
            #前向计算
            parameter=forward(m,n,parameter,d_h)  
            result_plus=result(parameter['h'],parameter['w_s'])  
            
            #后向计算        
            parameter=backword(m,n,parameter)   
            
            #计算M(S+)各个导数，直至batch次
            dw_r_l_plus=copy.deepcopy(parameter['dw_r_l'])
            dw_r_d_plus=copy.deepcopy(parameter['dw_r_d'])
            dw_r_t_plus=copy.deepcopy(parameter['dw_r_t'])
            dw_z_l_plus=copy.deepcopy(parameter['dw_z_l'])
            dw_z_t_plus=copy.deepcopy(parameter['dw_z_t'])
            dw_z_d_plus=copy.deepcopy(parameter['dw_z_d'])
            dw_z_i_plus=copy.deepcopy(parameter['dw_z_i'])
            db_r_l_plus=copy.deepcopy(parameter['db_r_l'])
            db_r_t_plus=copy.deepcopy(parameter['db_r_t'])
            db_r_d_plus=copy.deepcopy(parameter['db_r_d'])
            db_z_l_plus=copy.deepcopy(parameter['db_z_l'])
            db_z_t_plus=copy.deepcopy(parameter['db_z_t'])
            db_z_d_plus=copy.deepcopy(parameter['db_z_d'])
            db_z_i_plus=copy.deepcopy(parameter['db_z_i'])
            dU_plus=copy.deepcopy(parameter['dU'])
            dw_plus=copy.deepcopy(parameter['dw'])
            db_plus=copy.deepcopy(parameter['db'])
            dw_s_plus=copy.deepcopy(parameter['dw_s'])
     
            #计算M(S-),结构和上面相同
            m=np.shape(sminus[index])[0]
            n=np.shape(sminus[index])[1]
                
            parameter=initiallize_grad(parameter,d_h,d_s)
            parameter=initiallize_bin(parameter,m,n,d_h,d_s)
    
            parameter['s']=copy.deepcopy(sminus[index])
    
            parameter=forward(m,n,parameter,d_h)    
            result_minus=result(parameter['h'],parameter['w_s'])

            parameter=backword(m,n,parameter)
    
            dw_r_l_minus=copy.deepcopy(parameter['dw_r_l'])
            dw_r_d_minus=copy.deepcopy(parameter['dw_r_d'])
            dw_r_t_minus=copy.deepcopy(parameter['dw_r_t'])
            dw_z_l_minus=copy.deepcopy(parameter['dw_z_l'])
            dw_z_t_minus=copy.deepcopy(parameter['dw_z_t'])
            dw_z_d_minus=copy.deepcopy(parameter['dw_z_d'])
            dw_z_i_minus=copy.deepcopy(parameter['dw_z_i'])
            db_r_l_minus=copy.deepcopy(parameter['db_r_l'])
            db_r_t_minus=copy.deepcopy(parameter['db_r_t'])
            db_r_d_minus=copy.deepcopy(parameter['db_r_d'])
            db_z_l_minus=copy.deepcopy(parameter['db_z_l'])
            db_z_t_minus=copy.deepcopy(parameter['db_z_t'])
            db_z_d_minus=copy.deepcopy(parameter['db_z_d'])
            db_z_i_minus=copy.deepcopy(parameter['db_z_i'])
            dU_minus=copy.deepcopy(parameter['dU'])
            dw_minus=copy.deepcopy(parameter['dw'])
            db_minus=copy.deepcopy(parameter['db'])
            dw_s_minus=copy.deepcopy(parameter['dw_s']) 
            #对max函数求导
            if result_plus-result_minus<-4:
                pass
            else:

                #累加计算loss函数的导数，直至batch次        
                dw_r_l+=dw_r_l_plus-dw_r_l_minus
                dw_r_d+=dw_r_d_plus-dw_r_d_minus
                dw_r_t+=dw_r_t_plus-dw_r_t_minus
                
                dw_z_l+=dw_z_l_plus-dw_z_l_minus
                dw_z_t+=dw_z_t_plus-dw_z_t_minus
                dw_z_d+=dw_z_d_plus-dw_z_d_minus       
                dw_z_i+=dw_z_i_plus-dw_z_i_minus
                
                db_r_l+=db_r_l_plus-db_r_l_minus
                db_r_t+=db_r_t_plus-db_r_t_minus
                db_r_d+=db_r_d_plus-db_r_d_minus
                
                db_z_l+=db_z_l_plus-db_z_l_minus
                db_z_t+=db_z_t_plus-db_z_t_minus
                db_z_d+=db_z_d_plus-db_z_d_minus
                db_z_i+=db_z_i_plus-db_z_i_minus
                
                dU+=dU_plus-dU_minus
                dw+=dw_plus-dw_minus
                db+=db_plus-db_minus
                dw_s+=dw_s_plus-dw_s_minus
            #当叠加batch次后
            counter+=1
            if counter==batch:
                #重置计数器
                counter=0
                #计算导数的均值
                ave_dw_r_l=dw_r_l/batch
                ave_dw_r_d=dw_r_d/batch
                ave_dw_r_t=dw_r_t/batch       
                ave_dw_z_l=dw_z_l/batch
                ave_dw_z_t=dw_z_t/batch
                ave_dw_z_d=dw_z_d/batch      
                ave_dw_z_i=dw_z_i/batch           
                ave_db_r_l=db_r_l/batch
                ave_db_r_t=db_r_t/batch
                ave_db_r_d=db_r_d/batch            
                ave_db_z_l=db_z_l/batch
                ave_db_z_t=db_z_t/batch
                ave_db_z_d=db_z_d/batch
                ave_db_z_i=db_z_i/batch        
                ave_dU=dU/batch
                ave_dw=dw/batch
                ave_db=db/batch
                ave_dw_s=dw_s/batch
                #计算adagrad的分母
                ada_dw_r_l+=ave_dw_r_l*ave_dw_r_l
                ada_dw_r_d+=ave_dw_r_d*ave_dw_r_d
                ada_dw_r_t+=ave_dw_r_t*ave_dw_r_t
                ada_dw_z_l+=ave_dw_z_l*ave_dw_z_l
                ada_dw_z_t+=ave_dw_z_t*ave_dw_z_t
                ada_dw_z_d+=ave_dw_z_d*ave_dw_z_d
                ada_dw_z_i+=ave_dw_z_i*ave_dw_z_i
                ada_db_r_l+=ave_db_r_l*ave_db_r_l
                ada_db_r_t+=ave_db_r_t*ave_db_r_t
                ada_db_r_d+=ave_db_r_d*ave_db_r_d
                ada_db_z_l+=ave_db_z_l*ave_db_z_l
                ada_db_z_t+=ave_db_z_t*ave_db_z_t
                ada_db_z_d+=ave_db_z_d*ave_db_z_d
                ada_db_z_i+=ave_db_z_i*ave_db_z_i
                ada_dU+=ave_dU*ave_dU
                ada_dw+=ave_dw*ave_dw
                ada_db+=ave_db*ave_db
                ada_dw_s+=ave_dw_s*ave_dw_s 
#                print ('lambda',(rho[0]/np.sqrt(ada_dw_r_l)))
#                print('grad',ave_dw_r_l)

                
                mask=dict()
               
                if maskval==1:
                    for i in range(7):
                        maskseed=np.random.binomial(1,0.5,np.shape(parameter['w_r_l'])[0])
                        mask[i]=np.zeros_like(parameter['w_r_l'])
                        for line in range(len(mask[i])):
                            for col in range(len(mask[i][0])):
                                if maskseed[line]==1:
                                    mask[i][line][col]=1
                    for i in range(7,14):
                        maskseed=np.random.binomial(1,0.5,np.shape(parameter['b_r_l'])[0])
                        mask[i]=np.zeros_like(parameter['b_r_l'])
                        for line in range(len(mask[i])):
                            for col in range(len(mask[i][0])):
                                if maskseed[line]==1:
                                    mask[i][line][col]=1
                    maskseed=np.random.binomial(1,0.5,np.shape(parameter['U'])[0])
                    mask[14]=np.zeros_like(parameter['U'])
                    for line in range(len(mask[14])):
                        for col in range(len(mask[14][0])):
                            if maskseed[line]==1:
                                mask[14][line][col]=1
                    maskseed=np.random.binomial(1,0.5,np.shape(parameter['w'])[0])
                    mask[15]=np.zeros_like(parameter['w'])
                    for line in range(len(mask[15])):
                        for col in range(len(mask[15][0])):
                            if maskseed[line]==1:
                                mask[15][line][col]=1
                    maskseed=np.random.binomial(1,0.5,np.shape(parameter['b'])[0])
                    mask[16]=np.zeros_like(parameter['b'])
                    for line in range(len(mask[16])):
                        for col in range(len(mask[16][0])):
                            if maskseed[line]==1:
                                mask[16][line][col]=1
                    maskseed=np.random.binomial(1,0.5,np.shape(parameter['w_s'])[1])
                    mask[17]=np.zeros_like(parameter['w_s'])
                    for line in range(len(mask[17][0])):
                        if maskseed[line]==1:
                            mask[17][0][line]=1
                elif maskval==0:
                    mask[0]=np.ones_like(parameter['w_r_l'])
                    mask[1]=np.ones_like(parameter['w_r_t'])
                    mask[2]=np.ones_like(parameter['w_r_d'])
                    mask[3]=np.ones_like(parameter['w_z_l'])
                    mask[4]=np.ones_like(parameter['w_z_t'])
                    mask[5]=np.ones_like(parameter['w_z_d'])
                    mask[6]=np.ones_like(parameter['w_z_i'])
                    mask[7]=np.ones_like(parameter['b_r_l'])
                    mask[8]=np.ones_like(parameter['b_r_t'])
                    mask[9]=np.ones_like(parameter['b_r_d'])
                    mask[10]=np.ones_like(parameter['b_z_l'])
                    mask[11]=np.ones_like(parameter['b_z_t'])
                    mask[12]=np.ones_like(parameter['b_z_d'])
                    mask[13]=np.ones_like(parameter['b_z_i'])
                    mask[14]=np.ones_like(parameter['U'])
                    mask[15]=np.ones_like(parameter['w'])
                    mask[16]=np.ones_like(parameter['b'])
                    mask[17]=np.ones_like(parameter['w_s'])
                #mask            
                                #更新参数
                parameter['w_r_l']-=ave_dw_r_l*(rho[0]/np.sqrt(ada_dw_r_l))*mask[0]
                parameter['w_r_t']-=ave_dw_r_t*(rho[1]/np.sqrt(ada_dw_r_t))*mask[1]
                parameter['w_r_d']-=ave_dw_r_d*(rho[2]/np.sqrt(ada_dw_r_d))*mask[2]
                parameter['w_z_l']-=ave_dw_z_l*(rho[3]/np.sqrt(ada_dw_z_l))*mask[3]
                parameter['w_z_t']-=ave_dw_z_t*(rho[4]/np.sqrt(ada_dw_z_t))*mask[4]
                parameter['w_z_d']-=ave_dw_z_d*(rho[5]/np.sqrt(ada_dw_z_d))*mask[5]
                parameter['w_z_i']-=ave_dw_z_i*(rho[6]/np.sqrt(ada_dw_z_i))*mask[6]
                parameter['b_r_l']-=ave_db_r_l*(rho[7]/np.sqrt(ada_db_r_l))*mask[7]
                parameter['b_r_t']-=ave_db_r_t*(rho[8]/np.sqrt(ada_db_r_t))*mask[8]
                parameter['b_r_d']-=ave_db_r_d*(rho[9]/np.sqrt(ada_db_r_d))*mask[9]
                parameter['b_z_l']-=ave_db_z_l*(rho[10]/np.sqrt(ada_db_z_l))*mask[10]
                parameter['b_z_t']-=ave_db_z_t*(rho[11]/np.sqrt(ada_db_z_t))*mask[11]
                parameter['b_z_d']-=ave_db_z_d*(rho[12]/np.sqrt(ada_db_z_d))*mask[12]
                parameter['b_z_i']-=ave_db_z_i*(rho[13]/np.sqrt(ada_db_z_i))*mask[13]
                parameter['U']-=ave_dU*(rho[14]/np.sqrt(ada_dU))*mask[14]
                parameter['w']-=ave_dw*(rho[15]/np.sqrt(ada_dw))*mask[15]
                parameter['b']-=ave_db*(rho[16]/np.sqrt(ada_db))*mask[16]
                parameter['w_s']-=ave_dw_s*(rho[17]/np.sqrt(ada_dw_s))*mask[17]
                
                parameter['result'].append(4+result_plus-result_minus)
                #导数初始化，设置为0 
                dw_r_l=np.zeros_like(parameter['dw_r_l'])
                dw_r_d=np.zeros_like(parameter['dw_r_d'])
                dw_r_t=np.zeros_like(parameter['dw_r_t'])
                dw_z_l=np.zeros_like(parameter['dw_z_l'])
                dw_z_t=np.zeros_like(parameter['dw_z_t'])
                dw_z_d=np.zeros_like(parameter['dw_z_d'])
                dw_z_i=np.zeros_like(parameter['dw_z_i'])
                db_r_l=np.zeros_like(parameter['db_r_l'])
                db_r_t=np.zeros_like(parameter['db_r_t'])
                db_r_d=np.zeros_like(parameter['db_r_d'])
                db_z_l=np.zeros_like(parameter['db_z_l'])
                db_z_t=np.zeros_like(parameter['db_z_t'])
                db_z_d=np.zeros_like(parameter['db_z_d'])
                db_z_i=np.zeros_like(parameter['db_z_i'])
                dU=np.zeros_like(parameter['dU'])
                dw=np.zeros_like(parameter['dw'])
                db=np.zeros_like(parameter['db'])
                dw_s=np.zeros_like(parameter['dw_s'])
        innertime2=clock()
        print('one batch cost: '+str(innertime2-innertime1)+' s')
        print('Loss: '+str(4+result_plus-result_minus))
    return parameter
    
def initiallize_parameter(d_h,d_s,con=100):

    parameter=dict()
    parameter['w_r_l']=np.random.uniform(-np.sqrt(1./(d_s+4*d_h)), np.sqrt(1./(d_s+4*d_h)), (d_h,d_s+3*d_h))
    parameter['w_r_t']=np.random.uniform(-np.sqrt(1./(d_s+4*d_h)), np.sqrt(1./(d_s+4*d_h)),(d_h,d_s+3*d_h))
    parameter['w_r_d']=np.random.uniform(-np.sqrt(1./(d_s+4*d_h)), np.sqrt(1./(d_s+4*d_h)),(d_h,d_s+3*d_h))
    parameter['w_z_l']=np.random.uniform(-np.sqrt(1./(d_s+4*d_h)), np.sqrt(1./(d_s+4*d_h)),(d_h,d_s+3*d_h))
    parameter['w_z_t']=np.random.uniform(-np.sqrt(1./(d_s+4*d_h)), np.sqrt(1./(d_s+4*d_h)),(d_h,d_s+3*d_h))
    parameter['w_z_d']=np.random.uniform(-np.sqrt(1./(d_s+4*d_h)), np.sqrt(1./(d_s+4*d_h)),(d_h,d_s+3*d_h))
    parameter['w_z_i']=np.random.uniform(-np.sqrt(1./(d_s+4*d_h)), np.sqrt(1./(d_s+4*d_h)),(d_h,d_s+3*d_h))
    parameter['b_r_l']=np.random.uniform(-np.sqrt(1./d_h), np.sqrt(1./d_h),(d_h,1))
    parameter['b_r_t']=np.random.uniform(-np.sqrt(1./d_h), np.sqrt(1./d_h),(d_h,1))
    parameter['b_r_d']=np.random.uniform(-np.sqrt(1./d_h), np.sqrt(1./d_h),(d_h,1))
    parameter['b_z_l']=np.random.uniform(-np.sqrt(1./d_h), np.sqrt(1./d_h),(d_h,1))
    parameter['b_z_t']=np.random.uniform(-np.sqrt(1./d_h), np.sqrt(1./d_h),(d_h,1))
    parameter['b_z_d']=np.random.uniform(-np.sqrt(1./d_h), np.sqrt(1./d_h),(d_h,1))
    parameter['b_z_i']=np.random.uniform(-np.sqrt(1./d_h), np.sqrt(1./d_h),(d_h,1))
    
    parameter['w']=np.random.uniform(-np.sqrt(1./(d_s+d_h)), np.sqrt(1./(d_s+d_h)),(d_h,d_s))
    parameter['U']=np.random.uniform(-np.sqrt(1./(4*d_h)), np.sqrt(1./(4*d_h)),(d_h,3*d_h))
    parameter['b']=np.random.uniform(-np.sqrt(1./d_h), np.sqrt(1./d_h),(d_h,1))
    parameter['w_s']=np.random.uniform(-np.sqrt(1./d_h), np.sqrt(1./d_h),(1,d_h))
    parameter['b_s']=np.random.rand(1)
    parameter['result']=[]
    return parameter
    
def initiallize_grad(parameter,d_h,d_s,con=100):
    parameter['dw_r_l']=np.zeros((d_h,d_s+3*d_h))
    parameter['dw_r_t']=np.zeros((d_h,d_s+3*d_h))
    parameter['dw_r_d']=np.zeros((d_h,d_s+3*d_h))
    parameter['dw_z_l']=np.zeros((d_h,d_s+3*d_h))
    parameter['dw_z_t']=np.zeros((d_h,d_s+3*d_h))
    parameter['dw_z_d']=np.zeros((d_h,d_s+3*d_h))
    parameter['dw_z_i']=np.zeros((d_h,d_s+3*d_h))
    parameter['db_r_l']=np.zeros((d_h,1))
    parameter['db_r_t']=np.zeros((d_h,1))
    parameter['db_r_d']=np.zeros((d_h,1))
    parameter['db_z_l']=np.zeros((d_h,1))
    parameter['db_z_t']=np.zeros((d_h,1))
    parameter['db_z_d']=np.zeros((d_h,1))
    parameter['db_z_i']=np.zeros((d_h,1))
    
    parameter['dw']=np.zeros((d_h,d_s))
    parameter['dU']=np.zeros((d_h,3*d_h))
    parameter['db']=np.zeros((d_h,1))
    parameter['dw_s']=np.zeros((1,d_h))
    parameter['db_s']=np.zeros((1)) 
    
    return parameter
    
def initiallize_bin(parameter,m,n,d_h,d_s,con=100):
    parameter['h']=np.zeros((m,n,d_h,1))
    parameter['e']=np.zeros((m,n,d_h,1))
    parameter['h_pie']=np.zeros((m,n,d_h,1))
    parameter['z']=dict()
    parameter['z']['l']=np.zeros((m,n,d_h,1))
    parameter['z']['t']=np.zeros((m,n,d_h,1))
    parameter['z']['d']=np.zeros((m,n,d_h,1))
    parameter['z']['i']=np.zeros((m,n,d_h,1))
    parameter['delta_r']=dict()
    parameter['delta_r']['l']=np.zeros((m,n,d_h,1))
    parameter['delta_r']['t']=np.zeros((m,n,d_h,1))
    parameter['delta_r']['d']=np.zeros((m,n,d_h,1))
    parameter['delta_z_pie']=dict()
    parameter['delta_z_pie']['l']=np.zeros((m,n,d_h,1))
    parameter['delta_z_pie']['t']=np.zeros((m,n,d_h,1))
    parameter['delta_z_pie']['d']=np.zeros((m,n,d_h,1))
    parameter['delta_z_pie']['i']=np.zeros((m,n,d_h,1))
    parameter['delta_pie']=np.zeros((m,n,d_h,1))
    return parameter
def testout(h,w_s,b_s):
    return w_s.dot(h[-1][-1])+b_s
def gradcheck(para,parameter):       

    parameter[para][-1][-1]+=1e-6
    hplus=calhmn(parameter,sminus,index)
    
    Jplus=testout(hplus,parameter['w_s'],parameter['b_s'])   
    parameter[para][-1][-1]-=2e-6
    
    hplus=calhmn(parameter,sminus,index)
    
    Jminus=testout(hplus,parameter['w_s'],parameter['b_s']) 
    print('The derivative of the function calculate by the defination',((Jplus-Jminus)/(2e-6))[0][0]) 
    
    
def make_rho():
    rho=np.zeros(18)
    rho[0]=1e-5
    rho[1]=1e-5   
    rho[2]=1e-5
    rho[3]=1e-4
    rho[4]=1e-4
    rho[5]=1e-4   
    rho[6]=1e-4
    rho[7]=1e-4
    rho[8]=1e-4
    rho[9]=1e-4   
    rho[10]=1e-3
    rho[11]=1e-3
    rho[12]=1e-3
    rho[13]=1e-3
    rho[14]=1e-3
    rho[15]=1e-3
    rho[16]=1e-2   
    rho[17]=1e-2
    return rho

import pickle
from time import clock

with open(u'C:\\Users\\Lab\\Desktop\\实验\\欧氏距离归一化数据\\S_reduce.txt','rb') as f:   
    sminus=pickle.load(f)
with open(u'C:\\Users\\Lab\\Desktop\\实验\\欧氏距离归一化数据\\S_plus.txt','rb') as f:   
    splus=pickle.load(f)
  
d_h=5;d_s=1
parameter=initiallize_parameter(d_h,d_s)
wrt=copy.deepcopy(parameter)
rho=make_rho()*10

start=clock()
parameter=minbatch(parameter,sminus,splus,d_h,d_s,10,rho,1,50)
end=clock()
print('runtime:',end-start)

with open(u'result_3.7_multi10_dropout_lossplot_No1.txt','wb') as f:   
    pickle.dump(parameter,f)
