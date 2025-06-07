# -*- coding: utf-8 -*-
"""
Spyder Editor
xulei
"""

#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import math
year = [1970,1971,1972,1973,1974,1975,1976,1977,1978,1979,1980,1981,1982,1983,1984,1985,1986,1987,1988,1989,1990,0]
people = [830,852,872,892,909,924,937,950,963,975,987,1001,1017,1030,1044,1059,1075,1093,1110,1127,1143,0]
Rf = np.zeros(21) 

def Lagrange_F(x):#此函数定义由拉格朗日插值法求得的函数。
    F = -37*x**4/15000+29347*x**3/1500-1396607*x**2/24+923101735*x/12-38133121326
    return F
def Lagrange_dF(X):#此函数定义由拉格朗日插值法求得的函数的导数。
    F = -37*X**3/3750+29347*X**2/500-1396607*X/12+923101735/12
    return F
 
def Lagrange(arr_x, arr_y,x):#此函数定义拉格朗日插值法。
    arr_x_len = len(arr_x)
    result = 0.0
    down = np.arange(0, arr_x_len)#down与up代表存储计算后的分母与分子的数组。
    up = np.arange(0, arr_x_len)
    for k in range(0, arr_x_len):#通过两层循环遍历arr_x中数组的值
        fenmu = 1.0 
        fenzi = 1.0
        for i in range(0, arr_x_len):
            if k!=i: #通过判断语句跳过（Xi-Xi）
                fenmu  = fenmu*(arr_x[k]-arr_x[i])# 通过累乘将分子分母全部计算并存于数组up与down中。
                fenzi  = fenzi*(x-arr_x[i])
        down[k] = fenmu
        up[k] = fenzi
    for m in range(0, arr_x_len):  #最后通过一个循环使得计算好的分子分母进行组合并与arr_y对应相乘
         result += up[m]*arr_y[m]/down[m]    #组成拉格朗日基函数，再将其累加得到结果。
    """
    (x-1975)(x-1980)(x-1985)(x-1990)*830/15000+
    (x-1970)(x-1980)(x-1985)(x-1990)*924/-3750+
    (x-1970)(x-1975)(x-1985)(x-1990)*987/2500+
    (x-1970)(x-1975)(x-1980)(x-1990)*1059/-3750+
    (x-1970)(x-1975)(x-1980)(x-1985)*1143/15000
    """
   # print("(X-1975)(X-1980)(X-1985)(X-1990)*%f/%f+(X-1970)(X-1980)(X-1985)(X-1990)*%f/%f+(X-1970)(X-1975)(X-1985)(X-1990)*%f/%f+(X-1970)(X-1975)(X-1980)(X-1990)*%f/%f+(X-1970)(X-1975)(X-1980)(X-1985)*%f/%f"%(arr_y[0],down[0],arr_y[1],down[1],arr_y[2],down[2],arr_y[3],down[3],arr_y[4],down[4]))
    return result



"""
     0      1      2      3      4      5
0  1970  830    
1  1975  924     18.8
2  1980  987     12.6   -0.62
3  1985  1059    14.4   0.18   0.053
4  1990  1143    16.8   0.24   0.004   0.00246
"""
def Newton(arr_x, arr_y,x):#此函数定义牛顿插值法。
    arr_x_len = len(arr_x)#table用于存储差商表，c代表Ci， x_cha代表（X-Xi）累乘后的结果。
    table = np.zeros([arr_x_len,arr_x_len+1])
    c = np.arange(0, arr_x_len,dtype = float)
    x_cha = np.arange(0, arr_x_len,dtype = float)
    ci = 0
    for i in range(0, arr_x_len):#此循环目的是将已知点的x,y值填入差商表。
        table[i][0] = arr_x[i]
        table[i][1] = arr_y[i]
    #for m in range(1, 2):#阶arr_x_len-1
    for l in range(2, arr_x_len+1):#此部分通过两层循环计算差商表，也是此函数最重要的内容，
        m = l-1  #参数m的作用是表示在计算几阶差商，并通过m对每阶的计算方法进行调整。
        for h in range(l-1, arr_x_len):
            fenzi = (table[h-1][l-1] -table[h][l-1])
            fenmu =  (table[h-m][l-(m+1)] -table[h][l-(m+1)])
            reslt = fenzi/fenmu
            table[h][l] =  reslt
    #print(table)
    for h in range(0, arr_x_len+1):#此循环目的在于将差商表中的Ci提取出来，
        for l in range(0, arr_x_len+1):# 也就是对角线上的值，需要注意C0应该为1。
            if h+1 == l:
                c[ci] = table[h][l]
                ci =ci+1
    x_cha[0] = 1
    for n in range(1, arr_x_len):#此循环目的在于将每个基函数的（X-Xi）累乘后的结果计算出来并存储。
        relut = 1.0
        for o in range(1,n+1):
            relut =relut*(x-arr_x[o-1])
        x_cha[n] = relut
    out = 0.0
    for n in range(0, arr_x_len-1):#最后通过一个循环使得计算好的x_cha与Ci对应相乘，
        out =  out + x_cha[n]*c[n]   #组成基函数，再将其累加得到结果。
    """
    830.000000+
    18.800000*(X-1970)+
    -0.620000*(X-1970)(X-1975)+
    0.053333*(X-1970)(X-1975)(X-1980)+
    -0.002467*(X-1970)(X-1975)(X-1980)(X-1985)
    """
    #print("%f+%f*(X-1970)+%f*(X-1970)(X-1975)+%f*(X-1970)(X-1975)(X-1980)+%f*(X-1970)(X-1975)(X-1980)(X-1985)"%(c[0],c[1],c[2],c[3],c[4]))
    
    return out



"""
    x_arr = [1970,1975,1980,1985,1990]
    y_arr = [830,924,987,1059,1143]

构建3次埃尔米特拟合函数f(x) = -37*x**4/15000+29347*x**3/1500-1396607*x**2/24+923101735*x/12-38133121326
f(1970) = 830  f(1975) = 924  f(1980) =987   f'(1975) = 13.75

   0      1       2        3        4     
   1970   830    
   1975   924    ( )  
   1975   924    13.75    ( )
   1980   987    ( )      ( )    ( )


"""
def hermite3(x):#此函数定义埃尔米特插值法。
    plt.subplot(1, 2, 1)
    arr_x_len = int(input("请输入列的值的个数:"))
    arr_x = np.arange(0, arr_x_len,dtype = float)
    arr_y = np.arange(0, arr_x_len,dtype = float)
    table = np.zeros([arr_x_len,arr_x_len+1])
    c = np.arange(0, arr_x_len,dtype = float)
    x_cha = np.arange(0, arr_x_len,dtype = float)
    ci = 0
    i = 0
    for i in range(0,arr_x_len):
        table_l1 = float(input("请输入第一列的第%d个值:"%i))#由于不能确定输入值与导数值的类型
        arr_x[i] = table_l1 ##如三个函数值与一个导数值或者是两个函数值与两个导数值，
    for j in range(0,arr_x_len):  #所以需要自己构建重节点差商表的表头并将导数值填写到表中对应位置
        table_l2 = float(input("请输入第二列的第%d个值:"%j)) #最后根据作者输入的内容构建重节点差商表。
        arr_y[j] = table_l2
    for i in range(0, arr_x_len):
        table[i][0] = arr_x[i]
        table[i][1] = arr_y[i]
    df_num = int(input("请输入导数值的值的个数:"))
    for i in range(0,df_num):
        df_j = int(input("请输入第%d个导数值的阶数:"%i))
        df_h = int(input("请输入第%d个导数值的所处行:"%i))
        df = float(input("请输入第%d个导数值的值:"%i))
        table[df_h][df_j+1] = df
    for l in range(2, arr_x_len+1):#此部分通过两层循环计算差商表，也是此函数最重要的内容，
        m = l-1              #参数m的作用是表示在计算几阶差商，并通过m对每阶的计算方法进行调整。
        for h in range(l-1, arr_x_len):
            if table[h][l] == 0:
                fenzi = (table[h-1][l-1] -table[h][l-1])
                fenmu =  (table[h-m][l-(m+1)] -table[h][l-(m+1)])
                reslt = fenzi/fenmu
                table[h][l] =  reslt
    print(table)
    for h in range(0, arr_x_len+1):#此循环目的在于将差商表中的Ci提取出来，
        for l in range(0, arr_x_len+1): #也就是对角线上的值，需要注意C0应该为1。
            if h+1 == l:
                c[ci] = table[h][l]
                ci =ci+1
    x_cha[0] = 1
    for n in range(1, arr_x_len):##此循环目的在于将每个基函数的（X-Xi）累乘后的结果计算出来并存储。
        relut = 1.0
        for o in range(1,n+1):
            relut =relut*(x-arr_x[o-1])
        x_cha[n] = relut
    out = 0.0
    for n in range(0, arr_x_len-1):##最后通过一个循环使得计算好的x_cha与Ci对应相乘，
        out =  out + x_cha[n]*c[n]  #组成基函数，再将其累加得到结果。
    
    x_cha[0] = 1
    for k in range(1965, 1995):#此部分循环与上面两个循环作用相同，目的在于将拟合好的函数画出来
        for n in range(1, arr_x_len):#，x的范围被规定在0.1-5之间，并将计算好的结果画在画布上。
            relut = 1.0
            for o in range(1,n+1):
                relut =relut*(k-arr_x[o-1])
            x_cha[n] = relut
        pout = 0.0
        for n in range(0, arr_x_len-1):
            pout =  pout + x_cha[n]*c[n]      
        plt.scatter(k,pout, color="blue")
        if k==year[i]:
            Rf[i]= abs( people[i]-pout)
            print("R(f) = %f"%(abs( people[i]-pout)))
            i = i+1
        print(k,pout)
    plt.subplot(1, 2, 2)
    plt.plot(Rf)
    print(Rf)
    return out 
        
"""


用y = a0 + a1x +a2x**2.来拟合下表函数，w权重取1
用二次多项式拟合
f0 =        [ 1  1  1 1 ]
f1 = x =    [ 1  2  3  4 ]
f2 = x **2 = [ 1  4  9 16 ]
y         =[  4 10 18 26 ]
w         =[  1  1  1  1 ]
|wf0f0  wf1f0  wf2f0 |  a0  |  f0y
|wf0f1  wf1f1  wf2f1 |  a1  |  f1y
|wf0f2  wf1f2  wf2f2 |  a2  |  f2y

/4    10  30   /  a0  /  58
/10   30  100  /  a1  /  182
/30   100  354 /  a2  /  622
"""
def dotw(x,y,w):#此函数定义带权值的内积
    x_len = len(x)
    y_len = len(y)
    w_len = len(w)
    out = 0.0
    if x_len == y_len== w_len:#此判断防止参数长度不一致
        for i in range(0, x_len):#此循环在计算带权值内积
            out = out+x[i]*y[i]*w[i]
    return out

def polyfit_2(arr_x, arr_y,w,x): #此函数定义利用最小二乘法拟合二次多项式。
    n = 0
    plt.subplot(1, 2, 1)
    arr_x_len = len(arr_x)
    table = np.zeros([3,3])##table用于存储计算好的正则方程组，fy_w用于存储与y内积后的结果。
    fy_w = np.arange(0, arr_x_len-1,dtype = float)
    f0 = np.arange(0, arr_x_len,dtype = float)
    f1 = np.arange(0, arr_x_len,dtype = float)
    f2 = np.arange(0, arr_x_len,dtype = float)
    fy = np.arange(0, arr_x_len,dtype = float)
    for i in range(0, arr_x_len):#此循环用于构建函数族。
        f0[i] = 1       
        f1[i] = arr_x[i]
        f2[i] = arr_x[i]*arr_x[i]
        fy[i]= arr_y[i]
    table[0][0] = dotw(f0,f0,w);table[0][1] = dotw(f1,f0,w); #此部分用于将各个函数族内积后的结果存在正则方程中。
    table[0][2] = dotw(f2,f0,w);fy_w[0] = dotw(f0,fy,w);
    table[1][0] = dotw(f0,f1,w);table[1][1] = dotw(f1,f1,w);
    table[1][2] = dotw(f2,f1,w);fy_w[1] = dotw(f1,fy,w);
    table[2][0] = dotw(f0,f2,w);table[2][1] = dotw(f1,f2,w);  
    table[2][2] = dotw(f2,f2,w);fy_w[2]= dotw(f2,fy,w);       
    print(table)
    solo=np.linalg.solve(table,fy_w)#np.linalg.solve(table,fy_w)是np中求解线性方程组的函数，
    #解出结果后该二次多项式已经求解完成。
    print("%f + %f*X + %f*X^2" %(solo[0],solo[1],solo[2]))
    out = solo[0]+solo[1]*x+solo[2]*x*x
    for i in range(1970, 1986):#此循环目的在于将拟合出来的函数画在画布上。
        out_p = solo[0]+solo[1]*i+solo[2]*i*i
        plt.scatter(i,out_p, color="blue")
        print(i,out_p)
        if i==year[n]:
            Rf[n]= abs( people[n]-out_p)
            print("R(f) = %f"%(abs( people[n]-out_p)))
            n = n+1
    plt.subplot(1, 2, 2)
    plt.plot(Rf)
    print(Rf)
    return out 





"""
某栋别墅售价500万，首付20%，余下部分可分期付款，
10年付清，每年付款数相同，若年贴现率为6%，
按连续贴现计算，利用数值积分方法计算每年应付款多少万元
       9
x =  ( f  500*0.8*0.06*((1-0.06)^x) dx  +  500*0.8)/10
       0
"""
    

def getSumF(sta,end,add):#此函数针对复化函数内的f =  (1-0.06)**(i+add)累加
    """                #参数是开始和结束,而add参数是为了辛普胜算中点而特意存在
    a = 0
    b = 9
    n = 9
    h = (9-0)/n = 1
    F = (1-0.06)^x
    """
    sum = 0.0
    for i in range(sta, end):
        value = (1-0.06)**(i+add)
        sum = sum+value
    
    return sum
    
def CompoundingSimpson():#此函数针对复化辛普胜函数
    """
                        9
x =  (  500*0.2*0.06*  f ((1-0.06)^x) dx  +  500*0.2)/10
                       0
    n取9
    h= 1
    x0 =0 x1=1 x2=2x3=3 x4=4  x5=5 x6=6 x7=7  x8=8 x9=9
    """
    a = 0
    b = 9
    n = 9
    h = (b-a)/n
    Sn =(h/6)*((1-0.06)**(a) +(1-0.06)**(b) +2*getSumF(1,n-1,0)+4* getSumF(0,n-1,h/2 ))
    
    x = (  500*0.8*0.06*Sn +  500*0.8)/10  #上面Sn只是计算了积分部分，此部分才是全部
    print(x)
    return x

def CompoundingTrapezium():#此函数定义复化梯形
    """
                        9
x =  (  500*0.2*0.06*  f ((1-0.06)^x) dx  +  500*0.2)/10
                       0
    n取9
    h= 1
    x0 =0 x1=1 x2=2x3=3 x4=4  x5=5 x6=6 x7=7  x8=8 x9=9
    """
    a = 0
    b = 9
    n = 9
    h = (b-a)/n
    Sn =(h/2)*((1-0.06)**(a) +(1-0.06)**(b) +2*getSumF(1,n-1,0))
    x = (  500*0.8*0.06*Sn +  500*0.8)/10 #上面Sn只是计算了积分部分，此部分才是全部
    print(x)
    return x


"""
针对人口变化规律建立常微分方程模型，可通过问题1中的数据得到方程参数，
使用常微分方程数值解法对人口进行预测，并与1的预测结果进行比较。
"""

def getKR(arr_x,arr_y,w):  #此函数求常微分方程组中的参数，采用的时最小二乘法
    #最小二乘法 1970  830   #此部分的代码就不写了，这个和上面的最小二乘法是类似的
                            #区别在于上面是是求二次的拟合，这里是一次的拟合，毕竟只有两个参数
    arr_x_len = len(arr_x)
    table = np.zeros([2,2])#正则方程组
    
    fy_w = np.arange(0, arr_x_len-2,dtype = float)
    f0 = np.arange(0, arr_x_len,dtype = float)
    f1 = np.arange(0, arr_x_len,dtype = float)
    fy = np.arange(0, arr_x_len,dtype = float)
    for i in range(0, arr_x_len):
        f0[i] = 1
        f1[i] = 1/(arr_x[i]-1970)
        fy[i]= math.log(arr_y[i],math.e)
    table[0][0] = dotw(f0,f0,w);table[0][1] = dotw(f1,f0,w);fy_w[0] = dotw(f0,fy,w);
    table[1][0] = dotw(f0,f1,w);table[1][1] = dotw(f1,f1,w);fy_w[1] = dotw(f1,fy,w);
    
    solo=np.linalg.solve(table,fy_w)
    print("%f + %f*X " %(solo[0],solo[1] ))
    #print(math.e**(solo[0]+solo[1]* 1/(1985-1970)))
    for i in range(1971, 1986):
        out_p = math.e**(solo[0]+solo[1]* 1/(i-1970))
        plt.scatter(i,out_p, color="blue")
        print(i,out_p)
    """
       ln(k) = A =6.921151  
            
             k= math.e**A
             1/(-r) = B = -0.181870
             r = -1/B
    """
    A=solo[0]
    B=solo[1]
    k= math.e**A
    r = -1/B
    print(k,r)
    return k,r

def df(k,r,x,y):#此函数定义的是人口模型的常微分方程
    y = math.log(y,math.e)#为什么要将xy这样处理呢，因为我在计算kr时将常微分方程进行了线性化
    x =  1/(x-1970)  #这就导致求得的结果与输入是我线性后的样子
    out = r*y*(1-(y/k))
    print(out)
    return out     
    
def euler_forward(minlimt,maxlimt,y0,h):#此函数定义向前欧拉法
    """
    Logistic 阻滞增长模型
    y' = r(1-y/k)y    带入1970  830求解
    y  = k/[1+((k/830) -1)*e^(-r(t-1970))]   同除系数(k/830) -1)，在取ln
    lny= ln(k) +  1/(-r)  *  1/(t-1970)
     Y =     A +    B          X
    """
    arr_x = [1971,1975,1980,1985]#此题的常微分方程采用Logistic 阻滞增长模型
    arr_y = [852,924,987,1059]
    w =[1,1,1,1]
    plt.subplot(1, 2, 1)
    k,r = getKR(arr_x,arr_y,w)#再求Logistic 阻滞增长模型的两个参数
    res = []
    xi = minlimt
    yi = y0
    i = 1
    while xi<=maxlimt: # 在求解区间范围
        y = yi + h*df(k,r,xi,yi)#公式
        plt.scatter(xi, yi, color="red")
        print('xi:{:.2f}, yi:{:.6f}'.format(xi,yi))
        if xi==year[i]:
            Rf[i] = abs( people[i]-yi)
            print("R(f) = %f"%(abs( people[i]-yi)))
            i = i+1
        res.append(y)
        xi, yi = xi+h, y
    plt.subplot(1, 2, 2)
    plt.plot(Rf)
    print(Rf)
    return res
"""
Yn+1 = Yn + h*f(Xn+1,Yn+1) 
==>  Yn+1 = Yn + h (  Yn+1- 2Xn+1/Yn+1  )   
     y(0) = 1

"""
def euler_backward(minlimt,maxlimt,y0,h,ero):#此函数定义向后欧拉法，也就是隐式的
    plt.subplot(1, 2, 1)
    arr_x = [1971,1975,1980,1985]
    arr_y = [852,924,987,1059]
    w =[1,1,1,1]
    k,r = getKR(arr_x,arr_y,w)#再求Logistic 阻滞增长模型的两个参数
    i=1
    res = []
    xi = minlimt
    xij1 = minlimt+h
    yi = y0
    yij1 = y0 + h*df(k,r,minlimt,y0) 
    i = 1
    res.append(y0)
    while xi<=maxlimt: # 在求解区间范围
        y_pre = yi + h*df(k,r,xij1,yij1)
        ero_pre = yij1
        
        while abs(y_pre-ero_pre)>ero:#与其他不同的原因是，利用迭代法求最优解
            ero_pre = y_pre
            y_pre = yi + h*df(k,r,xij1,y_pre)
        if xi==year[i]:
            Rf[i]= abs( people[i]-y_pre)
            print("R(f) = %f"%(abs( people[i]-y_pre)))
            i = i+1   
        res.append(y_pre)
        plt.scatter(xi, y_pre, color="red")
        print('xi:{:.2f}, yi:{:.6f}'.format(xi,y_pre))
        xi, yi = xi+h, y_pre
        xij1,yij1 = xi+h, y_pre + h*df(k,r,xi,yi) 
    plt.subplot(1, 2, 2)
    plt.plot(Rf)
    print(Rf)
    return res

def euler_modified(minlimt,maxlimt,y0,h,ero):#此函数改进的欧拉法
    arr_x = [1971,1975,1980,1985]
    arr_y = [852,924,987,1059]
    w =[1,1,1,1]
    i = 1
    plt.subplot(1, 2, 1)
    k,r = getKR(arr_x,arr_y,w)
    i = 1
    res = []
    xi = minlimt
    yi = y0

    while xi <= maxlimt: # 在求解区间范围
        yp = yi + h*df(k,r,xi, yi)
        y = yi + h/2 * (df(k,r,xi, yi) + df(k,r,xi, yp))
        print('xi:{:.2f}, yi:{:.6f}'.format(xi,yi))
        if xi==year[i]:
            Rf[i]= abs( people[i]-yi)
            print("R(f) = %f"%(abs( people[i]-yi)))
            i = i+1  
        plt.scatter(xi, yi, color="red")
        res.append(y)
        xi, yi = xi+h, y
    plt.subplot(1, 2, 2)
    plt.plot(Rf)
    print(Rf)
    return res
 



"""
在解决WLAN网络信道接入机制建模问题时，需求解关于碰撞概率非线性方程组

"""


def NewtonMethod():#此函数定义牛顿法
    p=0.1
    p_last = 0.0
    e = 0.0000001
    i = 0
    while abs(p-p_last)>e:
        b00 = (2 * (1 - p) * (1 - 2 * p)) / (        #
            16 * (1 - (2 * p) ** 7) * (1 - p) + (1 - 2 * p) * (
                1 - p ** 33) + 16 * 2 ** 6 * p ** 7 * (
                    1 - p ** 26) * (1 - 2 * p))
        F = p - b00 * (1 - p ** 33) / (1 - p)     #是将右侧移到左侧形成一个函数，变形为求根问题
        molecule = -4 * (1 - 3 * p) * (1 - (2 * p) ** 7) * (  #此部分是b00导数的分子
            1 - p) + 4 * (1 - p) * (1 - 2 * p) - 224 * (2 * p) ** 6 * (
                1 - p) * (1 - 2 * p) - 16 * (2 * p) ** 7 + 4 * (1 -2 * p) * (
                    1 - p ** 33) - 33 * p ** 32 * (1 - 2 * p)
        denominator = (16 * (1 - (2 * p) ** 7) * (1 - p) + (1 - 2 * p) * ( #此部分是b00导数的分母
            1 - p ** 33) + 16 * 2 ** 6 * p ** 7 * (1 - p ** 26) * (1 - 2 * p)) ** 2
        b00_dp = -1*molecule / denominator   #此部分是b00导数 
        F_dp = 1 + b00_dp * ((1 - p ** 33) / (1 - p) - p * 33 * p ** 32 / ( #此部分是F导数 
            1 - p) + p * p ** 33 / (1 - p) ** 2)
        p_last  = p
        p = p - F / F_dp#公式
        i=  i+1
        plt.scatter( i, p, color="red")
        print(p)
    return p
def getF(p): #此函数就是又求了一边F，目的是为了迎合二分法，减少代码量二分法
    b00 = (2 * (1 - p) * (1 - 2 * p)) / (
        16 * (1 - (2 * p) ** 7) * (1 - p) + (1 - 2 * p) * (
            1 - p ** 33) + 16 * 2 ** 6 * p ** 7 * (
                1 - p ** 26) * (1 - 2 * p))
    y = p - b00 * (1 - p ** 33) / (1 - p)
    return y
def DichotomyMethod():#此函数定义二分法，原理有点类似快速排序
    low =-0.1
    hig =0.1
    mid =low
    mid_lats = 0
    e = 0.0000001
    i =0
    while  getF(low)*getF(hig)>0:#两个相乘大于零，说明结果没在此区间，此函数目的扩大搜索区间
        low =low*2                  #因为我们取得区间是固定的，如果根不在此范围咋办，所以要扩大
        hig =hig*2
    while abs(mid-mid_lats)>e:
        i = i+1
        mid_lats = mid
        if getF(low)*getF(mid)<0:#两个相乘小于零，说明结果在此区间
            hig = mid
        else:#两个相乘大于零，说明结果没在此区间
            low = mid
        print("high:%f  low:%f " %(hig,low ))
        
        mid = (hig+low)/2
        plt.scatter(mid, i, color="red")
    print( mid)
    
    return mid



"""
q = q1i+q2j+q3k
R = []
归一化 R/（q0^2 + q1^2 +q2^2 +q3^2 ）^0.5
p2后 = R*p1 前

Q1 = 0.35 + 0.2i+0.3j+0.1k
Q2 = -0.5 + 0.4i + 0.1j + 0.2k
最后的光线方向为P = [0.5,0,0.2]T
求两次的方向向量


RQ1*X0 = X1 
RQ2*X1 = P
"""   
def getR(Q):#此函数就是为了构建题目中的R矩阵
    q0 = Q[0][0]
    q1 = Q[1][0]
    q2 = Q[2][0]
    q3 = Q[3][0]
    R = np.zeros([3,3],dtype = float)
    R[0][0] = (1-2*q2**2-2*q3**2) ;
    R[0][1] = (2*q1*q2-2*q0*q3) ;
    R[0][2] = (2*q1*q3+2*q0*q2) ;
    R[1][0] = (2*q1*q2+2*q0*q3) ;
    R[1][1] = (1-2*q1**2-2*q3**2) ;
    R[1][2] = (2*q2*q3-2*q0*q1) ;
    R[2][0] = (2*q1*q3-2*q0*q2) ;
    R[2][1] = (2*q2*q3+2*q0*q1) ;  
    R[2][2] = (1-2*q1**2-2*q2**2) ;  
    return R
def getNormalization(Q):#此函数目的在于将向量归一化
    q0 = Q[0][0]
    q1 = Q[1][0]
    q2 = Q[2][0]
    q3 = Q[3][0]
    normalization = (q0*q0 + q1*q1 +q2*q2 +q3*q3)**0.5#归一化公式
    Q[0][0] = Q[0][0]/normalization
    Q[1][0] = Q[1][0]/normalization
    Q[2][0] = Q[2][0]/normalization
    Q[3][0] = Q[3][0]/normalization
    return Q
def getLU(R):   #此次函数将R矩阵，分解成LR矩阵，然后求解其中的参数
    """
    1    0   0       U00   U01   U02       R00    R01    R02
    L10  1   0   *   0     U11   U12   =   R10    R11    R12 
    L20  L21 1       0     0     U22       R20    R21    R22
    
    U00 = R00  U01 = R01   U02 = R02
    L10*U00 = R10  L10*U01+U11 = R11  L10*U02+U12 = R12
    L20*U00 = R20  L20*U01+L21*U11 = R21   L20*U02+L21*U12+U22= R22  
    
    U00=R00         U01 = R01           U02 = R02
    L10=R10/U00    U11 = R11 - (L10*U01)  U12 = R12 -(L10*U02)
    L20 = R20/U00  L21 = (R21-(L20*U01))/U11   U22= R22-(L20*U02+L21*U12)
    """
    L = np.zeros([3,3],dtype = float)#根据注释中的推的结果体现
    U = np.zeros([3,3],dtype = float)
    L[0][0] = 1
    L[1][1] = 1
    L[2][2] = 1
    U[0][0] = R[0][0]    
    U[0][1] = R[0][1]           
    U[0][2] = R[0][2]
    L[1][0] = R[1][0]/U[0][0]    
    U[1][1] = R[1][1] - (L[1][0]*U[0][1])  
    U[1][2] = R[1][2] -(L[1][0]*U[0][2])        
    L[2][0] = R[2][0]/U[0][0 ] 
    L[2][1] = (R[2][1]-(L[2][0]*U[0][1]))/U[1][1]   
    U[2][2] = R[2][2]-(L[2][0]*U[0][2]+L[2][1]*U[1][2])      
    return L,U
def getXlp(L,P):#此函数目的在于我们将UX假设成为Y，因此我们要根据L,P求解Y
    # L*Y = P
    """
    1    0   0       Y00      P00
    L10  1   0   *   Y10   =  P10
    L20  L21 1       Y20      P20
    
    Y00 = P00  L10*Y00+Y10 = P10      L20*Y00 + L21*Y10 + Y20 = P20
    
    Y00 = P00  Y10 = P10 - (L10*Y00)  Y20 = P20 - (L20*Y00 + L21*Y10)
    """
    Y = np.zeros([3,1],dtype = float)
    Y[0][0] = P[0][0] 
    Y[1][0] = P[1][0] - (L[1][0]*Y[0][0]) 
    Y[2][0] = P[2][0] - (L[2][0]*Y[0][0] + L[2][1]*Y[1][0])
    return Y
def getXuy(U,Y):#此函数目的在于我们将UX假设成为Y，因此我们要根据U,Y求解X
    # U*X1 = Y
    """
       U00   U01   U02     X00        Y00
       0     U11   U12  *  X10   =    Y10 
       0     0     U22     X20        Y20
       
       U00*X00+ U01*X10 +U02*X20 = Y00
       U11*X10+U12*X20 = Y10
       U22*X20 = Y20
       
       X20 = Y20/U22
       X10 = (Y10 -(U12*X20))/U11
       X00 = (Y00-( U01*X10 +U02*X20))/U00
       
    """
    X = np.zeros([3,1],dtype = float)
    X[2][0]= Y[2][0]/U[2][2]
    X[1][0] = (Y[1][0] -(U[1][2]*X[2][0]))/U[1][1]
    X[0][0] = (Y[0][0]-( U[0][1]*X[1][0] +U[0][2]*X[2][0]))/U[0][0]
    return X
def TriangularFactorization3():#此函数定义三阶的三角分解法
    #RQ2*X1 = P
    #RQ1*X0 = X1 
    P = np.zeros([3,1],dtype = float)#将数据存进来
    P[0][0] = 0.5
    P[1][0] = 0
    P[2][0] = 0.2
    Q2 = np.zeros([4,1],dtype = float)
    Q2[0][0] = -0.5
    Q2[1][0] = 0.4
    Q2[2][0] = 0.1   
    Q2[3][0] = 0.2     
    Q2 = getNormalization(Q2)#归一化
    RQ2 = np.zeros([3,3],dtype = float)
    RQ2 = getR(Q2)#得到矩阵R
    L,U = getLU(RQ2)#得到L,U
    
    #RQ2*X1 = P  L*U*X1 = P  U*X1 = Y L*Y = P   U*X1 = Y
    Y = getXlp(L,P)#求解Y
    print(Y)
    #U*X1 = Y
    X1 = getXuy(U,Y)#求解X，但为什么叫X1呢，因为根据题目此次求解的结果是光线旋转第二次的结果，第一次就叫X0
    #RQ1*X0 = X1 
    Q1 = np.zeros([4,1],dtype = float)#以下就是在求解第一次旋转的光线，过程和上面一致便不在赘述
    Q1[0][0] = 0.35
    Q1[1][0] = 0.2
    Q1[2][0] = 0.3   
    Q1[3][0] = 0.1     
    Q1 = getNormalization(Q1)
    RQ1 = np.zeros([3,3],dtype = float)
    RQ1 = getR(Q1)
    L,U = getLU(RQ1)
    
    #RQ1*X0 = X1   L*U*X0 = X1  U*X0 = Y  L*Y = X1   U*X0 = Y
    Y = getXlp(L,X1)
    print(Y)
    #U*X0 = Y
    X0 = getXuy(U,Y)
    print(X0)
    print(X1)
    #print(RQ2@RQ1@X0)
    #soloX1=np.linalg.solve(RQ2,P)
    #soloX0=np.linalg.solve(RQ1,X1)
    #print(soloX0)
    #print(soloX1)
    return X0,X1


def get_MaxColumn(RTriUpAdd,col,sta,end):#此函数服务于最大行的高斯消元法
    """                #此函数的目的在于将某一列指定的范围内找到最大的一行，并将这一行换到范围内最高的一行
    R00 R01 R02 R03
    R10 R11 R12 R13
    R20 R21 R22 R23
    0.7826087   0.60869565  0.13043478  0.5 
   -0.26086957  0.13043478  0.95652174  0.  
    0.56521739 -0.7826087   0.26086957  0.2  
    """
    maxValue = abs(RTriUpAdd[sta][col])#随机指定最大值，我们要的是绝对值最大
    maxrow = 0
    temp = np.zeros([1,4],dtype = float)
    for i in range(sta, end+1):#此次循环找最大值#需要注意我们不要0，主对角有0会使得后续分母为0
        if (abs(RTriUpAdd[i][col])>maxValue)&(RTriUpAdd[i][col]!=0):
            maxValue = abs(RTriUpAdd[i][col])
            maxrow = i
    temp[0][0] =    RTriUpAdd[sta][0]#找到最大的换一下位置，将最大的那一行换到最前面
    temp[0][1] =    RTriUpAdd[sta][1]
    temp[0][2] =    RTriUpAdd[sta][2]
    temp[0][3] =    RTriUpAdd[sta][3]
    RTriUpAdd[sta][0] =  RTriUpAdd[maxrow][0]
    RTriUpAdd[sta][1] =  RTriUpAdd[maxrow][1]
    RTriUpAdd[sta][2] =  RTriUpAdd[maxrow][2]
    RTriUpAdd[sta][3] =  RTriUpAdd[maxrow][3]
    RTriUpAdd[maxrow][0] =  temp[0][0]
    RTriUpAdd[maxrow][1] =  temp[0][1]
    RTriUpAdd[maxrow][2] =  temp[0][2]
    RTriUpAdd[maxrow][3] =  temp[0][3]
    return col,maxrow,RTriUpAdd
 
def Gaussian_MaxColumn():#此函数定义最大行的高斯迭代法
    #RQ2*X1 = P
    #RQ1*X0 = X1 
    """
    [[ a[0][0]   a[0][1]  a[0][2]     x0     a[0][3]       ]
    [ 0.         a[1][1]  a[1][2]  *  x1     a[1][3]]
    [ 0.          0.      a[2][2]     x2     a[2][3]]]
    
    a[2][2]  *   x2   =   a[2][3]       
    a[1][1] *x1 + a[1][2]*x2  =  a[1][3]
    a[0][0] *x0 + a[0][1] *x1 + a[0][2]*x2  =  a[0][3]
    
      x2   =   a[2][3] /a[2][2]   
      x1   =  (a[1][3]- a[1][2]*x2)/ a[1][1] 
     x0  = ( a[0][3] - a[0][1] *x1 + a[0][2]*x2 )/a[0][0] 

     
    """    
    P = np.zeros([3,1],dtype = float)#把数据存进来
    P[0][0] = 0.5
    P[1][0] = 0
    P[2][0] = 0.2
    Q2 = np.zeros([4,1],dtype = float)
    Q2[0][0] = -0.5
    Q2[1][0] = 0.4
    Q2[2][0] = 0.1   
    Q2[3][0] = 0.2     
    Q2 = getNormalization(Q2)
    RQ2 = np.zeros([3,3],dtype = float)
    RQ2 = getR(Q2)#先归一化然后得到R
    RQ2TriUpAdd = np.zeros([3,4],dtype = float)
    RQ2TriUpAdd=np.append(RQ2,P,axis=1)# 0是往下，1是往右插入
       #与以往不同我们需要将R与P合在一起，目的是在接下来的操作中能够同步
    maxrow,maxcol ,RQ2TriUpAdd= get_MaxColumn(RQ2TriUpAdd,0,0,2)#找到最大行，并交换
    temp= np.zeros([2,4],dtype = float)
    for i in range(0, 4):#此循环将第二行下三角化为零，
        temp[0][i] =  RQ2TriUpAdd[1][i]- (RQ2TriUpAdd[0][i] *RQ2TriUpAdd[1][0]/RQ2TriUpAdd[0][0] )
        temp[1][i]=  RQ2TriUpAdd[2][i]- (RQ2TriUpAdd[0][i] *RQ2TriUpAdd[2][0]/RQ2TriUpAdd[0][0] )
    for i in range(0, 4):#此循环将第三行下三角化为零
        RQ2TriUpAdd[1][i] = temp[0][i] 
        RQ2TriUpAdd[2][i] = temp[1][i]
    print(RQ2TriUpAdd)
  
    maxrow,maxcol ,RQ2TriUpAdd= get_MaxColumn(RQ2TriUpAdd,1,1,2)
    temp= np.zeros([2,4],dtype = float)
    for i in range(1, 4):#为什么又来一遍，因为刚才是从第一列开始的，这次我们从第二列开始，
        temp[1][i]=  RQ2TriUpAdd[2][i]- (RQ2TriUpAdd[1][i] *RQ2TriUpAdd[2][1]/RQ2TriUpAdd[1][1] )
    for i in range(1, 4):
        RQ2TriUpAdd[2][i] = temp[1][i]    
    print(RQ2TriUpAdd)  

    X1 = np.zeros([3,1],dtype = float)   #根据最开始注释中推到的结果体现就好了
    X1[2][0]   =   RQ2TriUpAdd[2][3] / RQ2TriUpAdd[2][2]   
    X1[1][0]   =  (RQ2TriUpAdd[1][3]- RQ2TriUpAdd[1][2]*X1[2][0])/ RQ2TriUpAdd[1][1] 
    X1[0][0]  = ( RQ2TriUpAdd[0][3] - (RQ2TriUpAdd[0][1] *X1[1][0] + RQ2TriUpAdd[0][2]*X1[2][0]) )/RQ2TriUpAdd[0][0] 
   
    
    Q1 = np.zeros([4,1],dtype = float)#一下就是再来一次求解x0不在赘述
    Q1[0][0] = 0.35
    Q1[1][0] = 0.2
    Q1[2][0] = 0.3   
    Q1[3][0] = 0.1     
    
    Q1 = getNormalization(Q1)
    RQ1 = np.zeros([3,3],dtype = float)
    RQ1 = getR(Q1)
    RQ1TriUpAdd = np.zeros([3,4],dtype = float)
    RQ1TriUpAdd=np.append(RQ1,X1,axis=1)# 0是往下，1是往右插入
    
    
    maxrow,maxcol ,RQ1TriUpAdd= get_MaxColumn(RQ1TriUpAdd,0,0,2)
    temp= np.zeros([2,4],dtype = float)
    for i in range(0, 4):
        temp[0][i] =  RQ1TriUpAdd[1][i]- (RQ1TriUpAdd[0][i] *RQ1TriUpAdd[1][0]/RQ1TriUpAdd[0][0] )
        temp[1][i]=  RQ1TriUpAdd[2][i]- (RQ1TriUpAdd[0][i] *RQ1TriUpAdd[2][0]/RQ1TriUpAdd[0][0] )
    for i in range(0, 4):
        RQ1TriUpAdd[1][i] = temp[0][i] 
        RQ1TriUpAdd[2][i] = temp[1][i]
    #print(RQ2TriUpAdd)    
    maxrow,maxcol ,RQ1TriUpAdd= get_MaxColumn(RQ1TriUpAdd,1,1,2)
    
    temp= np.zeros([2,4],dtype = float)
    for i in range(1, 4):
        temp[1][i]=  RQ1TriUpAdd[2][i]- (RQ1TriUpAdd[1][i] *RQ1TriUpAdd[2][1]/RQ1TriUpAdd[1][1] )
    for i in range(1, 4):
        RQ1TriUpAdd[2][i] = temp[1][i]    
    #print(RQ2TriUpAdd)  
 
    X0 = np.zeros([3,1],dtype = float)
    X0[2][0]   =   RQ1TriUpAdd[2][3] / RQ1TriUpAdd[2][2]   
    X0[1][0]   =  (RQ1TriUpAdd[1][3]- RQ1TriUpAdd[1][2]*X0[2][0])/ RQ1TriUpAdd[1][1] 
    X0[0][0]  = ( RQ1TriUpAdd[0][3] - (RQ1TriUpAdd[0][1] *X0[1][0] + RQ1TriUpAdd[0][2]*X0[2][0]) )/RQ1TriUpAdd[0][0] 
   
    print(X0)
    print(X1)
    
    
    return X0,X1





def getDLU(R):#此函数目的在于求解在迭代法求解过程中需要用到的DLU
    """
             R00    R01    R02
             R10    R11    R12 
             R20    R21    R22
    """
    D = np.zeros([3,3],dtype = float)
    L = np.zeros([3,3],dtype = float)
    U = np.zeros([3,3],dtype = float)
    D[0][0]=R[0][0] 
    D[1][1]=R[1][1] 
    D[2][2]=R[2][2]   
    L[1][0]=-1*R[1][0] 
    L[2][0]=-1*R[2][0]    
    L[2][1]=-1*R[2][1] 
    U[0][1]=-1*R[0][1] 
    U[0][2]=-1*R[0][2]
    U[1][2]=-1*R[1][2]    
    return D,L,U
def Jacobi():#此函数定义雅可比迭代法
    # X_k+1 = B*Xk + f
    # A = D-L-U  
    # B = D逆(L+U)  f = D逆*b
    #RQ2*X1 = P
    P = np.zeros([3,1],dtype = float)#为什么没有采用题目的数据，因为我试了一下数据不对，我怀疑不收敛
    RQ2 = np.zeros([3,3],dtype = float)#也怀疑过我代码写错了，所以用一下数据测试一下（在调用那里的系统函数）
    RQ2[0][0] =10 ; RQ2[0][1] = -1; RQ2[0][2] =-2 ;
    RQ2[1][0] = -1; RQ2[1][1] = 10; RQ2[1][2] = -2; 
    RQ2[2][0] = -1; RQ2[2][1] = -2; RQ2[2][2] = 5; 
    P[0][0] = 7.2
    P[1][0] = 8.3
    P[2][0] = 4.2
    D,L,U = getDLU(RQ2)#获得DLU

    B = np.linalg.inv(D)@(L+U)   #求解B， np.linalg.inv(D)是求逆的，@是矩阵相乘
    f = np.linalg.inv(D)@P
    
    
    X1 = np.array([[0],[0],[0]])#随机指定X1
    Xk = B@X1+f#先算一次，目的是为了迭代
    while abs(Xk[0][0]-X1[0][0])>0.00005:
        # X_k+1 = B*Xk + f
        X1=Xk
        Xk= B@X1+f
    print(Xk)
    return Xk



def G_S():#此函数定义高斯塞尔德迭代法
    # X_k+1 = G*Xk + F
    # A = D-L-U  AX = b
    # G = (D-L)逆*U  F = (D-L)逆*b
    #RQ2*X1 = P
    P = np.zeros([3,1],dtype = float)#和雅可比迭代法重复部分不在追述
    RQ2 = np.zeros([3,3],dtype = float)
    RQ2[0][0] =10 ; RQ2[0][1] = -1; RQ2[0][2] =-2 ;
    RQ2[1][0] = -1; RQ2[1][1] = 10; RQ2[1][2] = -2; 
    RQ2[2][0] = -1; RQ2[2][1] = -2; RQ2[2][2] = 5; 
    P[0][0] = 7.2
    P[1][0] = 8.3
    P[2][0] = 4.2
    D,L,U = getDLU(RQ2)
    # G = (D-L)逆*U  F = (D-L)逆*b
    G = np.linalg.inv(D-L)@U #两种方法就是这里不同
    F = np.linalg.inv(D-L)@P 
 
    X1 = np.array([[0],[0],[0]])
    Xk = G@X1+F
    while abs(Xk[0][0]-X1[0][0])>0.00005:
        # X_k+1 = B*Xk + f
        X1=Xk
        Xk= G@X1+F
        
    print(Xk)
    return Xk



mode = 10
   
"""
1.插值与拟合
在国家统计局网站上搜索某省份（或全国）历年人口数据，
利用插值或拟合方法建立该地区人口随时间变化规律，
并进行预测。
"""

if mode==1:
    #拉格朗日  1994  119850
    
    x_arr = [1970,1975,1980,1985,1990]
    y_arr = [830,924,987,1059,1143]
    i=0
    plt.subplot(1, 2, 1)
    for n in range(1965, 1995):
         ypre = Lagrange(x_arr,y_arr,n)
         plt.scatter(n,ypre, color="blue")
         print(n,ypre)
         if n==year[i]:
             Rf[i]= abs( people[i]-ypre)
             print("R(f) = %f"%(abs( people[i]-ypre)))
             i = i+1
    plt.scatter(x_arr, y_arr, color="red")
    plt.subplot(1, 2, 2)
    plt.plot(Rf)
    print(Rf)
elif mode==2:
    #牛顿   1994  119850
    x_arr = [1970,1975,1980,1985,1990]
    y_arr = [830,924,987,1059,1143]
    i=0
    plt.subplot(1, 2, 1)
    for n in range(1965, 1995):
        ypre = Newton(x_arr,y_arr,n)
        plt.scatter(n,ypre, color="blue")
        print(n,ypre)
        if n==year[i]:
            Rf[i]= abs( people[i]-ypre)
            print("R(f) = %f"%(abs( people[i]-ypre)))
            i = i+1
    plt.scatter(x_arr, y_arr, color="red")
    plt.subplot(1, 2, 2)
    plt.plot(Rf)
    print(Rf)
    
elif mode==3:
    #埃尔米特   1974  90859
    """
    请输入列的值的个数:4
    请输入第一列的第0个值:1970
    请输入第一列的第1个值:1975
    请输入第一列的第2个值:1975
    请输入第一列的第3个值:1980
    请输入第二列的第0个值:830
    请输入第二列的第1个值:924
    请输入第二列的第2个值:924
    请输入第二列的第3个值:987
    请输入导数值的值的个数:1
    请输入第0个导数值的阶数:1
    请输入第0个导数值的所处行:2
    请输入第0个导数值的值:13.75
    """
    xpre = 1974
    ypre = hermite3(xpre)
   

elif mode==4:
    #最小二乘法 1974  90859
    xpre = 1974
    """ x_arr =[1,2,3,4]
    y_arr =[4,10,18,26]
    """
    x_arr = [1970,1975,1980,1985]
    y_arr = [830,924,987,1059]
    w_arr =[1,1,1,1]
    ypre = polyfit_2(x_arr,y_arr,w_arr,xpre)
    
    """
    2.数值积分与微分
    某栋别墅售价500万，首付20%，余下部分可分期付款，
    10年付清，每年付款数相同，若年贴现率为6%，
    按连续贴现计算，利用数值积分方法计算每年应付款多少万元。
    """
elif mode==5:

    #复化辛普生
    CompoundingSimpson()
elif mode==6:
    #复化梯形
    CompoundingTrapezium()
    
    """
    3.常微分方程数值算法
    针对人口变化规律建立常微分方程模型，
    可通过问题1中的数据得到方程参数，
    使用常微分方程数值解法对人口进行预测，
    并与1的预测结果进行比较。
    """   
elif mode==7:
    #向前欧拉法求微分方程
    euler_forward(1971,1985,852,1)

elif mode==9:
     #向后欧拉法求微分方程
     euler_backward(1971,1985,852,1,0.0000001)
elif mode==10:
     #向后欧拉法求微分方程
     euler_modified(1971,1985,852,1,0.0000001)  
     
     
     """
     4.非线性方程求解
     在解决WLAN网络信道接入机制建模问题时，
     需求解关于碰撞概率非线性方程组：
     试用数值方法求解碰撞概率。 
     """ 
elif mode==11:
 
    #二分法
    DichotomyMethod()
elif mode==12:
    #牛顿迭代法
    NewtonMethod()

    """
    5.线性方程组
    为避免“万向锁”问题的出现，我们常用四元数
    （其中为虚数单位)来描述物体的旋转，
    该四元数对应的旋转矩阵为：
    即旋转前坐标与旋转后坐标关系为：.
    现有一光线，光源经两次旋转，后，光线方向为
    试求两次旋转前光线的方向向量。
    （注：为保证为正交矩阵，需先将四元数归一化，
    即用四元数除以其长度。)
    """
elif mode==13:
    #最大行的高斯消元法
    Gaussian_MaxColumn()
elif mode==14:
    #雅可比与高斯赛德尔迭代法
    RQ2 = np.zeros([3,3],dtype = float)
    P = np.zeros([3,1],dtype = float)
    RQ2[0][0] =10 ; RQ2[0][1] = -1; RQ2[0][2] =-2 ;
    RQ2[1][0] = -1; RQ2[1][1] = 10; RQ2[1][2] = -2; 
    RQ2[2][0] = -1; RQ2[2][1] = -2; RQ2[2][2] = 5; 
    P[0][0] = 7.2
    P[1][0] = 8.3
    P[2][0] = 4.2
    solo=np.linalg.solve(RQ2,P)
    #print(solo)
    Jacobi()
    G_S()
elif mode==15:
    #三角分解法
    TriangularFactorization3()















