#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
from numpy import ogrid


# In[6]:


np. __version__


# In[7]:


get_ipython().system('pip install --upgrade numpy')


# In[8]:


np.__version__


# In[9]:


import numpy as np


# In[10]:


np.__version__


# In[ ]:



#1
a = [4, 3, 5, 7, 6, 8]
indices = [0, 1, 4]
np.take(a, indices)
#array([4, 3, 6])

#2

x = np.array([[[0], [1], [2]]])
x.shape
(1, 3, 1)
np.squeeze(x).shape
(3,)
np.squeeze(x, axis=0).shape
(3, 1)

#3

np.all([[True,False],[True,True]])
False
np.all([[True,False],[True,True]], axis=0)
#array([ True, False])

#4

np.any([[True, False], [True, True]])
True
np.any([[True, False], [False, False]], axis=0)
#array([ True, False])

#5
x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
x
#array([[3, 0, 0],
#       [0, 4, 0],
 #      [5, 6, 0]])
np.nonzero(x)
#(array([0, 1, 2, 2]), array([0, 1, 0, 1]))


x[np.nonzero(x)]
array([3, 4, 5, 6])
np.transpose(np.nonzero(x))
#array([[0, 0],
       [1, 1],
       [2, 0],
       [2, 1]])

#6

a = np.arange(10)
a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.where(a < 5, a, 10*a)
array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])
This can be used on multidimensional arrays too:


np.where([[True, False], [True, True]],
         [[1, 2], [3, 4]],
         [[9, 8], [7, 6]])
#array([[1, 8],
#       [3, 4]])


#7

choices = [[0, 1, 2, 3], [10, 11, 12, 13],
  [20, 21, 22, 23], [30, 31, 32, 33]]
np.choose([2, 3, 1, 0], choices
# the first element of the result will be the first element of the
# third (2+1) "array" in choices, namely, 20; the second element
# will be the second element of the fourth (3+1) choice array, i.e.,
# 31, etc.
)
#array([20, 31, 12,  3])
np.choose([2, 4, 1, 0], choices, mode='clip') # 4 goes to 3 (4-1)
#array([20, 31, 12,  3])
# because there are 4 choice arrays
np.choose([2, 4, 1, 0], choices, mode='wrap') # 4 goes to (4 mod 4)
#array([20,  1, 12,  3])
# i.e., 0


#8

a = np.array([[1, 2], [3, 4], [5, 6]])
a
#array([[1, 2],
#       [3, 4],
#[5, 6]])
np.compress([0, 1], a, axis=0)
#array([[3, 4]])
np.compress([False, True, True], a, axis=0)
#array([[3, 4],
  #     [5, 6]])
np.compress([False, True], a, axis=1)
#array([[2],
     #  [4],
      #  [6]])


#9

a = np.array([1,2,3])
np.cumprod(a) # intermediate results 1, 1*2
              # total product 1*2*3 = 6
#array([1, 2, 6])
a = np.array([[1, 2, 3], [4, 5, 6]])
np.cumprod(a, dtype=float) # specify type of output
#array([   1.,    2.,    6.,   24.,  120.,  720.])


#10

a = np.array([1,2,3])
b = np.array([0,1,0])
np.inner(a, b)



#11

a = np.array([1, 2])
a.fill(0)
a
#array([0, 0])
a = np.empty(2)
a.fill(1)
a
#array([1.,  1.])


#12
a = np.array([1+2j, 3+4j, 5+6j])
a.imag
#array([2.,  4.,  6.])
a.imag = np.array([8, 10, 12])
a
#array([1. +8.j,  3.+10.j,  5.+12.j])
np.imag(1 + 1j)
1.0


#13

np.prod([1.,2.])
2.0

#14

a = np.arange(5)
np.put(a, [0, 2], [-44, -55])
a
array([-44,   1, -55,   3,   4])

#15

x = np.arange(6).reshape(2, 3)
np.putmask(x, x>2, x**2)
x
#array([[ 0,  1,  2],
       #[ 9, 16, 25]])

#16

a = np.array([1+2j, 3+4j, 5+6j])
a.real
#array([1.,  3.,  5.])
a.real = 9
a
#array([9.+2.j,  9.+4.j,  9.+6.j])
a.real = np.array([9, 8, 7])
a
#array([9.+2.j,  8.+4.j,  7.+6.j])
np.real(1 + 1j)
1.0


#17

np.sum([0.5, 1.5])
2.0
np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
1
np.sum([[0, 1], [0, 5]])
6
np.sum([[0, 1], [0, 5]], axis=0)
#array([0, 6])
np.sum([[0, 1], [0, 5]], axis=1)
#array([1, 5])
np.sum([[0, 1], [np.nan, 5]], where=[False, True], axis=1)#array([1., 5.])


#18

x = np.array([[0, 2], [1, 1], [2, 0]]).T
x
#array([[0, 1, 2],
     #  [2, 1, 0]])


#19

a = np.array([[1, 2], [3, 4]])
np.mean(a)
2.5
np.mean(a, axis=0)
#array([2., 3.])
np.mean(a, axis=1)
#array([1.5, 3.5])

#20

a = np.array([[1, 2], [3, 4]])
np.std(a)
1.1180339887498949 # may vary
np.std(a, axis=0)
#array([1.,  1.])
np.std(a, axis=1)
#array([0.5,  0.5])

#21

a = np.array([[1, 2], [3, 4]])
np.var(a)
1.25
np.var(a, axis=0)
#array([1.,  1.])
np.var(a, axis=1)
#array([0.25,  0.25])

#22

x = [1, 2, 3]
y = [4, 5, 6]
np.cross(x, y)
#array([-3,  6, -3])
#One vector with dimension 2.

x = [1, 2]
y = [4, 5, 6]
np.cross(x, y)
#array([12, -6, -3])



#23
np.dot([2j, 3j], [2j, 3j])
(-13+0j)

#24

rl = np.outer(np.ones((5,)), np.linspace(-2, 2, 5))
rl
#array([[-2., -1.,  0.,  1.,  2.],
     #  [-2., -1.,  0.,  1.,  2.],
      #  [-2., -1.,  0.,  1.,  2.],
       # [-2., -1.,  0.,  1.,  2.],
       # [-2., -1.,  0.,  1.,  2.]])
im = np.outer(1j*np.linspace(2, -2, 5), np.ones((5,)))
im
#array([[0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j, 0.+2.j],
      # [0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j, 0.+1.j],
      #  [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
      # [0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j, 0.-1.j],
       #[0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j, 0.-2.j]])
grid = rl + im
grid
#array([[-2.+2.j, -1.+2.j,  0.+2.j,  1.+2.j,  2.+2.j],
     #  [-2.+1.j, -1.+1.j,  0.+1.j,  1.+1.j,  2.+1.j],
      # [-2.+0.j, -1.+0.j,  0.+0.j,  1.+0.j,  2.+0.j],
      # [-2.-1.j, -1.-1.j,  0.-1.j,  1.-1.j,  2.-1.j],
       #[-2.-2.j, -1.-2.j,  0.-2.j,  1.-2.j,  2.-2.j]])

#25

a = np.array([1+2j,3+4j])
b = np.array([5+6j,7+8j])
np.vdot(a, b)
(70-8j)
np.vdot(b, a)
(70+8j)



#26

np.array([1,2,3])

#27
np.arange(3)

#28
x= array ([3,4,5])
    y=x
        z.np.copy

#29

np.empty ([4,6])
   # array ([[-9.744 , 6.69 ] , [2,.13 , 3.06]])

#30

a=([1,2,3] ,[4,5,6])
    np.empty_like(a)
      #  array([[-1073 , -1078 , 3]])

#31

np.eye(2, dtype=int)
    #array ([1,0],[0,1])

# 32

np.identity(3)
  #  array([1. , 0. , 0.] , [0., 1. , 0.] , [0. , 0., 1.])

#33

np.linspace(2.0, 3.0, num=5)
   # array([2.  , 2.25, 2.5 , 2.75, 3.  ])

# 34

np.logspace(2.0, 3.0, num=4)
   # array([ 100.        ,  215.443469  ,  464.15888336, 1000. 
       
# 35

np.mgrid[0:5,0:5]

#array([[[0, 0, 0, 0, 0],
 #       [1, 1, 1, 1, 1],
  #      [2, 2, 2, 2, 2],
   #     [3, 3, 3, 3, 3],
    #    [4, 4, 4, 4, 4]],
     #  [[0, 1, 2, 3, 4],
      #  [0, 1, 2, 3, 4],
        #[0, 1, 2, 3, 4],
            #[0, 1, 2, 3, 4],
           # [0, 1, 2, 3, 4]]])

       
# 36

from numpy import ogrid
    ogrid[-1:1:5j]
     #  array([-1. , -0.5,  0. ,  0.5,  1. ])
           
# 37
           
np.ones(5)
# array([1., 1., 1., 1., 1.])
           
# 38
           
x = np.arange(6)
    x = x.reshape((2, 3))
           x
          # array([[0, 1, 2],  [3, 4, 5]])
           np.ones_like(x)
           
# 39

np.zeros(5)
3array([ 0.,  0.,  0.,  0.,  0.])
           
# 40
y = np.arange(3, dtype=float)
    y
        np.zeros_like(y)
         #  array([0., 1., 2.])

  
# 41
           x = np.array([1, 2, 2.5])
           x
       #array([1. ,  2. ,  2.5])
           
# 42

x = np.arange(9.0).reshape(3,3)
np.atleast_1d(x)
     #array([[0., 1., 2.],
      # [3., 4., 5.],
       # [6., 7., 8.]])
           
# 43

x = np.arange(3.0)
    np.atleast_2d(x)
        #array([[0., 1., 2.]])
        
        np.atleast_2d(x).base is x
        # True

           
# 44

x = np.arange(3.0)
    np.atleast_3d(x).shape
    #(1, 3, 1)

           
# 45

x = np.array([[1, 2], [3, 4]])
    m = np.asmatrix(x)
        x[0,0] = 5
        m
        
        #matrix([[5, 2],
        #[3, 4]])
           
# 46

x = np.arange(8.0)
    np.array_split(x, 3)
        #[array([0.,  1.,  2.]), array([3.,  4.,  5.]),
        #array([6.,  7.])]
           
# 47
    
a = np.array((1,2,3))
  b = np.array((2,3,4))
    np.column_stack((a,b))
    
      #  array([[1, 2],
      # [2, 3],
      # [3, 4]])
  
    

# 48

a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6]])
        np.concatenate((a, b), axis=0)
        
         #   array([[1, 2],
          #   [3, 4],
            #  [5, 6]])
        

# 49


a = np.arange(4).reshape(2,2)
    a
    
      #  array([[0, 1],
       # [2, 3]])
    

# 50
 x = np.arange(16.0).reshape(2, 2, 4)
    x    
    
  #  array([[[ 0.,   1.,   2.,   3.],
       # [ 4.,   5.,   6.,   7.]],
       # [[ 8.,   9.,  10.,  11.],
        #[12.,  13.,  14.,  15.]]])

        
    

