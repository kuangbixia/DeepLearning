# Python符号

## 1 *args

​	作为函数的参数，相当于是一个元组

```python
def foo(param1, *param2):
    print(param1)
    print(param2)
 

foo(1,2,3,4,5)
'''
1
(2,3,4,5)
'''
```



## 2 **kargs

​	作为函数的参数，相当于是一个对象

```python
def bar(param1, **param2):
    print(param1)
    print(param2)
    for param in param2:
        print(param)
    
bar(1, a=2, b=3, c=4)
'''
1
{'a':2, 'b':3, 'c':4}
a 2
b 3
c 4
'''
```



# Python函数

## 1 enumerate(sequence, [start=0])

- sequence：序列 / 迭代器 / 其他支持迭代的对象
- start：下标起始位置

# Python包

## 1 matplotlib 散点图

