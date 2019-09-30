import timeit

def FN():
    a=10
    b=20
    c=30
    print("A:%s" % (a))
    print("B:%s" % (b))
    print("B:%s" % (c))

vv = timeit('for x in range(100): print(x)', number=100)
print(vv)
