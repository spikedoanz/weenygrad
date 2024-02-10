import numpy as np
from weenygrad import ADVect, Neuron 

def test_ops():
    a = ADVect(np.arange(3))
    b = ADVect(np.arange(3)+3)
    c = ADVect(np.arange(3)+6)
    d = ADVect(np.arange(1)+6)

    e = c @ b
    f = e + d
    g = f.relu()
    g.grad = 1

    g._backward()
    f._backward()
    d._backward()
    e._backward()
    for v in [g,f,d,e,c,b]:
        print(v)

def test_backward():
    a = ADVect(np.arange(3))
    b = ADVect(np.arange(3)+3)
    c = ADVect(np.arange(3)+6)
    d = ADVect(np.arange(1)+6)

    e = c @ b
    f = e + d
    g = f.relu()
    g.backward()
    for v in [g,f,d,e,c,b]:
        print(v)

def test_neuron():
    a = Neuron(10)
    b = ADVect(np.arange(10))
    c = a(b)
    c.backward()
    print(c)
    print(a.b)
    print(a.w)
    print(b)

if __name__=='__main__':
    test_neuron()




