import numpy as np
from weenygrad import ADVect
import torch

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

def test_matrix_vector_multiply():
    A = np.arange(6)
    A = ADVect(A.reshape(3,2))
    b = ADVect(np.arange(2)+6)
    c = A @ b
    c.backward()
    Awg, cwg = A, c 

    A = np.arange(6)
    A = A.reshape(3,2)
    A = torch.Tensor(A)
    b = torch.Tensor(np.arange(2)+6)
    c = A @ b
    c.backward()
    Apt, cpt = A, c

    # forward pass is correct
    assert cwg.data == cpt.data.item()
    # backward pass is correct
    assert Awg.grad == Apt.grad.item()

def test_sanity():
    x = ADVect([-4.0])
    z = [2.0] @ x + [2.0] + x
    q = z.relu() + z @ x
    h = (z @ z).relu()
    #y = h + q + q @ x
    y = q @ x
    y.backward()
    xwg, ywg = x, y

    x = torch.Tensor([-4.0])
    x.requires_grad = True
    z = (torch.Tensor([2.0]) @ x )+ torch.Tensor([2.0]) + x
    q = z.relu() + z @ x
    h = (z @ z).relu()
    #y = h + q + q @ x
    y = q @ x
    y.backward()
    xpt, ypt = x, y

    # forward pass is correct
    assert ywg.data == ypt.data.item()
    # backward pass is correct
    assert xwg.grad == xpt.grad.item()

if __name__ == '__main__':
    test_sanity()
    #test_matrix_vector_multiply()
