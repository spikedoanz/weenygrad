import numpy as np
from weenygrad import ADVect
import torch

def test_addition():
    a = ADVect(np.arange(10))
    b = ADVect(np.arange(10)-14)
    c = a + b + b
    c.backward()
    awg, cwg = a, c

    a = torch.Tensor(np.arange(10))
    a.requires_grad = True
    b = torch.Tensor(np.arange(10)-14)
    c = a + b + b
    apt, cpt = a, c
    assert cwg.data == cpt.data.item()
    assert awg.grad == apt.grad.item()


def test_matrix_vector_multiply():
    A = np.arange(6)
    A = ADVect(A.reshape(3,2))
    b = ADVect(np.arange(2)+6)
    c = A @ b
    c.backward()
    d = c.sum()
    Awg, dwg = A, d

    A = np.arange(6)
    A = A.reshape(3,2)
    A = torch.Tensor(A)
    A.requires_grad = True
    b = torch.Tensor(np.arange(2)+6)
    c = A @ b
    d = c.sum()
    d.backward()
    Apt, dpt = A, d

    # forward pass is correct
    assert dwg.data == dpt.data.item()
    # backward pass is correct
    assert np.allclose(Awg.grad, Apt.grad.numpy())

def test_sanity(DEBUG=False):
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
    if DEBUG:
        print(xwg.grad)
        print(xpt.grad.item())

    assert ywg.data == ypt.data.item()
    assert xwg.grad == xpt.grad.item()

if __name__ == '__main__':
    test_addition
    test_matrix_vector_multiply()
    test_sanity() 
