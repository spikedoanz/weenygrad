import numpy as np
from weenygrad import weenygrad as wg
import torch


def test_addition():
    a = wg.ADVect(np.arange(10))
    b = wg.ADVect(np.arange(10)-14)
    c = a + b + b
    c = c.sum()
    c.backward()
    awg, cwg = a, c

    a = torch.Tensor(np.arange(10))
    a.requires_grad = True
    b = torch.Tensor(np.arange(10)-14)
    c = a + b + b
    c = c.sum()
    c.backward()
    apt, cpt = a, c
    assert np.allclose(cwg.data, cpt.data.numpy())
    assert np.allclose(awg.grad, apt.grad.numpy())


def test_multiplication():
    a = wg.ADVect(np.arange(1, 11))  # start from 1 to avoid zero multiplication
    b = wg.ADVect(np.arange(11, 21))
    c = a * b
    c = c.sum()
    c.backward()
    awg, cwg = a, c

    a = torch.Tensor(np.arange(1, 11))  # start from 1 to avoid zero multiplication
    a.requires_grad = True
    b = torch.Tensor(np.arange(11, 21))
    c = a * b
    c = c.sum()
    c.backward()
    apt, cpt = a, c

    assert np.allclose(cwg.data, cpt.data.numpy())
    assert np.allclose(awg.grad, apt.grad.numpy())


def test_matrix_vector_multiply():
    A = np.arange(6)
    A = wg.ADVect(A.reshape(3,2))
    b = wg.ADVect(np.arange(2)+6)
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

def test_division():
    a = wg.ADVect(np.arange(1, 11))  # start from 1 to avoid zero division
    b = 3
    c = a / b
    c.backward()
    c = c.sum()
    awg, cwg = a, c

    a = torch.Tensor(np.arange(1, 11))  # start from 1 to avoid zero division
    a.requires_grad = True
    b = 3
    c = a / b
    c = c.sum()
    c.backward()
    apt, cpt = a, c

    assert np.allclose(cwg.data, cpt.data.numpy())
    assert np.allclose(awg.grad, apt.grad.numpy())

def test_log():
    a = wg.ADVect(np.arange(1, 11))  # start from 1 to avoid log(0)
    c = a.log()
    c = c.sum()
    c.backward()
    awg, cwg = a, c

    a = torch.Tensor(np.arange(1, 11))  # start from 1 to avoid log(0)
    a.requires_grad = True
    c = torch.log(a)
    c = c.sum() 
    c.backward()
    apt, cpt = a, c
    assert np.allclose(cwg.data, cpt.data.numpy())
    assert np.allclose(awg.grad, apt.grad.numpy())

def test_sanity(DEBUG=False):
    x = wg.ADVect([-4.0])
    z = [2.0] @ x + [2.0] + x
    q = z.relu() + z @ x
    h = (z @ z).relu()
    y = q @ x
    y.backward()
    xwg, ywg = x, y

    x = torch.Tensor([-4.0])
    x.requires_grad = True
    z = (torch.Tensor([2.0]) @ x )+ torch.Tensor([2.0]) + x
    q = z.relu() + z @ x
    h = (z @ z).relu()
    y = q @ x
    y.backward()
    xpt, ypt = x, y
    if DEBUG:
        print(xwg.grad)
        print(xpt.grad.item())

    assert ywg.data == ypt.data.item()
    assert xwg.grad == xpt.grad.item()

if __name__ == '__main__':
    test_addition()
    test_multiplication()
    test_log()
    test_division()
    test_matrix_vector_multiply()
    test_sanity(True) 
    
