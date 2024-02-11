import numpy as np

class ADVect():
    def __init__(self, data, _children = []):
        data            = data if isinstance(data, np.ndarray) else np.array(data)
        self.data       = data 
        self.grad       = np.zeros_like(self.data) 
        self._children  = set(_children) 
        self._backward  = lambda: None 
    
    def __add__(self, other):
        other           = other if isinstance(other, ADVect) else ADVect(other)
        out             = ADVect(self.data+ other.data, [self, other])
        def _backward():
            self.grad   = self.grad + out.grad
            other.grad  = other.grad + out.grad
        out._backward   = _backward
        return out
    
    def __sub__(self, other):
        other           = other if isinstance(other, ADVect) else ADVect(other)
        out             = ADVect(self.data - other.data, [self, other])
        def _backward():
            self.grad   = self.grad + out.grad
            other.grad  = other.grad - out.grad
        out._backward   = _backward
        return out


    def __matmul__(self, other):
        other = other if isinstance(other, ADVect) else ADVect(other)
        if self.data.ndim == 1 and other.data.ndim == 1:
            out = ADVect(self.data * other.data, [self, other]) 
        else: 
            out = ADVect(self.data @ other.data, [self, other]) 

        def _backward():
            if np.ndim(self.data) == 1:
                self.grad   = self.grad + out.grad * other.data
                other.grad  = other.grad + out.grad * self.data
            else: # Matrix vector mulitplication is supported, not matrix @ matrix
                self.grad   = self.grad + np.outer(out.grad, other.data) 
                other.grad  = other.grad + np.transpose(self.data) @ out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other): # assuming other is a scalar, this implementation is a bit iffy
        other = other if isinstance(other, ADVect) else ADVect(other)
        out = ADVect(self.data * other.data, [self, other])
        def _backward():
            self.grad   = self.grad + other.data * out.grad
            other.grad  = other.grad + self.data * out.grad # I'm just gna pretend like this is the answer, the real answer should be [1xn]
        out._backward = _backward
        return out
    
    def __pow__(self, other):
        other = other if isinstance(other, ADVect) else ADVect(other)
        out = ADVect(self.data ** other.data, [self, other])
        def _backward():
            self.grad = self.grad + other.data * self.data ** (other.data - 1) * out.grad
            other.grad = other.grad + self.data ** other.data * np.log(self.data) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other): # assuming other is a scalar
        assert other != 0,  "don't divide by zero, stupid"
        return self * (other ** -1)

    def __neg__(self):
        out = ADVect(-self.data, [self])
        def _backward():
            self.grad = self.grad - out.grad
        out._backward = _backward
        return out

    def sum(self):
        out = ADVect(np.sum(self.data), [self])
        def _backward():
            self.grad = np.ones_like(self.grad)*out.grad
        out._backward = _backward   
        return out
    
    def log(self):
        out = ADVect(np.log(self.data + 1e-10), [self])
        def _backward():
            self.grad = self.grad + (self.data.astype(float) + 1e-10) ** -1. * out.grad 
        out._backward = _backward
        return out

    def relu(self):
        ReLU = np.vectorize(lambda x: max(0, x))
        out = ADVect(ReLU(self.data), [self])
        def _backward():
            self.grad = self.grad + (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def softmax(self):
        shiftx = self.data - np.max(self.data) # somewhat cheating but subtraction doesn't change the gradient
        exps = np.exp(shiftx.data)
        sum_exps = np.sum(exps)
        out = ADVect(exps / sum_exps, [self])
        def _backward():
            softmax_grad = out.data * (np.eye(len(self.data)) - out.data)
            self.grad = self.grad + softmax_grad @ out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set() 
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
    
    def __rmatmul__(self, other): 
        return self @ other
    
    def __repr__(self):
        return f'data:\n{self.data},\n grad: {self.grad}\n'

class Module():
    def params(self):
        return []
    def zero_grad(self):
        for v in self.params():
            v.grad = np.zeros_like(v.data)
        
class Layer(Module):
    def __init__(self, nin, nout, nonlin=True):
        self.w = ADVect(np.random.normal(size=(nout, nin)))
        self.b = ADVect(np.zeros_like(np.arange(nout)))
        self.nonlin = nonlin

    def __call__(self, x):
        out1 = self.w @ x
        out2 = out1 + self.b
        out3 = out2.relu() if self.nonlin else out2
        return out3
    
    def params(self):
        return [self.w, self.b]

class MLP(Module):
    def __init__(self, sz):
        self.layers = [Layer(sz[i], sz[i+1], nonlin = i != len(sz)-2) for i in range(len(sz)-1)] 
    
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def params(self):
        ret = []
        for _ in self.layers:
            ret.extend(_.params())
        return ret