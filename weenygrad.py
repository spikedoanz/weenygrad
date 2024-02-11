import numpy as np

class ADVect():
    def __init__(self, data, children = []):
        data = data if isinstance(data, np.ndarray) else np.array(data)
        self.data =data 
        self.grad = np.zeros_like(self.data)
        self.children = set(children)
        self._backward = lambda: None 
    
    def __add__(self, other):
        other = other if isinstance(other, ADVect) else ADVect(other)
        out = ADVect(self.data+ other.data, [self, other])
        def backward():
            if self == other:
                self.grad = other.grad + out.grad
            else:
                self.grad = self.grad + out.grad
                other.grad = other.grad + out.grad
        out._backward = backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, ADVect) else ADVect(other)
        if self.data.ndim == 1 and other.data.ndim == 1:
            out = ADVect(self.data * other.data, [self, other]) 
        else: 
            out = ADVect(self.data @ other.data, [self, other]) 
        def backward():
            if np.ndim(self.data) == 1:
                self.grad   = self.grad +   out.grad * other.data
                other.grad  = other.grad +  out.grad * self.data
            else: # Matrix vector mulitplication is supported, not matrix @ matrix
                self.grad = self.grad + np.outer(out.grad, other.data) 
                other.grad = other.grad + np.transpose(self.data) @ out.grad
        out._backward = backward
        return out

    def sum(self):
        out = ADVect(np.sum(self.data), [self])
        def backward():
            self.grad = np.ones_like(self.grad)*out.grad
        out._backward = backward   
        return out

    def relu(self):
        ReLU = np.vectorize(lambda x: max(0, x))
        out = ADVect(ReLU(self.data), [self])
        def backward():
            self.grad = self.grad + (out.data >= 0) * out.grad
        out._backward = backward
        return out

    def backward(self):
        topo, visited = [], []
        def build_topo(v):
            if v not in visited:
                topo.append(v)
                visited.append(v)
                for node in v.children:
                    build_topo(node)
                return topo
            return []
        self.grad = np.ones_like(self.data)
        build_topo(self)
        for v in topo:
            v._backward()

    def __radd__(self, other):
        return self + other

    def __rmatmul__(self, other): 
        return self @ other
    
    def __repr__(self):
        return f'data:\n{self.data},\n grad: {self.grad}'

class Module():
    def params(self):
        return []
    def zero_grad(self):
        for v in self.params():
            v.grad = np.zeros_like(v.data)
        
class Layer(Module):
    def __init__(self, nin, nout, nonlin=True):
        self.w = ADVect(np.random.normal(size=(nout, nin)))
        self.b = ADVect(np.random.normal(size=nout))
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