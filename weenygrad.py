import numpy as np

class ADVect():
    def __init__(self, value, children = ()):
        self.value = value
        self.grad = np.zeros_like(self.value)
        self.children = children
        self._backward = lambda: None 
    
    def __add__(self, other):
        out = ADVect(self.value + other.value, [self, other])
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = backward
        return out
    
    def __matmul__(self, other):
        out = ADVect(self.value @ other.value, [self, other])
        def backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = backward
        return out

    def relu(self):
        ReLU = np.vectorize(lambda x: max(0, x))
        out = ADVect(ReLU(self.value), [self])
        def backward():
            ReLU_diff = np.vectorize(lambda x: 0 if x < 0 else 1)
            self.grad += ReLU_diff(out.grad)
        out._backward = backward
        return out

    def backward(self):
        topo = []
        visited = set() 
        def build_topo(v):
            if self not in visited:
                topo.append(v)
                visited.add(v)
                for node in v.children:
                    build_topo(node)
                return topo
            return []
        build_topo(self)
        self.grad = np.ones_like(self.value)
        for v in topo:
            print(v)
            v._backward()
    
    def __repr__(self):
        return f'{self.value}, grad: {self.grad}'

class Module():
    def params(self):
        return []
    def zero_grad(self):
        for v in self.params():
            v.grad = np.zeros_like(v.value)
        
class Neuron(Module):
    def __init__(self, nin):
        self.w = ADVect(np.random.normal(size=nin))
        self.b = ADVect(np.random.normal(size=1))
    
    def __call__(self,x):
        out1 = self.w @ x
        out2 = out1 + self.b 
        out3 = out2.relu()
        return out3

if __name__ == '__main__':

    a = Neuron(10)
    b = ADVect(np.arange(10))
    c = a(b)
    c.backward()
    print(c)
    print(a.w)
    print(a.b)

