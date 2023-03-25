import random
from .engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []
    
class Neuron(Module):
    
    def __init__(self,nin, nonlin=True) -> None:
        self.w = [Value(random.random(-1,1) for _ in range(nin))]
        self.b = Value(0)
        self.nonlin = True
        
    def __call__(self,x):
        res = sum((wi*xi for wi,xi in zip(self.w, x)),self.b)
        return res.relu() if self.nonlin else res

    def __repr__(self) -> str:
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
    
    def parameters(self):
        return self.w + [self.b]        

class Layer(Module):
    
    def __init__(self, nin, nout, **kwargs) -> None:
        self.neurons = [Neuron(nin,**kwargs) for _ in range(nout)]
    
    def __call__(self,x):
        res = [n(x) for n in self.neurons]
        return res[0] if len(res) == 1 else res
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters]
    
    def __repr__(self):
        return f"Layer of [{','.join(str(n) for n in self.neurons)}]"
        
class MLP(Module):
    
    def __init__(self, nin, nouts) -> None:
        ncounts = [nin] + nouts
        self.layers = [Layer(ncounts[i],ncounts[i+1],nonlin=i!=len(nouts)-1) for i in range(len(ncounts)) ]
        
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters] 
    
    def __repr__(self) -> str:
        return f"MLP of [{','.join(str(layer) for layer in self.layers)}]"           