


# It's a class that holds a value
class Value():
    
    # Initial Construction
    def __inti__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = _children
        self._op = _op
        
    
    # Representation
    def __repr__(self) -> str:
        print(f"Value(data={self.data},grad={self.grad})")
        
    # Addition
    def __add__(self,other):
        other = other if isinstance(other, Value) else Value(other)
        res = Value(self.data+other.data,(self,other),'+')
        def _backward():
            self.grad += res.grad
            other.grad += res.grad
        res._backward = _backward
        
        return res
    
    #Reverse Addition
    def __radd__(self,other):
        return self+other
    
    # Multiplication
    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        res = Value(other.data*self.data,(self,other),'*')
        def _backward():
            self.grad += res.grad * other.data
            other.grad += res.grad * self.data
        res ._backward = _backward
        
        return res
    
    # Reverse Multiplication   
    def __rmul__(self,other):
        return self * other
    
    # Negetive
    def __neg__(self):
        return self * -1
    
    # Subtraction
    def __sub__(self,other):
        return self + (-other)
    

    # Reverse Subtraction    
    def __rsub__(self,other):
        return self + (-other)
    
    # Power function
    def pow(self,other):
        assert isinstance(other,(int,float))
        res = Value(self.data**other,(self,),f'**{other}')
        def _backward():
            self.grad = (other)*(self.data**(other-1)) * res.grad 
        res._backward = _backward
        
        return res      
    
    # Division
    def __truediv__(self, other):
        return self * other**-1

    # Reverse Division
    def __rtruediv__(self, other): 
        return other * self**-1
    
    # Activation function
    def relu(self):
        res = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (res.data > 0) * res.grad
        res._backward = _backward

        return res
    
    def backward(self):
        
        # topological order all of the children in the graph
        topo = []
        visited = set()
        def dfs(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    dfs(child)
                topo.append(v)
        dfs(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()