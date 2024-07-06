class Value:
  def __init__(self, data, _children=(), _operation=''):
    self.data = data
    self.gradient = 0

    self._backward = lambda: None

    self._children = _children
    self._operation = _operation

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      self.gradient += out.gradient
      other.gradient += out.gradient

    out._backward = _backward

    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      self.gradient += other.data * out.gradient
      other.gradient += self.data * out.gradient
    
    out._backward = _backward

    return out

  def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data ** other, (self,), f'**{other}')

    def _backward():
      self.gradient += (other * (self.data ** (other-1))) * out.gradient

    out._backward = _backward
    

    return out

  def relu(self):
    out = Value(0 if self.data < 0 else self.data, (self,), f'ReLU')
    
    def _backward():
        self.gradient += (out.data > 0) * out.gradient

    out._backward = _backward

    return out

  def backward(self):

    topological_sorted = []
    visited = set()

    def build_topological_sort(node):
      if node not in visited:
        visited.add(node)
        for child in node._children:
          build_topological_sort(child)
        topological_sorted.append(node)
      
    build_topological_sort(self)

    self.gradient = 1

    for node in reversed(topological_sorted):
      node._backward()

  def __neg__(self):
    return self * -1

  def __radd__(self, other):
    return self + other

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return other + (-self)

  def __rmul__(self, other):
    return self * other

  def __truediv__(self, other):
    return self * other**-1

  def __rtruediv__(self, other): 
    return other * self**-1

  def __repr__(self):
    return f'Value(Data:{self.data}, Gradient:{self.gradient})'