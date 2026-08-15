"""
*! Reverse-mode autograd needs a scalar objective at the end.

Partial Derivatives -> Jacobian Matrix
#Addition
dX = dA
dW = dA

# MatMul
W.grad += dA @ X.T
X.grad += W.T @ dA
"""

import numpy as np

class Tensor:
    def __init__(self, data, prev=()):
        self.data = np.array(data)
        self.shape = self.data.shape

        self.prev = prev
        self.grad = 0 #np.zeros_like(self.data)
        self._backward = lambda: None

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other))

        def backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = backward

        return out

    def __matmul__(self, other):
        out =  Tensor(self.data @ other.data, (self, other))

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data),(self,))

        def backward():
            self.grad += (out.grad * (self.data > 0))
        out._backward = backward

        return out

    def mse(self, target):

        diff = self.data - target.data

        out = Tensor(
            np.mean(diff ** 2),
            (self, target)
        )

        def backward():
            n = self.data.size

            self.grad += (2 / n) * diff * out.grad

        out._backward = backward

        return out

    def backward(self):
        # 1. Build graph
        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)

                for parent in node.prev:
                    build(parent)

                topo.append(node)

        build(self)

        # 2. Start gradient at loss
        self.grad = 1.0

        # 3. Walk backwards
        for node in reversed(topo):
            node._backward()


class Linear:
    def __init__(self, in_features, out_features):
        scale = np.sqrt(2 / in_features)
        self.w = Tensor(np.random.randn(in_features, out_features) * scale)
        self.b = Tensor(np.zeros(out_features))

    def __call__(self, x):
        return x @ self.w + self.b

    def parameters(self):
        return [self.w, self.b]


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self):
        for param in self.parameters:
            param.data = param.data + (-self.lr * param.grad)

    def zero_grad(self):
        for param in self.parameters:
            param.grad.fill(0)


np.random.seed(42)
X = Tensor(np.random.randn(1, 768))
Y = Tensor(np.random.randn(1, 9))
print(f"Target: {Y.data}\n")
layer1 = Linear(768,128)
layer2 = Linear(128,9)

optimizer = SGD(
    layer1.parameters() + layer2.parameters(),
    lr=0.01
)

for i in range(10):

    # Forward
    logits = layer2(layer1(X))

    # Loss
    loss = logits.mse(Y)

    # Backward
    loss.backward()

    #Print Loss
    print(
        "loss:", loss.data,
        "prediction:", logits.data,
        "target:", Y.data
    )

    # Update weights
    optimizer.step()

    # Clear gradients
    optimizer.zero_grad()




# W = Tensor([[1,3],[0.2,0.1]])
# X = Tensor([[1,7],[9,8]])
# B = Tensor([[8,4],[7,3]])
#
# Y = Tensor([[2,3],[1,9]])
#
# Z = W @ X
# A = Z + B
# R = A.relu()
# L = R.mse(Y)
#
# L.backward()

# def out(node):
#     for parent in node.prev:
#         out(parent)
#     print(f"Node Data: {node.data}")
#     print(f"Node Gradients: {node.grad}")
#
# out(loss)