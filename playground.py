import mxnet as mx
from mxnet import nd
mx.random.seed(1)

x = nd.empty((3, 4))
print(x)

x = nd.zeros((3, 4))
print(x)

y = nd.ones((3, 4))
print(x)

print(nd.dot(x, y.T))
