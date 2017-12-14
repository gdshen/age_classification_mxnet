from mxnet import sym
from model import Net

x = sym.var('data')
net = Net()
y = net(x)
y.save('model.json')

