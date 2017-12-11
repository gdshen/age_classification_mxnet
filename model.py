from mxnet import ndarray as nd
from mxnet.gluon import HybridBlock, nn
from mxnet.gluon.model_zoo import vision as models


class Net(HybridBlock):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.features = models.resnet50_v2(pretrained=True).features
            self.features.collect_params().setattr('lr_mult', 0.1)
            self.output = nn.Dense(units=101)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        x = F.softmax(x)
        return x


if __name__ == '__main__':
    data = nd.random_uniform(shape=(4, 3, 224, 224))
    net = Net()
    net.collect_params().initialize()
    output = net(data)
    print(output)
