import mxnet as mx
from mxnet import gluon, autograd, nd, metric
from mxnet.gluon.loss import L1Loss, L2Loss

from data import AsianFaceDatasets, IMDBWIKIDatasets
from model import Net

import math
epoches = 60
lr = 0.001
mom = 0.9
weight_decay = 0.0005

data_ctx = mx.gpu(0)
model_ctx = mx.gpu(0)
scale = nd.arange(0, 101, ctx=model_ctx).reshape((101, 1))

l1_loss = L1Loss()
# l1_loss = L2Loss()
net = Net()

# training_datasets = AsianFaceDatasets(csv_path='/home/gdshen/datasets/face/asian/train.csv',
#                                       img_dir='/home/gdshen/datasets/face/asian/images')
# test_datasets = AsianFaceDatasets(csv_path='/home/gdshen/datasets/face/asian/test.csv',
#                                   img_dir='/home/gdshen/datasets/face/asian/images', train=False)

training_datasets = IMDBWIKIDatasets(csv_path='/home/gdshen/datasets/face/processed/train.csv', train=True)
test_datasets = IMDBWIKIDatasets(csv_path='/home/gdshen/datasets/face/processed/test.csv', train=False)


def evaluate_accracy(data_iterator, net):
    acc = metric.MAE()
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(data_ctx)
        label = label.as_in_context(data_ctx)
        output = net(data)
        prediction = nd.dot(output, scale)
        acc.update(preds=prediction, labels=label)
    return acc.get()[1]


if __name__ == '__main__':
    net.output.collect_params().initialize(init=mx.init.Uniform(scale=1/math.sqrt(2048)), ctx=model_ctx, force_reinit=False)
    # net.collect_params().initialize(init=mx.init.Xavier(), ctx=model_ctx)
    # net.load_params('/home/gdshen/datasets/mxnet_checkpoint/checkpoint-imdb-15.params', ctx=model_ctx)
    net.collect_params().reset_ctx(ctx=model_ctx)
    #
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'momentum': mom, 'wd': weight_decay})
    # trainer = gluon.Trainer(net.collect_params(), 'Adam', {'learning_rate': lr})
    data_iter = gluon.data.DataLoader(training_datasets, 10, shuffle=True, num_workers=8, last_batch='discard')
    eval_iter = gluon.data.DataLoader(test_datasets, 10, shuffle=False, num_workers=8, last_batch='discard')

    for epoch in range(1, epoches + 1):
        if epoch % 10 == 0:
            trainer.set_learning_rate(lr=trainer.learning_rate * 0.1)
        for batch_i, (data, label) in enumerate(data_iter):
            data = data.as_in_context(data_ctx)
            label = label.as_in_context(data_ctx)
            with autograd.record():
                output = net(data)
                output = nd.dot(output, scale)
                loss = l1_loss(output, label)
            loss.backward()
            trainer.step(batch_size=10)

            if batch_i % 100 == 0:
                print(f'Epoch {epoch} batch {batch_i} loss {loss.asnumpy().mean()}')
        if epoch % 5 == 0:
            net.save_params(f'/home/gdshen/datasets/mxnet_checkpoint/checkpoint-asian2-{epoch}.params')
        test_accuracy = evaluate_accracy(eval_iter, net)
        print(test_accuracy)
