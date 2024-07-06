from atto import Value
import neural as nn

mymlp = nn.MultiLayerPerceptron(1, [5, 5, 1])

x = [Value(1)]

for step in range(100):

    y = mymlp(x)

    label = Value(0)

    def loss(y, label):
        return (label - y)**2

    myloss = loss(y, label)

    myloss.backward()

    for param in mymlp.parameters():
        param.data -= 0.02 * param.gradient

    # print('y: \n',y)
    print('loss: ', myloss)

    mymlp.set_gradients_zero()

