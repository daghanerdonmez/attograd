import attograd.neural as nn

mymlp = nn.MultiLayerPerceptron(3, [4, 4, 1])

xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

for step in range(40):
    x = xs[step % len(xs)]
    label = ys[step % len(ys)]

    y = mymlp(x)

    def loss(y, label):
        return (y - label)**2

    myloss = loss(y, label)

    mymlp.set_gradients_zero()
    myloss.backward()

    for param in mymlp.parameters():
        param.data += -0.1 * param.gradient 

    print(f'Step: {step}, Loss: {myloss.data}')
