import numpy as np

from kernel import (
    PolynomialKernel,
    RBF,
    GaussianProcessClassifier,
    GaussianProcessRegressor
)


def create_toy_data(func, n=10, std=1., domain=[0., 1.]):
    # 输入
    x = np.linspace(domain[0], domain[1], n)
    # 加了噪声的输出
    t = func(x) + np.random.normal(scale=std, size=n)
    return x, t

def sinusoidal(x):
    return np.sin(2 * np.pi * x)


x_train, y_train = create_toy_data(sinusoidal, n=10, std=0.1)
x = np.linspace(0, 1, 100)

model = GaussianProcessRegressor(kernel=PolynomialKernel(3, 1.), beta=int(1e10))
model.fit(x_train, y_train)

y = model.predict(x)

print()