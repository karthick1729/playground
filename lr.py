import numpy as np
def train():
    """write linear regression with numpy"""
    #X = np.random.randint(1,100,size=100)
    X = np.linspace(0,1,10)
    Y = 4 * X + 1.2 + np.random.normal(0,0.01,1)
    lr = 0.1
    W = 1.2
    b = 0
    for epoch in range(100):
        grad_w = 0
        grad_b = 0
        for i in range(X.shape[0]):
            x = X[i]
            y = Y[i]
            y_hat = W * x + b
            residual = 1/2*(y_hat-y)**2 
            grad_w += 1/2*(y_hat-y)*W 
            grad_b += 1/2*(y_hat-y)
        W = W - lr * grad_w
        b = b - lr * grad_b
        print(f"epoch: {epoch}, loss: {residual}, W: {W:.3f}, b: {b}")
        input()
    # W = 0.01
    # lr = 10
    # b = 0
    # for epoch in range(10):
    #     Y_hat = W * X + b
    #     loss = np.mean((Y_hat-Y) ** 2)
    #     grad_w = np.mean(-2 * (Y_hat - Y) * X)
    #     grad_b = np.mean(-2 * (Y_hat - Y))
    #     W = W - lr * grad_w
    #     b = b - lr * grad_b
    #     print(f"epoch: {epoch}, loss: {loss}, W: {W}, b: {b}")

train()
    