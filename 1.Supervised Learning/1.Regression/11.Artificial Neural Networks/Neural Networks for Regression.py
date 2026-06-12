import numpy as np

# y(x) = b3 + W3*tanh(b2 # W2*tanh(b1+ W1*x))

# z1 = x                     # input layer     
# a2 = b1 + W1 @ z1         
# z2 = np.tanh(a2)           # second layer (hidden) 
# a3 = b2 + W2 @ z2  
# z3 = np.tanh(a3)           # third layer (hidden)    
# z4 = b3 + W3 @ z3          # output layer
# y  = z4                    # output
# r  = y - t                 # residual
# E  = 0.5 * r.T @ r         # suared error


class NeuralNetwork:

    def __init__(self, p, m, k, r, learning_rate=0.01):

        # p     input dimension
        # m     hidden layer 1
        # k     hidden layer 2
        # r     output dimension
        np.random.seed(1603)
        self.learning_rate = learning_rate

        # W1 of dimensions (m × p)
        self.W1 = np.random.randn(m, p) * 0.1 # standard normally distzributed random numbers
        self.b1 = np.zeros(m)

        # W2 of dimensions (k × m)
        self.W2 = np.random.randn(k, m) * 0.1 # standard normally distzributed random numbers
        self.b2 = np.zeros(k)

        # W3 of dimensions (r × k)
        self.W3 = np.random.randn(r, k) * 0.1 # standard normally distzributed random numbers
        self.b3 = np.zeros(r)

    def forward(self, x):

        self.z1 = x
        self.a2 = self.b1 + self.W1 @ self.z1
        self.z2 = np.tanh(self.a2)

        self.a3 = self.b2 + self.W2 @ self.z2
        self.z3 = np.tanh(self.a3)

        self.z4 = self.b3 + self.W3 @ self.z3
        self.y = self.z4

        return self.y


    def loss(self, t):
        self.r = self.y - t
        self.E = 0.5 * self.r.T @ self.r
        return self.E


    def backward(self):

        r = self.r

        # dE/db3 = r
        dE_db3 = r

        # dE/dW3 = r @ z3^T
        dE_dW3 = np.outer(r, self.z3) # rz3^T

        # dE/db2
        dE_db2 = (1.0 - self.z3**2)*(self.W3.T @ r) # Diag(1-z3^2)(W3^T)r

        # dE/dW2
        dE_dW2 = np.outer(dE_db2,self.z2) # Diag(1-z3^2)(W3^T)r(z2^T)

        # dE/db1
        dE_db1 = (1.0 - self.z2**2)*(self.W2.T @ dE_db2) # Diag(1-z2^2)(W2^T)Diag(1-z3^2)(W3^T)r

        # dE/dW1
        dE_dW1 = np.outer(dE_db1,self.z1) # Diag(1-z2^2)(W2^T)Diag(1-z3^2)(W3^T)r(z1^T)

        # Gradient Descent
        self.b1 -= self.learning_rate * dE_db1
        self.W1 -= self.learning_rate * dE_dW1

        self.b2 -= self.learning_rate * dE_db2
        self.W2 -= self.learning_rate * dE_dW2

        self.b3 -= self.learning_rate * dE_db3
        self.W3 -= self.learning_rate * dE_dW3

    def train_step(self, x, t):

        y = self.forward(x)
        loss = self.loss(t)
        self.backward()

        return y, loss
    

if __name__ == "__main__":
    np.random.seed(0)
    net = NeuralNetwork(p=2,m=4,k=3,r=1,learning_rate=0.01)

    x = np.array([0.5, -1.2])
    t = np.array([1.0])

    for epoch in range(1000):

        y, loss = net.train_step(x, t)

        if epoch % 100 == 0:
            print(epoch, loss)