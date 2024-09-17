import numpy as np
import matplotlib.pyplot as plt

class SLR:
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.B0=0
        self.B1=0

    def fit(self):
        first_sum=np.sum(self.x*(self.y-np.mean(self.y)))
        second_sum=np.sum(self.x*(self.x-np.mean(self.x)))

        self.B1=first_sum/second_sum
        self.B0=np.mean(self.y)-self.B1*np.mean(self.x)
        self.y_hat=self.B0+self.B1*self.x

    def r_resudal(self):
        sst=np.sum((self.y-np.mean(self.y))**2)
        sse=np.sum((self.y-self.y_hat)**2)
        ssr=1-sse/sst
        return ssr

    def sum_square_error(self):
        total_error=np.sum((self.y-self.y_hat)**2)
        return total_error
    def predict(self,x):
        return self.B0+self.B1*x



x = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2])
y = np.array([250, 300, 480, 430, 630, 730, ])

mod=SLR(x,y)
mod.fit()
yhat=mod.y_hat
plt.scatter(x, y)
plt.plot(x,yhat)
plt.show()
print(mod.sum_square_error())
print(mod.predict(5))
print(mod.r_resudal())
