#! /Users/luisroman/miniconda3/envs/fdd2/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Logit:
      '''
      This class is a logit classifier

      X in [m, n+1]
       n: variables
       m: observations

      y in [m, 1]

      theta in [1, n+1]

      np.dot(X, theta.T)
      
      ''' 
      def __init__(self, X, y, alpha=.005):
            self.X = self.add_ordinate(X)
            self.y = y
            self.theta = np.random.rand(X.shape[1]+1).reshape(1, -1)
            self.alpha = alpha # This is the learning rate
            self.loss_hist = []
            print(f'Loading data: X shape [{self.X.shape}]')
            print(f'Loading data: y shape [{self.y.shape}]')
            print(f'params shape: theta [{self.theta.shape}]')

      def add_ordinate(self, X):
            return np.hstack([np.ones(X.shape[0]).reshape(-1, 1), X])
      
            
      def forward(self, X=None):
            '''
            This function implements:
            the logit pass to X. 1/(1 + e-z*theta)
            '''
            X = self.add_ordinate(X) if not X is None else self.X
            return 1/(1 + np.exp(-np.dot(X, self.theta.T)))

      def loss(self):
            '''
            Computes cross entropy loss
            '''
            p = self.forward()
            return -np.mean(self.y*np.log(p) + (1-self.y)*np.log(1-p))

      def train(self, tol=1e-5, max_iter=10000):
            iters = 0
            loss = np.Inf
            while(loss > tol and iters < max_iter):
                  loss = self.loss()
                  if not iters % 500:
                        pass
                        ##print(f'loss: {loss}')
                  p = self.forward().reshape(-1, 1)
                  self.theta -= -self.alpha*np.mean((self.y - p)*self.X, axis=0)
                  iters += 1
                  self.loss_hist.append(loss)

if __name__ == '__main__':
      mean_1 = np.array([10, 10])
      mean_2 = np.array([7, 7])
      m_cov = np.array([[3, 0], [0, 3]])
      size = 500
      X = np.concatenate([np.random.multivariate_normal(size=size, mean=mean_1, cov=m_cov),
                          np.random.multivariate_normal(size=size, mean=mean_2, cov=m_cov)],
                         axis=0)
      y = np.concatenate([np.ones(size).reshape(-1, 1), np.zeros(size).reshape(-1, 1)])
      # Instantiate Model
      log1 = Logit(X=X, y=y)
      # Forward pass
      log1.train()
      # Loss
      plt.plot(range(len(log1.loss_hist)), log1.loss_hist)
      plt.savefig('loss.png')

      # ----------------------------------------
      # Plot curves
      # ----------------------------------------
      (x1_min, x2_min) = X.min(axis=0) - 1
      (x1_max, x2_max) = X.max(axis=0) + 1
      # Generate axis
      x1_axis = np.arange(x1_min, x1_max, step=.1)
      x2_axis = np.arange(x2_min, x2_max, step=.1)
      # Mesh grid
      x1x_, x2x_ = np.meshgrid(x1_axis, x2_axis)
      # Shape grid as features
      x1x = x1x_.flatten().reshape(-1, 1)
      x2x = x2x_.flatten().reshape(-1, 1)
      # Add bias term (vector of ones) and transpose (set into 'column' shape
      new_X = np.hstack((np.ones(x1x.shape), x1x, x2x))
      print(new_X.shape)
      # Generate predictions and shape them as grid.
      y_hat = log1.forward(new_X).reshape(x1x_.shape)
      # Generate plot
      fig, ax = plt.subplots(figsize=(10, 10))
      # Classification contour (notice that we need the grid shape)
      ax.contourf(x1x_, x2x_, y_hat)
      # Add population points
      sns.scatterplot(x='x1', y='x2', hue='class',
                      data=pd.DataFrame(np.hstack((y, X)),
                                        columns=['class', 'x1', 'x2']),
                      ax=ax)
      print('Saving classification region plot')
      fig.savefig('class_region.png')
      
      
