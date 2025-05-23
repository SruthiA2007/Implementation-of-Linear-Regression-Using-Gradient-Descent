# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and preprocess data: Read CSV data, extract features (X) and target (y), and convert them to float.
2.Normalize features: Apply StandardScaler to scale both features and target for better training performance.
3.Define model: Implement gradient descent-based linear regression with bias term added to input features.
4.Train model: Iterate to minimize error and update model parameters (theta) using gradient descent.
5.Predict new value: Scale the new input data, apply the trained model, and get the scaled prediction.
6.Inverse scale result: Convert the scaled prediction back to the original scale and print it


## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: SRUTHI A

RegisterNumber:  212224240162
*/
```
```
import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler
        
        def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
            X=np.c_[np.ones(len(X1)),X1]
            theta=np.zeros(X.shape[1]).reshape(-1,1)
            for _ in range(num_iters):
                predictions=(X).dot(theta).reshape(-1,1)
                errors=(predictions-y).reshape(-1,1)
                theta_=learning_rate*(1/len(X1))*X.T.dot(errors)
                pass
            return theta
        
        
        data=pd.read_csv('/content/50_Startups.csv',header=None)
        print(data.head())
        
        
        X=(data.iloc[1:, :-2].values)
        print(X)
        
        
        X1=X.astype(float)
        scaler=StandardScaler()
        y=(data.iloc[1:,-1].values).reshape(-1,1)
        print(y)
        
        
        X1_Scaled=scaler.fit_transform(X1)
        Y1_Scaled=scaler.fit_transform(y)
        
        
        print(X1_Scaled)
        print(Y1_Scaled)
        
        theta=linear_regression(X1_Scaled,Y1_Scaled)
        new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
        new_Scaled=scaler.fit_transform(new_data)
        prediction=np.dot(np.append(1,new_Scaled),theta)
        prediction=prediction.reshape(-1,1)
        pre=scaler.inverse_transform(prediction)
        print(f"Predicted value: {pre}")
```

## Output:

![Screenshot 2025-05-23 091700](https://github.com/user-attachments/assets/749c76c2-0e53-4597-b468-24f497979671)

![Screenshot 2025-05-23 091751](https://github.com/user-attachments/assets/fa8e3256-9d9e-44c1-9dde-c9ca477ea244)

![Screenshot 2025-05-23 091817](https://github.com/user-attachments/assets/3ed47c2d-6c55-4efa-bcbc-a37f32e70652)

![Screenshot 2025-05-23 091857](https://github.com/user-attachments/assets/c50fe4c2-30da-4812-b1e5-48ee3433f48b)
![Screenshot 2025-05-23 091925](https://github.com/user-attachments/assets/deeed545-0215-4240-b6e5-bb44fc21b3d3)






![Screenshot 2025-05-23 091951](https://github.com/user-attachments/assets/49bc471a-5954-4648-9b9a-a2ae8391c4d5)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
