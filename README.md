# Machine Learning 2018 - Project I

The Higgs boson is an elementary particle in the Standard Model of particle physics, produced by the quantum excitation of the higgs field and explains why some particles have mass. To confirm the existence of this particle, the CERN made several experiments from which we obtained real data and the objective of this work is to present a machine learning model that will give the best fit to the data provided. However the data also contains other possible particles, so the main objective is to determine if the data of each experiment belongs to the higss boson or to other particle depending on the values provided on the data set.

## Prerequisites

1. You will need to have Anaconda installed, the following tutorials will guide you through the installation.

Operating System | Tutorial
--- | ---
Mac | https://docs.anaconda.com/anaconda/install/mac-os/
Windows | https://docs.anaconda.com/anaconda/install/windows/
Ubuntu | https://docs.anaconda.com/anaconda/install/linux/

2. Once you have it installed, try running the following command on a terminal:

```
python
```

You should see a message similar to the following:
```
Python 3.6.5 |Anaconda, Inc.| (default, Mar 29 2018, 13:32:41) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
```

## Installing

1. In a terminal, change your directory to the location of the compressed zip file of the project.

```
cd {path_of_zip_file}
```

2. Unzip the zip file of the project.
```
unzip -a mlProject1.zip
```

## Running the best method and generating a predictions file

1. If your terminal is not in the location of the project files, change your directory to that location.
```
cd {path_of_project_files}
```

2. Run the run.py script in the terminal.
```
python run.py
```
A new file called "MLproject1.csv" will be generated, which contains the predictions and can be uploaded to [Kaggle](https://www.kaggle.com/c/epfml18-higgs/submit)

## Running other prediction methods

// TODO: explain how to run the other methods.

1. Least Squares Gradient Descent.
```
least_squares_GD(y, tx, initial_w, max_iters, gamma)
```

2. Least Squares Stochastic Gradient Descent.
```
least_squares_SGD(y, tx, initial_w, max_iters, gamma)
```

3. Least Squares.
```
least_squares(y, tx)
```

4. Ridge Regression
```
ridge_regression(y, tx, lambda_)
```

5. Logistic Regression
```
logistic_regression(y, tx, initial_w, max_iters, gamma)
```

6. Regularized Logistic Regression
```
reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)
```

## Hyper-Parameters Optimization and Cross Validation

In order to get the best values for lambda and gamma run the following command.

```
// TODO: COMMAND WILL GO HERE
```

## Authors

* **Andres Ivan Montero Cassab**
* **Jonas Florian Jäggi**
* **Elias Manuel Poroma Wiri**
