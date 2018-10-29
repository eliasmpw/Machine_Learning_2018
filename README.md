# Machine Learning 2018 - Project I

The Higgs boson is an elementary particle in the Standard Model of particle physics, produced by the quantum excitation of the higgs field and explains why some particles have mass. To confirm the existence of this particle, the CERN made several experiments from which we obtained real data and the objective of this work is to present a machine learning model that will give the best fit to the data provided. However the data also contains other possible particles, so the main objective is to determine if the data of each experiment belongs to the higss boson or to other particle depending on the values provided on the data set.

## Overview of the project's code
```
├── Report
│   ├── Machine_Learning_AEJ_HigssBossom.pdf    <-Report in .pdf format
│   └── Machine_Learning_AEJ_HigssBossom.tex    <-Report in LaTeX format
├── Cross_validation.ipynb  <-Parameters optimization
├── custom_helpers.py       <-Additional custom helpers
├── implementations.py      <-Contains 6 Methods implementation
├── proj1_helpers.py        <-Helpers provided for project 1
├── run.py                  <-Best method runner
````

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

3. In order to generate the plots/graphics used on the report, you need to install the **matplotlib** python package, by running the following command on the terminal:
    ```
    pip install matplotlib
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
    A new file called "best_model.csv" will be generated, which contains the predictions and can be uploaded to [Kaggle](https://www.kaggle.com/c/epfml18-higgs/submit)

## Running other prediction methods

In order to run other predictions methods, or running a prediction method with different paremeters, you should edit the run.py file.

1. Feature treatment methods (lines 69 to 72)

    **flag_add_offset** = True/False

    **flag_standardize** = True/False

    **flag_remove_outliers** = True/False

    **degree** = A number greater than 0.

2. Training model to apply (line 75)

    **flag_method** = A number according to the mapping:

    ```
    Methods mapping
    0    Linear Regression (Full gradient descent)
    1    Linear Regression (Stochastic gradient descent)
    2    Least Squares Method
    3    Ridge Regression
    4    Logistic Regression (Stochastic Gradient Descent)
    5    Regularized Logistic Regression (Stochastic Gradient Descent)
    ```

3. Set training parameters (lines 78 to 80)

    **max_iters** = A number, represent the iterations to do.

    **gamma** = A number, gamma value that you want to use.

    **lambda_** = A number, lambda value that you want to use.

## Hyper-Parameters Optimization and Cross Validation

In order to get the best values for lambda and gamma you can run the **Cross_validation.ipynb** notebook. The following instructions will help you run it.

1. Start Jupyter Notebook

    Run this command:
    ```
    jupyter notebook
    ```
    This will launch a browser window, with a file explorer open.

2. Open notebook

    Navigate through the folders, and open the **Cross_validation.ipynb** file.

3. Run the cells of the notebook

    Now you can manually press run on every cell that you want to run (The imports and required functions cells should always be ran). Or you can open the **Cell** menu and click on the option **Run All**

    You will see the results/outputs between cells.

## Authors

* **Andres Ivan Montero Cassab**
* **Jonas Florian Jäggi**
* **Elias Manuel Poroma Wiri**
