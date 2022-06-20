20-06-2022

# User guide

## How to use

To run the python script, you can either run it from your IDE, or use "main.py" in your terminal.
Before you can use it, check whether all dependencies are installed, which are specified in the requirements.txt file.
When the script runs, it will open up a file dialog, in which you should select an excel file with testing data. This excel sheet does not need to be in the same directory as main.py.
Some time will pass, after which two things happen.
The first is that in the terminal, a list of ALT (Accelerated Life Testing) models from the [Reliability library](https://reliability.readthedocs.io/en/latest/) is shown.
In this overview, all model parameters are listed with their values and sigma.
Additionally it will show the log-likelihood, AIC and BIC values, which indicate how well the model works for the given data.
The list is sorted by these goodness-of-fit indicators.
The second thing that will happen is that Reliability shows two plots. One with the reliability plots of all fitted models and another plot showing the reliability plot of the best ALT model.

## user parameters

Below the function definitions in main.py, there are a couple of variables that should be changed, based on the selected excel sheet.
The `test_condition_rows` variable specifies in which rows the test conditions (temperature / current) are present. 
Remember that Python is 0 indexed, so 0 corresponds to the first row. 
If the variable has 3 elements, main.py will try to find 3 test condition rows and accordingly, 3 tables with test data.
The `use_level_stress` variables are used to predict the lifetime at a given stress (temperature / current). 
Reliability will show you this lifetime (in hours) if you change `Fit_Everything_ALT` with a specific model. 
Note that temperature should be in Kelvin.
