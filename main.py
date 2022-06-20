# Made by Henri Vlot
# Student ID: 4363523
# %% - Imports
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reliability.ALT_fitters import Fit_Everything_ALT
from scipy.optimize import curve_fit


# %%
def get_test_conditions(skip_rows: int, file_path: str, sheet: int = 0):
    """Get temperature and current from the Excel file. """
    test_condition_row = pd.read_excel(
        file_path,
        sheet_name=sheet,
        skiprows=skip_rows,
        header=None,
        nrows=1
    )

    try:
        test_conditions = test_condition_row[[0]].loc[0][0].split()
    except AttributeError:
        raise AttributeError(
            """Could not parse test conditions from file.
            Make sure the correct rows are specified in `test_condition_rows` and that the following format is used:
            'Test Condition #   Temp   Current'"""
        )

    try:
        # Temperature of the test [K]
        temperature = int(test_conditions[3]) + 273.15
        # Current of the test [A]
        current = test_conditions[5]

        return temperature, float(current)
    except ValueError:
        raise ValueError(
            """Could not parse test conditions from file.
            Make sure the correct rows are specified in `test_condition_rows` and that the following format is used:
            'Test Condition #   Temp   Current'"""
        )


def get_data_values(skip_rows: int, file_path: str,
                    sheet: int = 0, num_samples: int = 24):
    """Extract the measurement data from the given excel file.

    - Input
        - skip_rows: Number of rows to skip
        - file_path: OS path to the file
        - fheet: Number of the sheet which is read out, defaults to 0 (first sheet)
        - num_samples: Number of samples tested, defaults to 24.
    - Output
        - hours_tested: list of hours tested
        - data_values: the percentage of lumen maintained after x hours.
    """
    data = pd.read_excel(
        file_path,
        skiprows=skip_rows,
        sheet_name=sheet,
        header=None,
        index_col=1,
        nrows=num_samples + 1,
    )

    # Column names of the excel sheet.
    names = ['', 'Flux (lm)', 'Vf (V)'] + list(data.iloc[0][3:])
    data.columns = names

    # Get rid of the first row and column, as they do not contain any useful
    # information. Due to the setup of the sheet, the entire column is 'NaN'.
    # Same for the first row.
    data = data.drop(data.columns[[0]], axis=1)
    data = data.drop(data.index[[0]])

    # Number of hours tested, corresponding to the columns of data_values
    hours_tested = np.array(data.columns[2:], dtype=float)
    data_values = np.array(data[data.columns[2:]])

    return hours_tested, data_values


def get_lifetimes(hours_tested, data_array, lifetimes, threshold_fraction=0.7):
    """Fit the lumen maintenance data to a decaying exponential to estimate the lifetime. """
    for row in data_array:
        fit = curve_fit(decaying_exp, hours_tested, row / 100, p0=[1, 0])

        a, b = fit[0]
        lifetime = -1 / b * np.log(threshold_fraction / a)
        lifetimes.append(lifetime)

    return lifetimes


def decaying_exp(x, a, b):
    """Function used to fit the lumen maintenance data to a decaying exponential. """
    return a * np.exp(-b * x)


# User inputs
num_samples: int = 24               # Number of samples tested

use_level_stress_T = (30 + 273)     # Temperature under normal use [K]
use_level_stress_I = 0.1            # Current under normal use [A]

# Rows in the Excel sheet which specify the test conditions.
test_condition_rows = [0, 40, 80]

# Due to the exponential fit, some lifetime estimates are unreasonably high.
# Estimated lifetimes above this threshold are removed from the dataset for better visualisation.
lifetime_threshold = 4E6

lifetimes: list[float] = []

file_path: str = askopenfilename()
num_sheets: int = len(pd.ExcelFile(file_path).sheet_names)

stresses: list[tuple[float, float]] = []

for i in range(num_sheets):
    for j in test_condition_rows:
        # The actual data values (including the temperature values) are 6 rows
        # lower than the test condition row
        stresses.append(get_test_conditions(j, file_path, i))
        hours_tested, data_values = get_data_values(j + 6, file_path, i, num_samples)
        lifetimes = get_lifetimes(hours_tested, data_values, lifetimes)

num_test_conditions = len(stresses)

# Generate stress arrays from the list of stresses.
# The stress arrays need to be as long as the lifetime array, so they need
# a transformation.

# stresses_1: temperature stresses [K]
stresses_1 = np.zeros((num_test_conditions, num_samples), dtype=int)

for i in range(num_test_conditions):
    for j in range(num_samples):
        stresses_1[i, j] = stresses[i][0]


# stresses_2: current stresses [A]
stresses_2 = np.zeros((num_test_conditions, num_samples), dtype=float)
for i in range(num_test_conditions):
    for j in range(num_samples):
        stresses_2[i, j] = stresses[i][1]


# The reliability library expects a 1D array for the stresses.
stresses_1.flatten()
stresses_2.flatten()

# Clean up errenous data
lifetimes = np.array(lifetimes, dtype=float)

# The reliability library does not take in negative lifetimes, so these datapoints need to be removed.
wrong_data_indexes = np.where((lifetimes < 0) | (lifetimes > lifetime_threshold))[0]
lifetimes = np.delete(lifetimes, wrong_data_indexes)
stresses_1 = np.delete(stresses_1, wrong_data_indexes)
stresses_2 = np.delete(stresses_2, wrong_data_indexes)

# %% - Fit the data
model = Fit_Everything_ALT(
    failures=lifetimes,
    failure_stress_1=stresses_1,
    failure_stress_2=stresses_2,
    use_level_stress=[use_level_stress_T, use_level_stress_I]
)

plt.show()
