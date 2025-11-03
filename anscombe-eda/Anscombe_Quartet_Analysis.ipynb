{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "3ad811ac-752d-404f-9e68-e0a076214fda",
   "metadata": {},
   "source": [
    "\\begin{titlepage}\n",
    "\\centering\n",
    "\\vspace*{4cm}\n",
    "\n",
    "{\\Huge \\textbf{Anscombe Quartet Data Visualisation}}\\\\[1.5cm]\n",
    "\n",
    "{\\Large \\textbf{Author:} Brian Lim}\\\\[0.3cm]\n",
    "{\\Large \\textbf{Course:} TER3 F2025}\\\\[0.3cm]\n",
    "{\\Large \\textbf{Instructor:} Mr. Andrade}\\\\[0.3cm]\n",
    "{\\Large \\textbf{Date:} October 31, 2025}\\\\[4cm]\n",
    "\n",
    "\\vfill\n",
    "\\end{titlepage}"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5b93db1b-ce9f-4705-a998-a45c3c481f30",
   "metadata": {},
   "source": [
    "\\newpage"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "138a459b-8277-412a-af3d-d84af4534ce7",
   "metadata": {},
   "source": [
    "## Executive Summary: \n",
    "This notebook explores Anscombe's quartet using summary statistics and visualizations.  \n",
    "Even though all four datasets share very similar statistical summaries, their scatterplots have very different patterns.  \n",
    "This demonstrates why visual inspection is important in Exploratory Data Analysis"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "154bc96f-abac-4444-9b33-76a08f9f9d6e",
   "metadata": {},
   "source": [
    "## Introduction: \n",
    "Anscombe's quartet consists of four datasets that have very similar statistical properties. Despite this, each dataset has very different patterns when visualized. This highlights a principle in Exploratory Data Analysis, showing that only presenting numerical summaries can be misleading and can withhold vital pieces of information. This notebook performs EDA on Anscombe's quartet using summary statistics, scatter pots with regression lines and residual plots to illustrate the differnces between the datasets and to emphasize the importance of creating a presentation with both visual and numerical analysis. "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9b8a78ed-6bf6-40dc-9a0c-d6443c83d3ba",
   "metadata": {},
   "source": [
    "## Data:\n",
    "The dataset being used is Anscombe's quartet, which nicludes four datasets labeled I, II, III, and IV. Each dataset contains x and y values. The data is loaded from a CSV file (but in this case hardcoded) into a pandas DataFrame for analysis. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "916c7ecb-205b-48e4-b72a-97b77805d722",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>dataset</th>\n",
       "      <th>x</th>\n",
       "      <th>y</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>I</td>\n",
       "      <td>10</td>\n",
       "      <td>8.04</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>I</td>\n",
       "      <td>8</td>\n",
       "      <td>6.95</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>I</td>\n",
       "      <td>13</td>\n",
       "      <td>7.58</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>I</td>\n",
       "      <td>9</td>\n",
       "      <td>8.81</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>I</td>\n",
       "      <td>11</td>\n",
       "      <td>8.33</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  dataset   x     y\n",
       "0       I  10  8.04\n",
       "1       I   8  6.95\n",
       "2       I  13  7.58\n",
       "3       I   9  8.81\n",
       "4       I  11  8.33"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "\n",
    "data = {\n",
    "    \"dataset\": [\"I\"]*11 + [\"II\"]*11 + [\"III\"]*11 + [\"IV\"]*11,\n",
    "    \"x\": [\n",
    "        10,8,13,9,11,14,6,4,12,7,5,\n",
    "        10,8,13,9,11,14,6,4,12,7,5,\n",
    "        10,8,13,9,11,14,6,4,12,7,5,\n",
    "        8,8,8,8,8,8,8,19,8,8,8\n",
    "    ],\n",
    "    \"y\": [\n",
    "        8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68,\n",
    "        9.14,8.14,8.74,8.77,9.26,8.10,6.13,3.10,9.13,7.26,4.74,\n",
    "        7.46,6.77,12.74,7.11,7.81,8.84,6.08,5.39,8.15,6.42,5.73,\n",
    "        6.58,5.76,7.71,8.84,8.47,7.04,5.25,12.50,5.56,7.91,6.89\n",
    "    ]\n",
    "}\n",
    "\n",
    "df = pd.DataFrame(data)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e39260c6-ecb7-4800-9bdd-deb32d271cb5",
   "metadata": {},
   "source": [
    "## Methods:\n",
    "For each dataset in Ancombe's quartet, the following analyses were performed:\n",
    "- **Summary Statistics:**\n",
    "    - Mean\n",
    "    - Variance\n",
    "    - Standard deviation\n",
    "    - Covariance\n",
    "    - Correlation\n",
    "    - Regression\n",
    "    - Coefficients\n",
    "    - R²\n",
    "- **Visualizations:**\n",
    "    - Scatter plots with regression lines\n",
    "    - Residual plots\n",
    "    - Overlaid comparison plot\n",
    "    - Box plots of X and Y distributions\n",
    "    - Interactive version (link)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab5b6ea7-9784-44bc-bac8-801800e63d70",
   "metadata": {},
   "source": [
    "## Summary Statistics and Linear Regression Formulas\n",
    "\n",
    "Below are the main summary statistics and regression formulas used in this report, along with brief definitions.\n",
    "\n",
    "---\n",
    "\n",
    "##### **1. Mean**\n",
    "*Definition:* The mean is the average of all data points; it represents the central value of a dataset.\n",
    "\n",
    "$$\n",
    "\\bar{x} = \\frac{1}{n} \\sum_{i=1}^{n} x_i\n",
    "$$\n",
    "\n",
    "---\n",
    "\n",
    "##### **2. Sample Variance**\n",
    "*Definition:* Measures how much the data points deviate from the mean, on average (in squared units).\n",
    "\n",
    "$$\n",
    "s^2 = \\frac{1}{n-1} \\sum_{i=1}^{n} (x_i - \\bar{x})^2\n",
    "$$\n",
    "\n",
    "---\n",
    "\n",
    "##### **3. Sample Standard Deviation**\n",
    "*Definition:* The square root of the variance; shows spread of data in the same units as the data.\n",
    "\n",
    "$$\n",
    "s = \\sqrt{s^2} = \\sqrt{\\frac{1}{n-1} \\sum_{i=1}^{n} (x_i - \\bar{x})^2}\n",
    "$$\n",
    "\n",
    "---\n",
    "\n",
    "##### **4. Covariance**\n",
    "*Definition:* Measures how two variables vary together; positive means they increase together, negative means they vary inversely.\n",
    "\n",
    "$$\n",
    "\\text{cov}(X,Y) = \\frac{1}{n-1} \\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})\n",
    "$$\n",
    "\n",
    "---\n",
    "\n",
    "##### **5. Pearson Correlation**\n",
    "*Definition:* Standardized covariance showing strength and direction of a linear relationship between two variables; ranges from -1 to 1.\n",
    "\n",
    "$$\n",
    "r = \\frac{\\text{cov}(X,Y)}{s_X s_Y} = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum_{i=1}^{n} (x_i - \\bar{x})^2 \\sum_{i=1}^{n} (y_i - \\bar{y})^2}}\n",
    "$$\n",
    "\n",
    "---\n",
    "\n",
    "##### **6. Linear Regression (Least Squares)**\n",
    "*Definition:* Finds the best-fitting line \\(y = mx + b\\) that minimizes the sum of squared errors.\n",
    "\n",
    "**Slope:**\n",
    "$$\n",
    "m = \\frac{\\sum_{i=1}^{n} (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^{n} (x_i - \\bar{x})^2}\n",
    "$$\n",
    "\n",
    "**Intercept:**\n",
    "$$\n",
    "b = \\bar{y} - m \\bar{x}\n",
    "$$\n",
    "\n",
    "---\n",
    "\n",
    "##### **7. Coefficient of Determination**\n",
    "*Definition:* Measures how well the regression line explains variability in the dependent variable; ranges from 0 to 1.\n",
    "\n",
    "$$\n",
    "R^2 = 1 - \\frac{\\sum_{i=1}^{n} (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^{n} (y_i - \\bar{y})^2}\n",
    "$$"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b0bf0cc4-b67f-41e5-9e4e-35137d6e5e54",
   "metadata": {},
   "source": [
    "## Summary Statistics:\n",
    "The summary statistics are calculated with the following code, creating the table that calculates the mean, variance, standard deviation, covarance, correlation, slope, intercept and R²"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "3e6ec221-b9a5-4bfe-baa9-14e2a98f7d11",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Dataset</th>\n",
       "      <th>Mean X</th>\n",
       "      <th>Mean Y</th>\n",
       "      <th>Variance X</th>\n",
       "      <th>Variance Y</th>\n",
       "      <th>SD X</th>\n",
       "      <th>SD Y</th>\n",
       "      <th>Covariance</th>\n",
       "      <th>Correlation (r)</th>\n",
       "      <th>Slope</th>\n",
       "      <th>Intercept</th>\n",
       "      <th>R²</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>I</td>\n",
       "      <td>9.0</td>\n",
       "      <td>7.500909</td>\n",
       "      <td>11.0</td>\n",
       "      <td>4.127269</td>\n",
       "      <td>3.316625</td>\n",
       "      <td>2.031568</td>\n",
       "      <td>5.000909</td>\n",
       "      <td>0.816421</td>\n",
       "      <td>0.500091</td>\n",
       "      <td>3.000091</td>\n",
       "      <td>0.666542</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>II</td>\n",
       "      <td>9.0</td>\n",
       "      <td>7.500909</td>\n",
       "      <td>11.0</td>\n",
       "      <td>4.127629</td>\n",
       "      <td>3.316625</td>\n",
       "      <td>2.031657</td>\n",
       "      <td>5.000000</td>\n",
       "      <td>0.816237</td>\n",
       "      <td>0.500000</td>\n",
       "      <td>3.000909</td>\n",
       "      <td>0.666242</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>III</td>\n",
       "      <td>9.0</td>\n",
       "      <td>7.500000</td>\n",
       "      <td>11.0</td>\n",
       "      <td>4.122620</td>\n",
       "      <td>3.316625</td>\n",
       "      <td>2.030424</td>\n",
       "      <td>4.997273</td>\n",
       "      <td>0.816287</td>\n",
       "      <td>0.499727</td>\n",
       "      <td>3.002455</td>\n",
       "      <td>0.666324</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>IV</td>\n",
       "      <td>9.0</td>\n",
       "      <td>7.500909</td>\n",
       "      <td>11.0</td>\n",
       "      <td>4.123249</td>\n",
       "      <td>3.316625</td>\n",
       "      <td>2.030579</td>\n",
       "      <td>4.999091</td>\n",
       "      <td>0.816521</td>\n",
       "      <td>0.499909</td>\n",
       "      <td>3.001727</td>\n",
       "      <td>0.666707</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "  Dataset  Mean X    Mean Y  Variance X  Variance Y      SD X      SD Y  \\\n",
       "0       I     9.0  7.500909        11.0    4.127269  3.316625  2.031568   \n",
       "1      II     9.0  7.500909        11.0    4.127629  3.316625  2.031657   \n",
       "2     III     9.0  7.500000        11.0    4.122620  3.316625  2.030424   \n",
       "3      IV     9.0  7.500909        11.0    4.123249  3.316625  2.030579   \n",
       "\n",
       "   Covariance  Correlation (r)     Slope  Intercept        R²  \n",
       "0    5.000909         0.816421  0.500091   3.000091  0.666542  \n",
       "1    5.000000         0.816237  0.500000   3.000909  0.666242  \n",
       "2    4.997273         0.816287  0.499727   3.002455  0.666324  \n",
       "3    4.999091         0.816521  0.499909   3.001727  0.666707  "
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import statsmodels.api as sm\n",
    "\n",
    "summary = []\n",
    "\n",
    "for name, subset in df.groupby(\"dataset\"):\n",
    "    x = subset[\"x\"]\n",
    "    y = subset[\"y\"]\n",
    "    mean_x = x.mean()\n",
    "    mean_y = y.mean()\n",
    "    var_x = x.var()\n",
    "    var_y = y.var()\n",
    "    sd_x = x.std()\n",
    "    sd_y = y.std()\n",
    "    cov = np.cov(x, y, ddof=0)[0, 1]\n",
    "    corr = x.corr(y)\n",
    "    model = sm.OLS(y, sm.add_constant(x)).fit()\n",
    "    slope = model.params.iloc[1]\n",
    "    intercept = model.params.iloc[0]\n",
    "    r2 = model.rsquared\n",
    "    summary.append([name, mean_x, mean_y, var_x, var_y, sd_x, sd_y, cov, corr, slope, intercept, r2])\n",
    "\n",
    "summary_table = pd.DataFrame(summary, columns=[\n",
    "    \"Dataset\", \"Mean X\", \"Mean Y\", \"Variance X\", \"Variance Y\",\n",
    "    \"SD X\", \"SD Y\", \"Covariance\", \"Correlation (r)\", \n",
    "    \"Slope\", \"Intercept\", \"R²\"\n",
    "])\n",
    "\n",
    "summary_table"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f435183-cce3-4e46-a865-a1e0993ba94b",
   "metadata": {},
   "source": [
    "## Visualization:\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "b8a9e48d-5256-451c-99b9-f2b66477b379",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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nvrkk090//33e2OffvqpfvvtN7Vr185bxG2BnDRpkvfrN9984y2St912m1avXq0ZM2ZozJgx3kJs8ufP7/1qi7YtvHZnPkOGDHrvvffUvn17bzHs0KGD95pt27apbt263sd0795duXLl8t4I2B3sY9mibaloNp+OHTtq3bp1Gj9+vJYtW6avvvrKu5ttC72l4p133nnq1auX93G24AMAEOpYq/9jX9PDDz/spZHb749lYyVKlNA111xzDt9tIIjFAwhKkydPjrd/oosXLz7ta3LmzBlftWrVxOt9+/ad9JoZM2Z4n2f+/PmJYyNHjvTG1q1bd9LrT/U56tWrF1+6dOnE65iYmDPObcGCBd5rpk+fftz47NmzTxqvVKlS/PXXXx+fXPbxHTp0SLy2r8PG7OtK8Pnnn3tjb7zxRrI/LwAAZ4O1OvlrtWnatGl85syZ42NjYxPHfvnlF++1PXr0SPbnBkIN6eVACLM7zsdWRj32vNiBAwf0999/66qrrvKuly5dmqzPeeznsDQw+xzXX3+9d1fero3dLTfvv/++Dh8+fMrP88Ybbyhnzpy68cYbvc+R8Ljsssu8eX/++ecp/KoBAAgdrNXHp5jb13zsbntCcTVSyxHOCLqBELZnzx5lz5498fqff/7Ro48+6qV82YJsKWV2fsskLMJnYqlkderUUbZs2bwF2z6HpcYd+zlsYb/99tu9qqqW8taoUSPvfNnBgwcTP8+aNWu819t5M/scxz5s3pb2BgBAuGOtPr7gXJ48eY6rYm4p9JdccomXLg+EK850AyFq8+bN3kJ5wQUXJI41a9bMazFixVeqVKni3aW281033XST9+uZ/Prrr6pdu7YqVKjg9dQsVqyYMmXKpA8//NA7U5bwORL6X9vZMztH9vHHH3uFWZ566ilvLOHPtUV8+vTpp/yzEs6lAQAQrlirj2fnw+3rt57cf/75pzZu3OgF/iNGjEjVPwcINgTdQIiyfqDGqpIaqzI6d+5c7452nz59El9ni9mJbCE+FVuU7Q74rFmzVLx48cTx06WXWTqcPQYPHuzdtbbUMKtAet9996lMmTKaM2eOVxTlTG1STjcfAABCGWv1yezPnzhxol577TWvaJt93hYtWqTK5waCFenlQAiy3p8DBw700tESzkClT5/e+9XVLvmPVRw9kaWjmZ07dx43fqrPYXfoLR3tWPam4cQ/x+7Wm4S0NbuTHRcX583zREeOHDnuz7b5nDgXAABCGWv1qVmAby3IrI2aBd6WBn/++eef8+cFghk73UCQ++ijj/TLL794i5+lYtkibu1GrLWG3eW2npomR44cuu6667wULSuYUrRoUa9Fid1FPpEVSDHW9qN58+ZeulfDhg291iKWoma/txYidp7LUsAs9ez3339P/PipU6fqueeeU5MmTby75FYgxl5nc7j55pu919giap9j6NChWr58ufe57c+xu/lWuOXpp5/WHXfckTgfa38yaNAgLwXP/jzrPwoAQChgrU4+29m+6667NGTIEO96wIABKfyuAyHE7/LpAJJuQ5LwyJQpU3yhQoXib7zxxvinn346fteuXSd9zObNm+ObNGkSnytXLq9FibXm2Lp1q/fxffv2Pe61AwcOjC9atGh8unTpjmtJMmvWrPiLL77Ya+lRsmTJ+OHDh8e//PLLx71m6dKl8S1atIgvXrx4fHR0dHyBAgXib7nllvglS5acNKdJkybFX3bZZfFZsmSJz549e3zlypXjn3jiCW9eCf7444/4Bg0aeM/bn3OmliS0DAMABAPW6rNrGZbgxx9/9J63ee3YsSOZ320gdEXZf/wO/AEAAAAACEec6QYAAAAAIEAIugEAAAAACBCCbgAAAAAAAoSgGwAAAACAACHoBgAAAAAgQAi6AQAAAAAIEIJuAAAAAAACJIPC3NGjR7V161Zlz55dUVFRfk8HAIAzio+P1+7du1WkSBGlSxc598dZswEAYblex/to3rx58bfcckt84cKF420qMTExxz3ft2/f+PLly8dnzZo1PleuXPG1a9eO/+abb87qz9i0aZP3uXnw4MGDB49Qe9gaFklYs3nw4MGDh8JwvfZ1p3vv3r265JJLdM899+i222476fly5cpp/PjxKl26tPbv368xY8aobt26Wrt2rfLnz5+sP8PulptNmzYpR44cqf41AACQ2nbt2qVixYolrmGRgjUbABCO63WURd4KApZGFhMTo8aNGyf5ReXMmVNz5sxR7dq1k/V5Ez4mNjaWBRwAEBIide2K1K8bABDe61bInOk+dOiQJk2a5H1Rtjt+OgcPHvQex34jAAAAAADwQ9BXZ3n//fd13nnnKXPmzF56+aeffqp8+fKd9vVDhw71AvOEh233AwAAAADgh6APumvWrKnly5dr4cKFuummm9SsWTNt27bttK/v0aOHt72f8LBzYQAAAAAA+CHog+5s2bLpggsu0FVXXaWXXnpJGTJk8H49nejoaC+f/tgHAAAAAAB+CPqg+1Q9PI89sw0AAAAAQLDytZDanj17vPZfCdatW+elkufJk0d58+bV4MGDdeutt6pw4cL6+++/9eyzz2rLli1q2rSpn9MGAAAAACD4g+4lS5Z4Z7YTdO7c2fu1TZs2mjhxon755RdNnTrVC7gtCL/88su1YMECVapUycdZAwDS0oEDB7R//34tW7ZMGzduVPHixVW1alVlyZLFK7IJAAAQzO8lfA26b7jhBiXVJvztt99O0/kAAIKLFcSMiYlRly5dtH379sRxuxE7atQoNWnSxOtUAQAAEKzvJUKmTzcAIPLuStsi2a5du5Oes0UzYbx58+bseAMAgKB9LxFyhdQAAJHB0sDsrnRS7Hl7HQAAQHLeS1SVFJXG7yUIugEAQcnOXR2bBnYq9ry9DgAAIKn3EhdIekPSUklN0/i9BOnlAICgZIVOkmPTpk0BnwsAAAjN9xL5JPWW9LCkjJLiJFVM4/cSBN0AgKBklUWTo1ixYgGfCwAACDH79+vaL7+UNahOKJP2oaRukn5I4/cSpJcDAIKStfKwyqJJseftdQAAAJ64OGnKFKlcOZV56SUv4LaU8tqSGpwi4E6L9xIE3QCAoGS9M62VR1LseXsdAACAPvlEuvRSyaqSb96s+GLFNP/++1VN0mc+vpcg6AYABCVr3WG9MydPnnzSjne+fPk0ZcoU73nahQEAEOFWrJDq1pXq1ZO+/16yvtsjRihq9WpdMnKkXvb5vURUfHx8vMLYrl27vGbn1hQ9R44cfk8HAJCCHpvWysMqi1qhEzt3ZWlgdlc6XAPuSF27IvXrBgCkkBVAe/JJ6ZVXJAtrM2aUOnRwY8cE2YF6L5HcdYtCagCAoGaLoT1q1arl91QAAEAwiI2Vhg2Txo61iNqN3XmnNGSIVLp00L2XIOgGAAAAAAS/Q4ek55+XBgyQ/v7bjV17rR3Mlq64QsGKoBsAAAAAELzi46W33pJ69JDWWhMwSRUqSMOHSw0bSlFRCmYE3QAAAACA4PTVV1KXLtI337jrggWl/v2le++VMoRGOBsaswQAAAAARI5Vq9zOdkyMu86aVeraVXr8cSl7doUSgm4AAAAAQHDYts3tZNvZ7bg4KV06t6ttY4ULKxQRdAMAAAAA/LVvnzRmjDunvXu3G7vlFlelvFIlhTKCbgAAAACAP+LipKlTpd69pa1b3dhll0kjR0o1ayocEHQDAAAAANK+Ivns2dITT0g//ODGSpZ0vbat57allYeJ8PlKAABAwMyfP18NGzZUkSJFFBUVpXfeeSfxucOHD6tbt26qXLmysmXL5r2mdevW2pqwYwEAwLGWLZNuvFG6+WYXcOfK5Xpt//KL1KJFWAXcJry+GgAAEBB79+7VJZdcomefffak5/bt26elS5eqd+/e3q9vv/22Vq1apVtvvdWXuQIAgtTGjVLr1i59fO5cKVMmV43811/dr9HRCkeklwMAgDOqX7++9ziVnDlz6tNPPz1ubPz48briiiu0ceNGFS9ePI1mCQAISjt3SkOHSk8/LR086MbuuksaNEgqVUrhjqAbAACkutjYWC8NPZelDAIAItOhQ9KECdKAAdI//7ixG25wRdKqVVOkIOgGAACp6sCBA94Z7xYtWihHjhynfd3Bgwe9R4Jdu3al0QwBAAEvkvbGG1KPHtJvv7mxihWlESPcOe6oKEUSznQDAIBUY0XVmjVrpvj4eE2w3Y0kDB061EtNT3gUK1YszeYJAAiQBQukq65yFcgt4C5USJo0SVqxQmrQIOICbkPQDQAAUjXg3rBhg3fGO6ldbtOjRw8vDT3hsWnTpjSbKwAglVnl8caNpeuukxYtkrJlk/r3l9aske6/X8oQuUnWkfuVAwCAVA+416xZo88//1x58+Y948dER0d7DwBACPvzT6lfP+mFF6S4OCl9eum++9yY7XLD351uen4CABAa9uzZo+XLl3sPs27dOu/3Vp3c1uw77rhDS5Ys0fTp0xUXF6c//vjDexyyIjoAgPCzd68rkHbBBdLEiS7gtlaRK1e6awLu4Ai66fkJAEBosIC6atWq3sN07tzZ+32fPn20ZcsWzZo1S5s3b1aVKlVUuHDhxMfChQv9njoAIDVZcP3ii1LZslLfvnZXVrr8cmnePOndd6ULL/R7hkHH1/Ryen4CABAabrjhBq842ukk9RwAIAzYz/mPPpKeeEL68Uc3Zj22rf92s2YRWSAtLM90J6fnJ+1HAAAAACAVffed1LWr9Pnn7jp3bql3b6l9eyvQ4ffsgl66cOv5SfsRAAAAAEgF69dLd98tVavmAm4LsC34/vVX6bHHCLjDKeg+m56ftB8BAAAAgHOwY4cLrsuXl6ZPd2MWfK9aJY0Y4Xa6ET7p5cf2/Pzss8/O2POT9iMAAAAAkAJ2TNeKXA8a5AJvU6uWNHKkdOmlfs8uZGUIt56fAAAAAICzcPSo9NprUs+eLqXcXHSR29W+6SaKpIVy0G09P9euXZt4ndDzM0+ePF6bEev5ae3C3n///cSen8aez5Qpk48zBwAAAIAwMH++1KWLtHixuy5cWBo4UGrbVkqf3u/ZhYUMfvf8rFmzZuK19fw0bdq0Ub9+/byen8Z6fh7Ldr2tdQkAAAAAIAV+/lnq1k167z13fd557toKpGXL5vfswoqvQTc9PwEAAAAgDVn2cL9+0osvSnFxbjf7gQekvn2lggX9nl1YCuoz3QAAAACAVLBnj/TUU64o2t69bqxxY2nYMFelHAFD0A0AAAAA4erIEenll91O9r81snTVVS74rlHD79lFBIJuAAAAAAg3dlT3/ffdOW07v23KlJGGDpXuuIOK5GmIoBsAAAAAwolVIu/aVZo3z11b6+U+faSHHpLoApXmCLoBAAAAIBz89pvUq5c0c6a7jo6WOnWSuneXcuXye3YRi6AbAAAAAELZP/9IgwZJ48dLhw+71PFWrVy/7eLF/Z5dxCPoBgAAAIBQdOCAC7QHD5Z27nRjdeq4ImlVqvg9O/yLoBsAAAAAQsnRo9KMGS6VfMMGN1a5sgu269Xze3Y4AUE3AAAAAISKzz+XunSRli5110WLujTy1q2l9On9nh1OgaAbAAAAAILdjz+69l8ffOCus2d3BdKsUFrWrH7PDkkg6AYAAACAYLV1q9S3r/Tyyy6tPEMG6cEHXQuwAgX8nh2SgaAbAAAAAILN7t3ujPZTT0n79rmx22+XhgyRypXze3Y4CwTdAAAAABAsjhyRXnzR7W5v2+bGrr5aGjVKql7d79khBQi6AQAAAMBv8fHSrFnu3PaqVW6sbFlp2DCpSRPXexshiaAbAAAAAPz07bdS167SggXuOl8+t9NtZ7czZvR7djhHBN0AAAAA4Idff5V69JDeeMNdZ84sde4sPfGElDOn37NDKiHoBgAAAIC09Pffrrf2hAnS4cMudbxNGzd2/vl+zw6pjKAbAAAAANLC/v3SuHHS0KFSbKwbq1dPGjFCuvhiv2eHACHoBgAAAIBAsv7a06dLvXpJmza5sSpVXLB9441+zw4BRtANAAAAAIEyd64rkrZsmbsuVkwaNEhq2VJKn97v2SENpEuLPwQAAIS2+fPnq2HDhipSpIiioqL0zjvvHPf822+/rbp16ypv3rze88uXL/dtrgAQFFaulOrXl+rUcQF3jhyu/Ze1A2vdmoA7ghB0AwCAM9q7d68uueQSPfvss6d9vkaNGho+fHiazw0AgsqWLdK997r08dmzXcuvRx91lcqtB3eWLH7PEGmM9HIAAHBG9evX9x6n06pVK+/X9evXp+GsACCI7NrlzmiPHu0KppmmTV3RtDJl/J4dfETQDQAAAAApZS2/Jk2S+veX/vrLjV1zjTRqlHTVVX7PDkGAoBsAAPji4MGD3iPBLtslAoBQER8vxcRI3btLa9a4sXLlJDtm06iR670N+H2mm6IsAABErqFDhypnzpyJj2JW0RcAQsHXX0vXXivdfrsLuPPnl6zmxQ8/SI0bE3AjeIJuirIAABC5evToodjY2MTHpoTetQAQrCzAvuMOqXp16auvXFG0J5+U1q6V2rd3RdOAYEovpygLAACRKzo62nsAQNCzs9oDB0oTJkhHjkjp0klt20oDBkhFi/o9OwQ5znQDAIAz2rNnj9baTs6/1q1b5x37ypMnj4oXL65//vlHGzdu1NatW73nV1kfWkmFChXyHgAQkqwK+dixrr92Qt0J2zS0TNzKlf2eHUJE2PXptoIsVojl2AcAADg3S5YsUdWqVb2H6dy5s/f7Pn36eNezZs3yrhs0aOBdN2/e3LueOHGir/MGgBSJi5OmTnWF0Xr2dAG3/fybM0f68EMCbkT2TrcVZelv5foBAECqueGGGxRvlXpPo23btt4DAELeJ59ITzwhrVjhrosXl4YMkVq0cGnlwFkKu781FGUBAAAAcNYsyK5Xzz3s9zlzSiNG2HkZqWVLAm6kWNjtdFOUBQAAAECy2SZd797S//2f671tFcgfeUTq1UvKm9fv2SEM+Bp0U5QFAAAAgC9iY12BNCuUduCAG2veXBo8WCpd2u/ZIYz4miNBURYAAAAAaerQIemZZ6QLLnBBtwXc110nffutNGMGATdSXVR8UlVRwoBVL8+ZM6d3vjtHjhx+TwcAgDOK1LUrUr9uAGnEwp633rIiUFJCtm2FCq79V8OGUlSU3zNEmK5bYXemGwAAAACO89VXUpcu0jffuOuCBSXreHTvvVIGQiIEFn/DAAAAAIQnqwllO9sxMe46a1apa1fp8cel7Nn9nh0iBEE3AAAAgPCybZvbyX7+eSkuzrX7sl1tGytc2O/ZIcIQdAMAAAAID/v2SaNHu3Pae/a4sVtucdcVK/o9O0Qogm4AAAAAoc12s6dOdf22/203rMsuk0aNkm64we/ZIcIRdAMAAAAI3Yrks2dLTzwh/fCDGytZUhoyRLrzTpdWDviMoBsAAABA6Fm2zBVFmzvXXefKJT35pPTII1J0tN+zAxIRdAMAAAAIHRs3uuB62jS3050pk/S//0k9e0p58vg9O+AkBN0AAAAAgt/OndLQodLTT0sHD7qxu+6SBg2SSpXye3bAaRF0AwAAAAhehw5Jzz0nDRwo/fOPG7PiaCNHStWq+T074IwIugEAAAAEH0sdf+MNqUcP6bff3Ji1/RoxQrr5Zikqyu8ZAslC0A0AAAAguCxYIHXpIi1a5K4LF5YGDJDatpUyEMIgtPA3FgAAAEBw+OUXqXt36d133XW2bK4d2OOPu98DIYigGwAAAIC//vhD6t9feuEFKS5OSp9euv9+qV8/qWBBv2cHnBOCbgAAAAD+2LtXeuopd07bfm8aNZKGDZMqVPB7dkCqIOgGAAAAkLaOHJEmT5b69pV+/92NXXGFq0h+3XV+zw5IVQTdAAAAANKuIvmHH7pz2j/95Masx7b1327WjIrkCEsE3QAAAAAC77vvpK5dpc8/d9d58ki9e0sPPyxFR/s9OyBgCLoBAAAABM769VKvXtKrr7prC7A7dpR69pRy5fJ7dkDAEXQDAAAASH07dkiDB0vPPCMdOuTG7r5bGjRIKlHC79kBaYagGwAAAEDqOXhQevZZF1xb4G1q1XJF0i691O/ZAWmOoBsAAADAuTt6VHrtNZc2binl5qKLXDuwm26iSBoiFkE3AAAAgHPzxReuSNqSJe66cGFp4ECpbVspfXq/Zwf4Kp2/fzwAAAgF8+fPV8OGDVWkSBFFRUXpnXfeOe75+Ph49enTR4ULF1aWLFlUp04drVmzxrf5Akgj1varYUOpZk0XcJ93ngu27d//vfcScAME3QAAIDn27t2rSy65RM/aOc1TGDFihMaNG6eJEyfq22+/VbZs2VSvXj0dOHAgzecKIA38/rv04INS5crS+++74Npaf61dKz35pJQtm98zBIIG6eUAAOCM6tev7z1OxXa5x44dqyeffFKNGjXyxv7v//5PBQsW9HbEmzdvnsazBRAwe/ZIo0a5x969bqxxY2nYMKl8eb9nBwQlX3e6SVUDACD0rVu3Tn/88Ye3TifImTOnrrzySn399de+zg1AKjlyRHr+eemCC6T+/V3AfdVV0oIFUkwMATcQrEE3qWoAAIQ+C7iN7Wwfy64TnjuVgwcPateuXcc9AASZ+Hjpvfekiy+WHnpI+vNPqUwZ6Y03pIULpRo1/J4hEPR8TS8nVQ0AgMg1dOhQ9bcdMwDBafFiV5F83jx3nTev1KePC74zZfJ7dkDISBduqWrcNQcApKm//5b69ZMefVSRqlChQt6vf9oO2DHsOuG5U+nRo4diY2MTH5s2bQr4XAEkw2+/SS1aSFdc4QLu6GipWzdXJK1jRwJuIFyC7pSmqtldcwvOEx7FihUL+FwBABFo40YXaJco4c432lGpDRsUiUqVKuUF13Pnzk0cs5vedjTs6quvPu3HRUdHK0eOHMc9APho+3bpscekChWkmTOlqCipTRvX/ssKpeXK5fcMgZAUdtXL7a55586dj1v0CbwBAKnmxx+t6Ij06quusJC59FKpe3fp/PMVrvbs2aO1tst1TEba8uXLlSdPHhUvXlydOnXSoEGDVLZsWS8I7927t1cotbFVNQYQ3Kxe0jPPSIMHS7GxbuzGG93PuipV/J4dEPIyhEKqmlUvT2DXVZL4x293ze0BAECqsqNNttMza9Z/Y7Vq2d1eqXZttyMUxpYsWaKaNWsmXifc4G7Tpo2mTJmiJ554wiuQ+sADD2jnzp2qUaOGZs+ercyZM/s4awBJOnpUmjFD6tXrv0wdK5g2cqRUt67fswPCRoZQSFVLCLITUtUefvhhv6cHAIiUqr0ffSQNH259Lt2YBde33ebON15+uSLFDTfc4BU5PR1r/TlgwADvASAEfPaZK5K2dKm7LlpUGjRIatVKSp/e79kBYcXXoJtUNQBAULK08ddec8H2ypVuLGNGqXVr9yaVfrQAQtUPP0hPPOFuKJrs2V3GjtWoyJrV79kBYcnXoJtUNQBAUNm3T5o8WRo1Slq/3o2dd5704IOuuJDtBAFAKNq61bX7sp9xllaeIYNk2aO9e0v58/s9OyCsRcUnlSsWBiwl3aqYWysSqqICAE5pxw5XfXzcOOmvv9yYvQm1nZ/27aXcudN0OpG6dkXq1w0E1O7d7oz2U0+5G4vm9tut5Y9UtqzfswMiYt0K2jPdAAAE3JYt0pgx0vPP25knN1aypEshb9dOypLF7xkCQMocPiy9+KLUr5+0bZsbq17dZfIk0coPQOoj6AYARJ5Vq9zOz//9n3tjmlCx14qjNWvm0i4BIBRZEqt1WbCfZ/azztiOtnVfaNIk7DstAMGIdxUAgMixeLF74xkT496YmmuvdUWEbrqJN6MAQtu337pMnQUL3HW+fFLfvq4uhRWDBOALgm4AQHiz4PrTT12w/fnn/43feqvbCbJ0SwAIZb/+KvXsKb3+uru2osNWoNh+xlEfAfAdQTcAIDzFxUlvveWC7WXL3Jiljd91l2uXU6mS3zMEgHOzfbs0cKD03HPuqIxl67Rp48bOP9/v2QH4F0E3ACC8HDjgzmqPGOF2f4z1nr3vPunxx6Xixf2eIQCcm/37XbcFq0AeG+vG6tVzP/esPgWAoELQDQAID/bGc+JEV438zz/dWJ48UseO0iOPSHnz+j1DADg31l972jTpySelTZvcWJUqLti+8Ua/ZwfgNAi6AQCh7Y8/pKefdumVu3a5sWLF3HnG+++XsmXze4YAcO7mzHFF0pYv/+/n3KBB0t13S+nS+T07AEkg6AYAhKa1a12/2SlTpIMH3VjFiq5wUIsWVOoFEB6+/97Vofj4Y3dthdGsaJpl8WTJ4vfsACQDQTcAILRYUbThw6U33nCplubqq12w3bAhOz4AwsPmzVKfPu7GonVhsBuJ7du71HJrBQYgZBB0AwCCn73h/OILV4n8k0/+G7/5Zql7d6lGDXpsAwgPdkzGbixafQormGaaNnVF08qU8Xt2AFKAoBsAELxsJ/udd9wb0EWL3JjtZDdv7tItL7nE7xkCQOqwll/PPy/17y/9/bcbsxuKI0dKV13l9+wAnAOCbgBA8Dl0yFXotYq8q1a5scyZpXvvdW2/SpXye4YAkHqZPDExLmtnzRo3Vq6cu9nYqBFZPEAYIOgGAASP3bulF16QRo+WtmxxY7lySR06uKJBBQr4PUMASD0LF7qK5ParsZ9x/fpJ991HMUggjBB0AwD899df0rhx0vjx0s6dbqxIEdf264EHpOzZ/Z4hAKQe29Hu0UN66y13bVXILYvHjs3w8w4IOwTdAAD/rF8vPfWU9NJL/xUMsrRKe+NpvWejo/2eIQCk7g3GAQOkiROlI0dcjYp27dyY3WgEEJYIugEAaW/lSndee8YMKS7OjVWr5s40Nm4spU/v9wwBIPXYTcWxY10HBqtOburXdz8HL7rI79kBCDCCbgBA2vnyS/em84MP/hu78UYXbNesScEgAOHFbiq+8orUu7fru22qVnUVyWvX9nt2ANIIQTcAIPBtvyzItkq8X33lxiy4vuMOqVs36bLL/J4hAKS+Tz5xR2VWrHDXxYtLgwdLd93l0soBRAyCbgBA4HrOzpzpgu0ff3RjmTJJbdtKXbpIZcv6PUMASH0WZFuwbUG3yZlT6tVL+t//XOtDABGHoBsAkLr27XOF0axA2oYNbsyq8T78sPTooxQLAhCeNm1yaeT/93+u97a1/HrkERdw583r9+wA+IigGwCQOrZvl559VnrmGenvv//rOdupkwu4rd82AISb2FhXq8IKpR044MbuvFMaMkQqXdrv2QEIAgTdAIBz390ZPVp64QVp7143Zm80u3aV2rRx/WcBINwcOuRaf1m7L7vpaK69Vho1SrriCr9nByCIEHQDAFLm559du5tp01y/WVOliqtEfvvtUgaWGABhyFLH33pL6tFDWrvWjVWo4Ha7b72VLgwATsI7IgDA2fnmG/fm8t13/xu74QYXbNetyxtOAOHLOjBYIUj7OWgKFpT695fuvZcbjQBOi34FAIDk7ezMnu2C66uv/i/gbtzYvfn8/HOpXj0C7gi3e/duderUSSVKlFCWLFlUvXp1LV682O9pAedu1SrpttukGjXcz7ysWaU+faQ1a6QHHyTgBhDaQTcLOAD4yNLGre1X1apS/frSvHmuIm+7dtJPP0kxMdKVV/o9SwSJ++67T59++qleeeUVrVy5UnXr1lWdOnW0ZcsWv6cGpMy2bVKHDlKlSu7nnfXXvu8+F2zbDrd1ZgCAUA+6WcABwAf797sCQeXLSy1auL6z2bJJjz0m/fab9PLL0oUX+j1LBJH9+/frrbfe0ogRI3TdddfpggsuUL9+/bxfJ0yY4Pf0gLNvfThokFSmjPTcc1JcnHTLLdL337uikbQ+BHAWMoTCAv7uu+96C7ixBfy9997zFvBB9sMQAJB6du6ULECy1je2w2Osv6z117bdnjx5/J4hgtSRI0cUFxenzJkzHzduWWpffvmlb/MCzooF11Onun7bW7e6scsuk0aOlGrW9Ht2AEJUhnBbwA8ePOg9EuzatSvg8wSAkGdvLi3Qtt3t3bvdWPHi0uOPuwJBtssNJCF79uy6+uqrNXDgQF144YUqWLCgZsyYoa+//trb7T4V1mwEVd2Kjz+WnnhCWrnSjZUs6XptW89tSysHgBRKFyoL+NatW70AfNq0ad4C/vvvv5/yY4YOHaqcOXMmPooVK5bm8waAkLF6tXT//VKpUm4nxwLuiy6SXnnFtcLp2JGAG8lmR8Hi4+NVtGhRRUdHa9y4cWrRooXSnSZgYc1GUFi6VLrxRle3wgLuXLlcr+1ffnHHawi4AZyjqHhbHYPYr7/+qnvuuUfz589X+vTpdemll6pcuXL67rvv9LP1iE3GXXNbxGNjY5UjR440nj0ABKklS6Thw12v2YRlwKrydusmNWhAFfJkOnDggHcUatmyZdq4caOKFy+uqlWrehlZJ2ZpnQ1buywIDdW1a+/evd7XULhwYd15553as2ePPvjggzRZswP1/wRhaMMG6cknpWnT3HWmTNL//if17MlRGgCpul6f9a27Nm3aeAFwWilTpozmzZvnLdibNm3SokWLdPjwYZUuXfqUr7c76/YFH/sAAPybPjlnjtvRufxy6c033VjDhpId2VmwwBUKIuBOFltgZ86cqbJly6p27dpq166d96td27g977e0XrMTZMuWzQu4d+zYoY8//liNGjVKkzU7FP6fIEhqV1gauRWKTAi477rL7WzbDjcBN4BUdtZBty1YVj3cFrAhQ4akWRXx5C7gAIBTFAayAPuKK1zAbYF3+vRSq1YulXLWLOmaa/yeZUix3dSYmBgvqNu+fftxz9m1jdvz9jo/pfWabevz7NmztW7dOq/zSM2aNVWhQgXv+xFoofL/BD46dMjVrrCK5HacxrIsbrhBsla006e7YzYAECzp5X/99Zd3bmvq1Kn66aefvAX93nvv9QLhjNa/NZUXcJti+fLltXbtWnXt2tVLD1uwYEGy/qxQT9EDgBSzN5R2NnvECNdT1mTJ4nrMWoG0EiX8nmHIshvAFsieGNwdK2/evFqzZo1y58591p8/NdeutFyzX3/9dfXo0UObN29Wnjx5dPvtt2vw4MHe1xLorzvQ/08Qwuyt7uuvu7Rxa3loKlZ0PxtvvpnsHgDBl15u8ufPr86dO2vFihX69ttvvaqkrVq1UpEiRfTYY495C1pqsS+gQ4cO3p3y1q1bq0aNGl4gntpvFAAgbFgFaNvFsV0bK5JmP5MtyOjTx51hHDeOgPsc2XnhpII7Y8/b6/yWlmt2s2bNvFosdk7bCp6OHz8+2QF3JP0/QRqyYzNXXSU1b+4C7sKFXZ/tFSuoXwEgzZxTOUZbUC19zB5W5Ozmm2/WypUrVbFiRY0ZMybkF3AACCl//in16uVafdl5RevyULSo9NRT0saNUv/+FoH5PcuwYAW6ksNqkQSLtFiz/RSK/08QQHY+244iXnedtGiR68JgPwPtJpNl+2QI6q65AMLMWf/EsSJms2bN0uTJk/XJJ5/o4osvVqdOnXTXXXclbqnbmSmrOG530AEAAWa7NxZYv/yyHWx1YxUquErkVhzIKvIiVVlF7OTwuwVWJK3ZofL/BGlw87FfP7ebbfUsrH6FZfz07SsVKuT37ABEqLMOuq2Y2dGjR72+m1ZJvEqVKie9xgqn5LIehwCAwLH0SGv79dpr0tGjbsyKpfXoId16K71lA8haUNn54DOdH7bX+SmS1uxQ+X+CANm71918tHPa9ntjO91Dh0oXXuj37ABEuLMOui0FrWnTpkn2urTF2yqXAgACUBDIWkANGybNnv3feL16Uvfu0vXXc0YxDVjP51GjRiVZlduet9f5KZLW7FD5f4JUduSINGWKq1lhR2oSbj5aXQtLLQeAIHDW2yBWfCWpxRsAEAC2k/3uu1L16q7FjQXctpN9553S0qXu2sYJuNOErYNNmjTx0rZt9/RY+fLl05QpU7zn/V4vI2nNDpX/J0jFG5AffihZ9oalj1vAXbq0y/z55hsCbgCh3zIslNAyDEDI95V99VWXMvnzz24sOlqy3bwuXVy/WfjGej7v37/fq4htBbrsvLClL9tu6rkEd5G6dqXG1x2o/ycIIt99J3XtKn3+ubvOk0fq3Vt6+GH38xEAgmzdonQjAASjPXukF190ZxQ3b3Zj9sO8fXupUyepYEG/Z4h/d1ftUatWLb+ngn/x/ySMrV/vOjTYjUhjAfajj7o6FmFQlwBA+CLoBoBg8vff0vjx0jPPSP/848as4q4F2g89JNEyEUCk2bFDGjzY/Vy07B/TqpU0cKBUooTfswOAMyLoBoBgYD2GR492bW727XNjF1zgUihbt7btO79nCABp6+BBdxPSAm4LvE3t2q5IGlXoAYQQgm4A8NOPP7rz2pYuaVV4zWWXuUrkTZq4HrMAEGmFI60gWs+eLqXcXHSRC7atUwMFIwGEGIJuAPDDwoWu7dd77/03Zjs4Fmzbr7ypBM65kNrGjRtVvHhxCqmFknnzXJHIJUvcdZEiLo28TRtuQgIIWQTdAJDWLW6GD5cWLHBjFlzffrvUrZtUrZrfMwRCmlWPjYmJUZcuXbR9+/bEcWshZj26rWWYVZlFEPrpJ/dz8P333fV557mbkI89JmXN6vfsAOCcEHQDQKBZ2rilSlqwvXKlG8uY0e3c2JntcuX8niEQFjvcFnC3s3Z6J7AAPGG8efPm7HgHE+uv3bev9NJLLq3cdrMffNCNFSjg9+wAIFUQdANAoFhBtMmTpVGj/juXaLs3VoXcqpEXLer3DIGwYSnltsOdFHu+UaNGBN3B0hbRfjbaY+9eN2Z1LIYOlcqX93t2AJCqCLoBILVZld1nn5XGjZP++suN5c/vAu2HH5Zy5/Z7hkDYsTPcx6aUn4o9b6+jh7fPmT8vv+x2sv/4w41ddZUrklajht+zA4CAIOgGgNSyZYs0Zoz0/PNuF8eULOlSyC21NUsWv2cIhC0rmpYcmzZtCvhccJqaFnZe285t//yzGytTxhWUtLoWFI8EEMYIugHgXP3yi9uleeUV6fBhN3bxxa4IUNOmUgZ+1AKBZlXKk6NYsWIBnwtOsHixq0g+f767zptX6tPHHbXJlMnv2QFAwPFOEABSatEiVxwtJsbt4phrr5V69JBuuomdGyANWVswq1KeVIq5PW+vQxr57TfXa9sKSRo7S2/HbOyGJFXkAUSQdH5PAABCigXXn3wi2ZnQK6+U3n7bjTVq5Hpv205O/foE3EAasz7c1hYsKfa8vQ4BZjc+OneWKlRwAbf9PGzdWlq92hVKI+AGEGHY6QaA5IiLk956y50/XLbMjVna+F13uTOKFSv6PUMgollFcuvDbU7s050vXz4v4G7cuDGVywPpwAHpmWekIUOknTvd2I03SiNGSFWq+D07APANQTcAnOlN5NSp7sz2r7+6saxZpQcekB57zA6SKlx6HFvLJavsbAWp7HyspeHariBBCkJFzpw5vT7c1hbM/i5b0TQ7w83f5QCz/tozZki9ekkbNrixypXdz8169fyeHQD4jqAbAE4lNlaaONFVI//zTzeWJ4/UsaP0yCOuEFCYiI2NVUxMzEm7g3b+1XYHbffQghkgFFhgbQ/agqWRzz5zHRqWLnXXRYtKgwZJrVpJ6dP7PTsACAoE3QBwLOsbO3asNGGCtGuXG7Nqx48/Lt13n5Qtm8KJ7XBbwN3OWpqdwALwhHHbPWSXEECiH3+UnnhC+vBDd509uysi+eijLhsIAJCIQmoAYNaulR580PXVtorkFnDbOW1LLbe0cnsjGWYBt7GUctvhToo9b68DAG3d6m5AWltEC7ittoVl/9jPSQu6CbgB4CQE3QAim6VE3nmnVL68NGmSdPCgdPXV0qxZ0sqVruJuxowKV3buNakWS8aet9cBiGC7d7ve2mXLSi+95M5x33679NNPrnha/vx+zxAAghbp5QAij7X4+vxzt6Nt7b8SNGjg+sfWqKFIYUXTksMKUgGIQIcPSy++KPXrJ23b5saqV7f+a+4GJQAgtHe64+Li1Lt3b5UqVcqrOlqmTBkNHDhQ8faGGQBS0vbL+mpbf+3atV3AbYV+rO3XihXS++9HVMBtrEp5clgFaAARxN5rvfOOq0Levr0LuG2X236GfvklATcAhMtO9/DhwzVhwgRNnTpVlSpV0pIlS7yiPlZFt6NVEAaA5LCU8enTXa/YVavcmBUFs3OJViDNznFHKGulZFXKk0oxt+ftdQAixDffuIrkFlwbSx3v29e1Sgzj4zYAEJE73QsXLvR6bTZo0EAlS5bUHXfcobp162rRokV+Tw1AqJxBfOopqXRp6d57XcCdK9d/vWTtHGIEB9zGsoisLVhS7Hl7HZAUstPCgBVDa9bM7WJbwG03J3v2dIUmO3Qg4AaAcNzprl69uiZNmqTVq1erXLlyWrFihb788kuNHj3a76kBCGaWBmkB9fjx0s6dbqxIEalzZ7dTY61t4LE2YNaH25zYpztfvnxewN24cWPaheGMyE4LYX//LQ0c6Fol2hnuqCipbVtpwADp/PP9nh0AhLygDrq7d++uXbt2qUKFCkqfPr13F33w4MFq2bLlaT/m4MGD3iOBfTyACLF+vSvuY5V1DxxwY+XKuV6yd98tRUf7PcOgZEGR9eG2zCKrUm5F0+wMt6WU244lATfONjvNWIbajBkzyE4LZtYKcNw4acgQ1ybR1KvnjuJYSzAAQPgH3a+//rqmT5+uV1991btrvnz5cnXq1ElFihRRmzZtTvkxQ4cOVf/+/dN8rgB8ZK29rBL5zJmuWJq5/HJXibxRI1csDUmywNoetWrV8nsqCFFkp4UQa/c1bZr05JPWmsCNVakijRwp1anj9+wAIOxExQfxYSvbabHd7g52juhfgwYN0rRp0/TLL78ke6fbPk9sbKxy5MiRJvMGkEbszOGwYdIHH/w3Vreu1K2bVLOmS5EEQpCtXZaBEEpr19GjR9WzZ0+NGDHiuOy0Hj16nPZjArFmHzhwQPv37/eyNqwlnlXoJ2vjGHPmuCJpy5e7a+tMMGiQywZKF9SlfgAgZNfroN7p3rdvn9KdsADYQm4L++lER0d7DwBhyv79W2sv29leuNCN2c+JO+5wwfallypYEQwgnAVDdpq96YmJiTmpPoFV4Lf6BFa/wN4cRaTvv3dHbT7+2F3bm0Mrkmbn7SmUCAABFdRBd8OGDb275PbG1BZwe6NqaWr33HOP31MDkNasuM+MGS7Y/uknN5Ypkyv206WL6x8bxAgGEO66du3qZadZfQBTuXJlbdiwwQusTxd02y54ZytweMJOd0pvatm/MSvediL7N5cwbvOLqJtcmzdLffpIU6a43ttWgdz6bltqeb58fs8OACJCUAfdzzzzjNd+pH379tq2bZt3t/zBBx9UH1s8AESGvXulF190rb8Szh7aDs3DD0uPPioVLqxgRzCASOB3dpplkdhNraTY81bsLSL+nVlhNLtJOWaMK5hmmja19AKpTBm/ZwcAESWog+7s2bNr7Nix3gNAhLHdYGv5Za2/EnaGCxaUHntMeughK7mtUEEwgEjgd3aa/XnHZpGcij1vrwvrgoGWFTRpkmRp+3/95cZq1HBF0q66yu/ZAUBECuqgG0AEst1sq3Zsbxr37XNjtitjhX8sRTUEg1KCAUQCv7PTrE5CclhLvLBkqeMxMa5rw5o1bqx8ebfbfeutFJYEAB8RdAMIDnZO23rDTp8uHTnixqpWdW8gb789pNt+RXwwgIjgd3aa7bAnR0rPjAe1r792Nya/+spdFygg9esn3XefO8MNAPAVQTcAf33zjWv79e67/41Zuy8Ltm+8MSx2ZyI6GADSiHUCsMKESWWV2PP2urBhO9rWku2tt9x11qzS44+7ADx7dr9nBwD4Fw0ZAfiTBvnRR9L110tXX+0CbguumzRxQfhnn7l+22EQcB8bDCQl7IIBII1Z6z3rBJAUe95eF/LsrLa1+qpY0QXcVsDOdrUtCB8wgIAbAIIMQff/t3cn8FVV1x7H/wEkWoUABoSoBFEGmUSL0McgZVBqkZL6BOEFRQQHhIIgimhtQSsICqKoUcCCs1Y/BsGZQRArAvpU4KkQBi2II/OMmvs+62wDJGQm955z7/19P5/7as69kJ3zwj5nnb32WgAix9LGn31Wat5c+uMfpXffdamPVmjJ0stffllq1UqxJq6CAcBHrVq10kMPPXTUQ67k5GQ9/PDD3vtRzepcWPXxs85yRSZtTrW59NNPpWnTpJQUv0cIAMgH6eUAws/a1cyYYZGltGGDO3biidJ117lq5KedplhmFcmtD7fJ26fbggELuNPS0qhcDhxjl4D27dt7PcHfe+89rVq1Sps3b/YKujVp0kTTp0/X6NGjtXr16uj7t/bLL9JTT0l33OH6bpvzznMVySm+CACBlxAKWZ5n7Nq5c6eSkpK0Y8cOVbbevgAiZ/t26ZFHpAcekL7/3h1LTnZpkYMGSdWqKZ5Yv24LDKxKuRVNsz3cllJuK9xRFwQgrOL12nUsP/eCBQvUqVMn77+tX3jLli1VrVo1bd26VcuWLTvUL3z+/PnR1SXg7bfdHu0VK9zXViNi7Fipd2+XVg4ACPx1i5VuAGVv82bp/vulxx6Tdu1yx1JTbZnXpZJbsZ84ZIG1vaLqhh+Iwi4BFmB/YPUhorlLgKWM33KLC7pNUpJ0++3SX/4Sla0TASCeEXQDKDtr1rh0xyeflA4edMeaNJFGjpQuvzzsrWuOXEm2G3CrGs5KMhAfYqZLgD0UsDRym0ctGdHmzcGDXcBdREFGAEAwEXQDOHbLl0vjx7tCaDk7Vtq2dW2/rMhPBKqQW1pPZmbmUXumraCS7Zm2PdWW/gMgNkV9y7AdO9w8allC+/e7Y716SXffLdWt6/foAADHgM1AAErHgut586TOnaWWLV3bGjvWrZv03nvS4sVS164RCbhthdsC7n79+h11w21f23F73z4HIDZFbZcAywqySuRWkdwqk9s8dcEF0tKl0nPPEXADQAwg6AZQ8iq6L70knX++dOGFVpVIqlBBuvJKaeVKafZsqU2biA7JUspthbsw9r59DkBsdwmYMWNGvi3DZs6c6b0fmK0m9pDS5tLGjV1xyR9/lBo2lF55RVq40D3MBADEBNLLARTPgQOuZc2ECVJWljtmK0bXXCMNH+4KpfnE9nAXllJq7H37HEXMgNhlW0h69eql7t27B7tLgGUDWUXynGJvp5wijRkj9e/vHmICAGIKMzuAwu3c6aqQ2z7Db75xx6pWdRV07WUtwAJUtbgwUVO1GEBsdglYvdrVupg1y31tnRws+LZMnZNO8nt0AIAwIegGkL/vvnP9ta3PthX4MaedJt10kzRgQKBuEGOmajGA2PT999Lo0dLUqW6LjvXXtnnUjtWq5ffoAABhRtANILf1663akPTPf7qUcmP7DK3t1//8j1SxooIm6qsWA4hNe/dKkya5quS7d7tjl1zivm7UyO/RAQAihEJqAJxPPpF695bq1ZMyMlzA3aqVS4P8v/+TrroqkAF3VFctBhCbbDXbHlzafGo9ty3g/u1vpXfekebMIeAGgDhD0A3EM6ueu2iRdPHFtlwsPf+8lJ0t/eEPrnrukiVS9+4uFTLAoq5qMYDYnVPfeENq3twVRdu8WapTx7X+WrZM+v3v/R4hAMAHpJcD8cgCa2vtZSmOOdVzLbC+/HLpllvcDWOUiZqqxQBi08cfu6Jo1kYxp+DkX/8qDRokJSb6PToAgI8IuoF4cvCg9OyzLtj+4gt3zG4G+/VzN4t16yqaBbpqMYDYZN0TLLh++mm30m3bcKzv9m23ucAbABD3CLqBeGD7CadNcwV9Nm1yx5KS3AqM3Rxaj1gAQPFt3y6NG+e6POQUnbRik3ff7VLKAQD4FUE3EMt+/FGaMsW9tm1zx6w9zbBh0nXXSZUrl/iv3L9/v/bt2+elcFt/bGvXRQo3gLjKGLJWinfdJW3d6o516CDde68rlgYAQB4E3UAs+uoraeJEafp0ad8+d8yq6Np+7SuuKPX+wh07digzM1MjRozI1Z7LipdZdXArVmZ7qwEg5ljq+IsvSqNGudaKxqqQT5gg/fGPUkKC3yMEAAQUQTcQS1atcjeAtm/bWtYYW3m59Vbpz3+Wypcv9V9tK9wWcPez/d95WACec9yKmbHiDSCmLF4sjRjhKpDnZAzdeadrpViBWykAQOGC3QcIQPH8+9/Sn/4kNW0qPfWUC7g7d5bmzZOWL5cuu+yYAm5jKeW2wl0Ye98+BwAxwQpOWtvECy5wAfeJJ7pgOytLGjCAgBsAUCwE3UA0pzq+9prUrp3Utq00Z45Lb7QA2wLtuXOlTp3KLOXR9nAfmVKeH3vfPgcAUe3bb6WBA6UmTVx7RXtoef310rp10h13uOAbAIBYCbrr1KmjhISEo16DrOoyEI9++sm1pmnWTLrkEum991yLGlt1sVUZ23PYokWZf1srmlYc1h8bAKLSnj1uJfuss6RHH3VZQ5ZFZFt3MjLo9AAAKJXA50UtX75cv+TsTfW2rK7ShRdeqB49evg6LiDi9u6V/vlP6b77XKE0c9JJbjXmxhullJSwfnurUl4cp59+eljHAQBl7uefpRkzpL/9za1ym5YtXUVySy0HACCWg+7q1avn+vqee+7RmWeeqfbt2/s2JiCirCWNtaexXrDWAszYv4uhQ6UbbpCqVo3IMKwtmFUpLyzF3N63zwFA1GzTef1119nhs8/csTPOcP23e/akIjkAID6C7iMdPHhQTz/9tIYPH+6lmOfnwIED3ivHzp07IzhCoAxt2iTdf7/02GMu5THnZvDmm13F3BNOiOhwrA+3tQXLr3p5DnvfPgcAfrFOC1bQ0epL2LYYy9Kxh4E2N+XqrPDRR24+fecd93W1am6/tmUPlbKtIgAAUR90z5o1S9u3b9dVFnAUYNy4cRozZkxExwWUKduXbW2/bN+27d82tn975Ei38uJTtVy7WbU+3CZvn+7k5GQv4E5LS6NdGADf7Nixw2ttmHeOsiwcm6NsDkvatk26/XbXWtFYgD1kiHTbbVKVKv4NHgAQsxJCIcutig5dunRRxYoVNceqNBcgv5Vu22NqF+LKlStHaKRAKSxdKo0fb0+XXMqjsW0U1mO7S5fApDkeuYpkRdPs31e+q0gASs2uXUlJSVF17bLCp1/l1Js4wg033KCHH3447D+3zU3PP/98gdk4Fk4v7tJFjd95RwkHD7qDV1wh3XWXlJpaou8FAEBJrltRs9JtF/J58+bp5ZdfLvRziYmJ3guIChZcv/22C7ZzUhyN9YW1le3/+i8FjQXW9urYsaPfQwEQIH4XPrWHgbbCnVdFSYMl3W4Z5G+95Q5aO0UrkkYNCgBABERN0D1jxgzVqFFDXbt29XsoQNlUyn3pJRdsf/KJO2Zp4336uD2GjRr5PUIAiKrCp5Z9k5NSXq5cOf2uZUt1379ffT7/XCm/ZsCtlPTL2LFqbhlEAckeAgDEvqgIurOzs72gu2/fvqrg035WoEzs3y/NnOlWWNavd8dOPFG69lpp2DDrt1WyQkAAEKWFT8u6+KnNleXLl9ewYcN0/SWXqGr//qq2bp333k81aijz3HOV/vbbmp6SouYE3ACACIqKCNbSyu1ievXVV/s9FKB0duyQMjKkyZOl775zx04+2RXvGTTI/XdJCgElJfnxUwBAmRU+Levip/Zw8oUXXtC3336rNpdeqoVbt3o3OeMlPfnzz7q1Wzc9N2CAqlmVcgAAIiiqCqnFSzEaxJBvvnH9tS3gzlnBqV1buukmqX9/t8pdgkJAxrI+evXqxYo3EMOi/dpVnMKnZV38dNu2bXruuec0yB5kSmou6WtJPxzxGSvo1rt3b1WtWrWEPxEAAKW/XhN0A+Gwdq1LIbdU8pwqubZP2/YR9uolHXdcgTeN9erVy7XCnZeteGdlZXHTCMSwaL52WeHTunXreoVPu1tRyAj93Fu3blX9+vWLnD/XrFnDajcAoEwU97pVrmy+HQDPRx+5XtoNGkhTp7qAu3VrafZsaeVK156mgIA7byGggtj79jkACCK/Cp9+8sknxZo/7XMAAERSVOzpBgLNkkUWLHCVyOfOPXzcbjhtZbtt22L/VVa7oDisPzYABI2fhU+ZPwEAQUXQDZSW9aOdNcv64kgffuiOlS/v0sdvuUVq1qxUhYCKw/Y8AkDQ+Fn4lPkTABBUBN1ASVnRn6eecnu216xxx6yo2YABrkBanTql/qutLZjtOSxqT6J9DgCC5qKLLpJfpWKYPwEAQcWebqC4rPr4ffdJZ5whXXONC7irVJH++lerHCRNmXJMAbexPtzWFqww9r59DgBwGPMnACCoWOkGivL999KDD1qvGWn7dnfs1FOl4cNd8F2pUpl9K2sDZn24Td4+3cnJyd4NY1paGu3CACAP5k8AQFDRMgwoyIYN0sSJ0uOPWwNtd8yqko8cKaWnSxUrhu1bW7/uffv2eVXKreiP7UG0lEhboeGGEYh98XrtKoufm/kTABC06xYr3UBeK1a4SuQvvOCKpZmWLV2wnZYmlQv/rgy7MbRXx44dw/69ACCWMH8CAIKGoBswlvCxeLELtl9//fDxiy5ybb9+/3spIcHPEQIAAACIQgTdiFteCuKePdqYkaGaM2eqxrp13vFQuXJK6NHDtf067zy/hwkAKGV6ubUvs1ZipJcDAPxE0I24tOPHH7Vi1ChVnzFDzX5NIbdd288nJqrSmDHqfP313v4MAED0sD11mZmZRxVSs1ZhVkjNCq0xtwMAIo2gG/Flzx79lJGh8uPGqd3Wrd6hHZIekfSApO+sB/ett2rGKaeoV69erIoAQBStcFvA3a9fv6PeswA85zhzOwAg0ujTjfhgKx5jxkipqTru5pt10tat+lbSSEm1Jd1mAfcRH7dVEktPBABEB5uzbe4uDHM7AMAPBN2IbRs3SsOGSbVrS6NHe8H33pQUXSupjqQJVuo/nz9mqyK2HxAAEB1szj4ypTw/zO0AAD+QXo7Y9Nln0oQJ0jPPSD//7I6de67X9uvFPXs0rX//Iv8K6+8KAIgOVjStOJjbAQCRRtCN2LJkiWv79corh49Zr1Zr+9W5s9f26/QFC4r1V51++unhGycAoExZlfLiYG4HAEQa6eWIjR7bb7whtW8vtW7tAm7rqf3nP0tLl0rz50sXXnioz7a1jrFKtoWx9+1zAIDowNwOAAgqgm5EL0sbf/ZZqXlz6Y9/lN59VzruOOnqq116+csvSy1bHvXHrFertY4pjL1vnwMARAfmdgBAUBF0I/pY5dlHHpHq15fS06UVK6STTpJuuknasEF6/HGpYcMC/7i1irFerTNmzDhqVSQ5OVkzZ8703qelDABED+Z2AEBQJYRClpsbu3bu3KmkpCTt2LFDlStX9ns4OBbbt7tge/Jk6Ycf3LHkZGnoUOmGG6Rq1Urc09Vax1glWyusY/v8LO3QVkG4KQPgp3i9dpXFz83cDgAI2nWLQmoIvs2bpfvvlx57TNq1yx2rU8carkr9+km/+U2p/lq7+bJXRyu0BgCICcztAICgIehGWB254mDtXKy6bLFXHNaske69V3rySengQXesaVNXibxnT6kCv74AAAAAgo2oBWFjaRaZmZkaMWKEtmzZcui47bWzYja2t87SMY6yfLlr+2WF0HJ2P7Rr54Ltiy8+VIUcAAAAAIKOoBthW+G2gLufpX/nYQF4zvFevXq5FW8LrufNc8G2tfjK0a2bNHKk1KZNJIcPAAAAAPFRvfzrr79Wnz59vNVRS0lu2rSpPvzwQ7+HhSJYSrmtcBfG3t+3e7f04otSixbSRRe5gNvSxq+8Ulq1Spo9m4AbAAAAQNQK9Er3tm3b1KZNG3Xo0EFvvPGGqlevrqysLFWtWtXvoaEItof7yJTyvCpK+vOWLUq0Httff+0OWkG0AQNc66/atSM3WAAAAACIx6B7/PjxXqsP67mZ44wzzvB1TCgeK5qWn0qSrpd0o6QUO2ABt7X6+stfpMGDXQswAAAAAIgRgU4vnz17tlq0aKEePXqoRo0aXtXradOm+T0sFINVKT/SKZLGWjAuacKvAfdGSVkDB0pffSWNHk3ADQAAACDmBDroXr9+vTIyMlSvXj299dZbGjhwoIYMGaInnniiwD9z4MABr0n5kS9Enj0gsX34dSU9IulLSaMkVZH0maS+klpWq6bku++WTjrJ7+ECAAAAQPyll2dnZ3sr3WPHjj0UyK1atUqPPvqo+va1sO1o48aN05gxYyI8UuT1mzVrtOyss5S6ZYvK/3psiaR7JM2RZI3AZkyc6BXHAwAAAIBYFeiV7lq1aqlRo0a5jp199tkF7hc2o0aN8vpD57w2brQkZkSEtf1auFD6wx+U+Lvfqe7SpV7APfe443SBpNa2ZcD6dCcna+bMmV6fbq9dGAAAAADEqECvdFvl8tWrV+c6tmbNGqWmphb4ZxITE70XIig727X2uuceaelSd6xcOenyy3Vg6FC1qF9foz/+2HsAYoXxLGPBVrgJuAEAAADEukAH3cOGDVPr1q299PKePXtq2bJlmjp1qvdCABw8KD3zjDRhgvTFF+6YBdL9+lkTbqluXdnjD3t17NjR79ECAMLs66+/1siRI702n3v37tVZZ53ldSCxrWIAAMSrQAfd559/vjIzM72U8TvvvNNrFzZ58mSlp6f7PbT4tmuXZFXkJ0063GM7KUkaNEgaMkQ6xWqVAwDiybZt27wMtQ4dOnhBd/Xq1ZWVlaWqVav6PTQAAHwV6KDbXHLJJd4LBdu/f7/27dunjz/+2Nvvbu26wpLC/cMP0pQp0kMP2d2VO1azpjR8uHTddVLlymX3vQAAUWX8+PHeFiJb2c5hD8sBAIh3gQ+6UTgrFmfZACNGjNCWLVsOHbd2Xffdd59XrCzJVqGPhfXRnjhRmj5d2rfPHatXT7rlFumKK2wj/TH+FACAaDd79mx16dJFPXr00KJFi3Tqqafqhhtu0DXXXFNom0975aDNJwAgFgW6ejmKXuG2gLtfv365Am5jX9txe98+VyorV7qg+swz3Qq3Bdy//a300kvS559LAwYQcAMAPOvXr1dGRobq1aunt956SwMHDtSQIUP0xBNPFPhnrM2nPRjOedlKOQAAsSYhFLI+T7HLnprbhdxWhCvHWPqz7Z+zm5u8AfeRbMW7xHvq3nvP8gSlV189fKxzZ+nWW60impSQcIwjBwDE2rWrYsWKXsG0999//9AxC7qXL1+uJUuWFHul2wLvaPq5AQDxa2cxr9esdEcx28NdWMBt7H37XJHs2ctrr0lt20rt2rmA24Lryy6Tli+X5s6VOnUi4AYA5KtWrVpq1KhRrmNnn322V2ukINbi025SjnwBABBr2NMdxQq7kTmS9ccu0E8/SS+84Fa2V61yxypWlPr2dW2/6tcvo9ECAGKZVS5fvXp1rmNr1qxRamqqb2MCACAICLqjmFUpL45898jt3Ss9/rgrkGaF0kylSq4K+bBhUkpKGY8WABDLhg0bptatW2vs2LHq2bOnli1bpqlTp3ovAADiGUF3FLO2YLZnu6g93fa5Q7ZulR5+WHrwQenHH92xGjWkG2+UBg6UqlSJwMgBALHm/PPP94p3jho1SnfeeafXLmzy5MlKT0+PzTaaAAAUE0F3FLMbCGsLZlXKC2Lv2+e0aZM0aZJkKw579rg3rX+qpZDbn7fPAABwDC655BLvFdNtNAEAKCGC7ihmT+ztBsLkvcFITk72bjAuPftsHX/DDdLTT7v92+acc6SRI6UePaQK/AoAAGKrjWZeOW00Ta9evVjxBgBEFC3DYsCRqXRWNM32cLf45Red+NBDKj9njqtMbtq3d8H2H/5AFXIACLB4uHaV9c8dtjaaAAAc43WLZc4YYE/s7dWxQwfp7belu+6SFi48/IG0NBds/+53fg4TAIBAtNHs2LFjxMYFAABBdywUZvn5Z+mll1zbr08+cccsbbxPH+nmm6U8fVMBAIg1ZdJGEwCAMCDojubCLPv3SzNnSvfeK61f746deKJ0zTXS8OHWKyzyYwIAwOc2muXKlVOrVq1UrVo1bd26VUuXLlV2dnbBbTQBAAgjgu5oLMyyfbuUkSE98ID03Xfu2MknS0OHSoMGSdWqRWYcAAAEhGWf1ahRQ1deeaUGDBigFStWaPPmzUpJSVGzZs00ffp0PfHEE7nbaAIAEAEE3UWwlHJb4S6Mvd+9e/fwB93ffCNNnuwC7l273DF7sn/TTVL//m6VGwCAOGTbvRYuXKgFCxaoTZs2R2WmjRkzRosWLXJtNAEAiKBykfxmsV6YJWyysqRrr5Xq1JEmTHABd+PG0pNPSmvXSkOGEHADAOKepZEPHjz4qOu2fW3H7X0AACKNoDvIhVk++kjq2VNq0ECaNk06eFBq3VqaPVtasUK64grpuOPK/vsCABCjmWn2OQAAIomguwSFWQpTZoVZrKf2/PnShRdKLVpIL77ojnXtKi1eLP3731K3blYlpmy+HwAAMSAQmWkAAOSDPd1FsIIrthessAu5vX/MhVl++UV65RXpnnuk5cvdsfLlpd69pVtukZo2Pba/HwCAGEbLMABAULFcWgQruGJtwQpj75e6MMuBA9Ljj7te2v/93y7gtoJsgwe7/dpPPUXADQBA0DLTAAAoJla6i2AVya0Pt8nbpzs5OdkLuNPS0kpeuXznTmnqVOn++6XNm92xKlVcyy8rjFajRpn+HAAAxLKIZaYBAFBCBN3FkJSU5PXhtrZgthfMUtPsSblduG2Fu0QB9/ffu/7ajzzi+m2bU0+Vhg+XrrlGqlQpbD8HAACxnpnWr1+/8GSmAQBQSgTdxWSBtb06duxYur9gwwa72kv//Ke0f787ZlXJbb92erqUmFim4wUAIJ6ELTMNAIBjRNAdbtbaa/x46YUXXLE007KldOutUvfuVCEHACCImWkAAJQRgu5wsBZf1t7Lgu3XXz98vEsXF2y3by8lJPg5QgAAYtIxZ6YBAFDGCLrLUna2NGeOC7aXLHHHbCX7sstcsE3xFgAAAACIK4HPbR49erQSEhJyvRo2bKhAOXhQeuIJqUkTKS3NBdy2R/u666TVq11qOQE3AABht3//fm3btk0LFizQzJkzvf+1r+04AAB+iIqV7saNG2vevHmHvq5QISDD3rNHmj5dmjhR2rjRHatcWRo4ULrxRqlmTb9HCABA3NixY4cyMzOPKqRmrcKskJoVWrN93wAARFJAotfCWZBdM0gB7I8/Sg89JE2ZIm3d6o7Z+IYNc6vbXNABAIgoW8m2gDu/lmEWgOcct0JrFFQDAERS4NPLTVZWllJSUlS3bl2lp6frP//5jz8Dse9rK9ipqdKYMS7gPvNM6bHHXEswa/9FwA0AQMTt27fPW+EujL1vnwMAIJICH3S3atXK25P15ptvKiMjQxs2bFC7du20a9eufD9/4MAB7dy5M9erzNx7r/TAA9LevW6Ptu3Vtj3b115r5VLL7vsAAIASsRZhR6aU58fet88BABBJgU8vv/jiiw/9d7NmzbwgPDU1Vf/617/Uv3//oz4/btw4jbFV6HAYPlz64gu3ot25M22/AAAIiOJmwVnvbgAAIinwK915ValSRfXr19fatWvzfX/UqFFeIZWcV5leXM84Q5o7V7rwQgJuAAACpHbt2sX63Omnnx72sQAAENVB9+7du7Vu3TrVqlUr3/cTExNVuXLlXC8AABDbzj33XK9KeWHsffscAACRFPig24qeLFq0SF9++aXef/99r91H+fLl1bt3b7+HBgAAAuKEE07w2oIVxt63zwEAEEmB39O9adMmL8C24ifVq1dX27Zt9cEHH3j/DQAAYKwNmD2YN3n7dCcnJ3sBd1paGu3CAAARlxAKhUKKYVa9PCkpydvfTao5ACAaROO1a/To0UcVMm3QoIG+sAKkEfy5rV+3tQWzKuVW18X2cFtKua1wE3ADAMpSca9bgV/pBgAA0aFx48aaN2/eoa8rVIj8bYYF1vbq2LFjxL83AAD5IegGAABlwoLsmjVr+j0MAAACJfCF1AAAQHTIyspSSkqK6tatq/T09CJ7Zx84cMBLzTvyBQBArCHoBgAAx6xVq1aaOXOm3nzzTWVkZGjDhg1q166ddu3aVeCfGTdunLcXLudFD20AQCyikBoAAAETC9eu7du3KzU1VZMmTVL//v0LXOm215E/twXe0fxzAwDix04KqQEAAL9UqVJF9evX19q1awv8TGJiovcCACCWkV4OAADK3O7du7Vu3TrVqlXL76EAAOCrmF/pzsmepzgLACBa5FyzomkH2IgRI9StWzcvpXzz5s36+9//rvLly6t3797F/ju4ZgMAYvF6HfNBd04BF4qzAACi8Rpme8WiwaZNm7wAe8uWLapevbratm2rDz74wPvv4uKaDQCIxet1zBdSy87O9p64V6pUSQkJCcf89+UUedm4cSNFXkqA81Y6nLfS4byVHucuGOfNLs12Abf2W+XKxc9OsLK8ZvO7XDDOTcE4N4Xj/BSMcxOf5yZUzOt1zK902w9/2mmnlfnfa78wsfZLEwmct9LhvJUO5630OHf+n7doWeEO+jWb3+WCcW4KxrkpHOenYJyb+Ds3ScW4XsfP43MAAAAAACKMoBsAAAAAgDAh6C4h6ydqFVnpK1oynLfS4byVDuet9Dh3pcN5Cx7+f1Iwzk3BODeF4/wUjHNTsETOTewXUgMAAAAAwC+sdAMAAAAAECYE3QAAAAAAhAlBNwAAAAAAYULQXQr33HOPEhISdOONN/o9lMD7+uuv1adPH5188sk64YQT1LRpU3344Yd+DyvwfvnlF91xxx0644wzvPN25pln6q677hIlGHJ799131a1bN6WkpHj/JmfNmpXrfTtff/vb31SrVi3vPHbu3FlZWVmKd4Wdt59++kkjR470/q2eeOKJ3meuvPJKbd682dcxR8vv3JGuv/567zOTJ0+O6BjjHfPnYcyRBWMeLBjz3LGdm88//1x/+tOfvN7N9vtz/vnn6z//+Y/iQVHnZ/fu3Ro8eLBOO+00b85p1KiRHn30UcUDgu4SWr58uR577DE1a9bM76EE3rZt29SmTRsdd9xxeuONN/TZZ59p4sSJqlq1qt9DC7zx48crIyNDDz30kDd529cTJkzQlClT/B5aoOzZs0fnnHOOHn744Xzft3P24IMPehP60qVLvYtfly5dtH//fsWzws7b3r179b//+79e0GL/+/LLL2v16tXeDQSK/p3LkZmZqQ8++MC78UBkMX8exhxZMObBgjHPlf7crFu3Tm3btlXDhg21cOFCrVixwvs9Ov744xUPijo/w4cP15tvvqmnn37am59tAdOC8NmzZyvmWfVyFM+uXbtC9erVC82dOzfUvn370NChQ/0eUqCNHDky1LZtW7+HEZW6du0auvrqq3Mdu/TSS0Pp6em+jSnobDrLzMw89HV2dnaoZs2aoXvvvffQse3bt4cSExNDzz33nE+jDP55y8+yZcu8z3311VcRG1c0n7tNmzaFTj311NCqVatCqampofvvv9+X8cUr5s/8MUcWjHmwYMxzJTs3l19+eahPnz6+jSno56dx48ahO++8M9ex8847L3T77beHYh0r3SUwaNAgde3a1Uu/QtHsqVWLFi3Uo0cP1ahRQ+eee66mTZvm97CiQuvWrTV//nytWbPG+/rTTz/Ve++9p4svvtjvoUWNDRs26Ntvv83179VSvVq1aqUlS5b4OrZos2PHDi9NrEqVKn4PJfCys7N1xRVX6Oabb1bjxo39Hk5cYv4sHubIkmEePIx5ruDz8tprr6l+/fpexojd+9q/p8LS8+Nxfp49e7a3/dTi8nfeecebqy+66CLFugp+DyBaPP/8816KkaWXo3jWr1/vpfhZKsltt93mnbshQ4aoYsWK6tu3r9/DC7Rbb71VO3fu9NKT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",
      "text/plain": [
       "<Figure size 1000x800 with 4 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "fig, axes = plt.subplots(2, 2, figsize=(10, 8))\n",
    "datasets = df[\"dataset\"].unique()\n",
    "\n",
    "for i, dataset in enumerate(datasets):\n",
    "    subset = df[df[\"dataset\"] == dataset]\n",
    "    sns.scatterplot(x=\"x\", y=\"y\", data=subset, ax=axes[i//2, i%2], s=60, color=\"black\")\n",
    "    X = sm.add_constant(subset[\"x\"])\n",
    "    model = sm.OLS(subset[\"y\"], X).fit()\n",
    "    slope, intercept = model.params[\"x\"], model.params[\"const\"]\n",
    "    x_vals = np.linspace(subset[\"x\"].min(), subset[\"x\"].max(), 100)\n",
    "    y_vals = intercept + slope * x_vals\n",
    "    axes[i//2, i%2].plot(x_vals, y_vals, color=\"red\")\n",
    "    axes[i//2, i%2].set_title(f\"Dataset {dataset}\")\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a8bcd66a-1eb1-4195-a279-66669fa9a3ad",
   "metadata": {},
   "source": [
    "## Scatterplot\n",
    "The graphs above visualize the different datasets that some outliers. A regression line (or a line of best fit) has been drawn so the overall points can be visualized into a pattern. Some datasets, specifically dataset III and IV include outliers that seem very far from the predicted pattern of the regression line. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "80c0ac5f-e2b9-43f6-a45c-fc360ce27c8c",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAq4AAAIjCAYAAADC0ZkAAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjYsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvq6yFwwAAAAlwSFlzAAAPYQAAD2EBqD+naQAAcUBJREFUeJzt3QeYU1X6x/FfkumV3qt0RMGuiKKCYsMuiqgg/tVVd61rX1x7W3tZ26rYUVFRUXEVRVBURMUG0kR6L9Nrkv/z3tmMM8OUwMykTL6f5wnD3JxJTm5ukjfnvuc9Lr/f7xcAAAAQ4dzh7gAAAAAQDAJXAAAARAUCVwAAAEQFAlcAAABEBQJXAAAARAUCVwAAAEQFAlcAAABEBQJXAAAARAUCVwAAAEQFAlcghkycOFEul0tz584Nd1cQQf744w/nuLDjI+Cmm25ytiF21ecYOOSQQ5xLXWbMmOHch/0EgkHgipjz73//23mj3G+//cLdlSbD9mcwH1LB2rx5s6666ir16dNHSUlJatGihUaMGKH3339f4fbKK6/owQcfrNdt3HHHHZoyZYpC5eqrr3aeo9NOO63Bb3vcuHHObQcuaWlp2mWXXXTKKafozTfflM/nC+u+bij5+flOINcYAdaXX36pE088UW3btlViYqK6deumCy64QCtWrGjw+wKiHYErYs7LL7/sfDDMmTNHS5YsCXd3ol5WVpbz0/ZpQ1i4cKEGDhyohx9+WIceeqgeffRRXX/99dqwYYOOPfZYXXvttQqnaAtc/X6/Xn31Vef5ee+995STk9Pg92HB1osvvuhcHnjgAZ1xxhlavHixE7wOGzZM2dnZTSJwvfnmmxs8cH3kkUd00EEH6eeff9bf/vY354u17bfXXntNu+++u2bPnq1o9N///te5AA0trsFvEYhgy5Ytcz4I3nrrLWdEw4LYf/7zn+HuVlSbOXOmPB6PrrvuunrfVklJifOhvXXrVud2K46KX3755RozZozuvvtu7bXXXjr11FMVSnl5eUpNTVW0sUBr1apV+vTTT51Razv2x44d26D3ERcXpzPPPLPStttuu0133XWXc1ycd955TiCG7UdaL7vsMg0ZMkTTpk1TSkpK+XUXXnihDjzwQOf18Ouvv6p58+ZRdawnJCQ0WH+AihhxRUyxQNU+AI455hjnA8F+rynf795779VTTz2lHj16OCNK++yzj7799ttKbdetW6dzzjlHnTp1ctq0b99exx9/vHMbFX344YcaOnSo0tPTlZGR4dyWjSZV9MYbbzgBWXJyslq1auUEAqtXr97utKydirVTiDb6aP/v2LGjHnvsMed6G7U57LDDnA+drl27bncfFUePLHBv2bKl05+zzz7bCRarsn7baJDdnvXd9pt9iFb0+eef6/TTT3dO61cdSdp1112dD2Pb53vvvXeN/QmwU8u//PKLM6paNZXDguMnn3xSzZo1q/RlI5C3W3WfV5c7N2vWLCfg7dKli/N8de7c2QmICwoKqt3PS5cu1dFHH+08dguaLR3C0hWWL19efmq84khzUVGR07eePXuW376dprftAfY3Fhg8//zz5bdh99dY7Bjv37+/M3o9fPjwao/5xmLP4xFHHOEc24sWLSrf/s477zjHUocOHZz9ZK+xW2+9VV6vt7xNbfu6uLhYN954o/N6yczMdI5PO04/++yz7fowadIkp13gtbfbbrvpoYceqtRm27ZtTgBpz5f1x54/+4IUSHOwY6t169bO/23UNdAfSx3YkfeBquwx2+3YsVAxaDW2T+655x6tXbvWOe6NvSdZe9snVdkXBAsWK76Ov/nmGx155JHOPrLbt/cgC5ary2OdP3++M1Jur1ULpGvy3HPPOe8xbdq0cR6rHVuPP/54UDmu9gXqhBNOcJ4v+3t77VV8bQDBYMQVMcU+tE866STnDX706NHOG64FoxZIVmVBlp1WtQDP3tjtQ8T+9vfff1d8fLzT5uSTT3YCOTvFZx+qdjr7448/dgLLwIesBVbjx493gjj7cLHA64cffnBGWOyDItDGPvisH3feeafWr1/vfLjah4y1tb8JsA/3o446SgcffLDTJ3tMf/3rX50PgxtuuMEJsKyfTzzxhBOQHnDAAerevXulx2bt7TbtQ8tOzdt+sA/DQLBn7LSvjczZKJ19iFuwa+3sQ836FHh89mFa1dNPP61LLrnE+XJw6aWXqrCwUD/99JPzQRp4zNWxU9nG+l0d+wC2gMA+6C2otA/3HWEBlD0OG82yoN3SRSzAtg9Uu66i0tJS57Hb47XHaB/87dq1c1IjrL2dEjcW4BoLco477jh98cUXOv/889WvXz/ni4S1s6AtkBpg+/X//u//tO+++zrtzI4+jmBZUGBfBq688krndzvm7Tiz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q/K/HTFihBP0zpo1S19++aXS0tKcEWKrx2r3YUGt/R64X+sLAESavNI8jflgjBZtXbTddSW+Ev37x387wSzBKwDOc++kvn376qeffnL+f9lll5Vv79atm2677Tb95S9/0b///W9nNa7MzExnxNNGVSsaP358+f932WUXJ/jdZ599nNFaC0JXrFihPfbYwxmVDdx2wGuvvebk2f7nP/8pX+3LRmNt9NVGVo844ghnlLaoqGi7+wWASFHkLdLL81/W+vz1tbZ7+qenNabfmJD1C0BkYsS1HsulBgLGTz75RMOGDXPSCWwE9KyzztLmzZuVn59f62189913GjlypLp06eL83dChQ53tFrCaCy+8UJMmTdKgQYOcEdzZs2eX/62N0i5ZssT5Owty7WLpAoWFhVq6dGmjPnYAaMg0gdcWvlZnu2Jfsd7//X3nvRdA7CJw3Ul26r579+7O6f1jjz1Wu+++u958800nGH3sscecNnbKviZ5eXnOqX5LIXj55Zf17bff6u233670d0cddZSzFOvll1+uNWvWOMFxIM3ARmX32msvzZs3r9Jl0aJFTq4sAEQDt8utzYWbg2q7ZNsSJ3UAQOwiVWAnWA7rzz//7ASUFqjaKfv77ruvfOnU119/vVJ7SxewqgQV/fbbb86o7F133eVM6DJz587d7r5sYtbYsWOdy0EHHaSrrrpK9957rzNRzNIF2rRp4wS/1anufgEgkrjkcoLXYEpfJcUlyePyhKRfACITI651sBzRdevWafXq1fr+++91xx13OLP3bZT17LPPVs+ePZ2JUo888oh+//13p1LAE088Uek2LDfVRkinT5+uTZs2OSkElh5ggWXg76y0lk3UqujGG290qgJYSoBN/Jo6dar69evnXGfVBmyilvXFJmctW7bMyW295JJLtGrVqvL7tTzchQsXOvdr/QSASGIpAEM6Dgmq7fE9jpfHTeAKxDIC1zpY3db27ds7QaDN0P/ss8+cSVQWUHo8Hqe6gJXDuvvuuzVgwADntP+dd95Z6TZsNr9N1jrttNOcEdR77rnH+WnltN544w3179/fGXm1kdSKLLC97rrrnDSEgw8+2Lk/y3k1KSkpmjlzphMAn3TSSU5Ae+655zo5roER2PPOO099+vRxJnfZ/VnlAQCIJOkJ6frL7n+ps13/lv3VPrV9SPoEIHK5/E080z07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7rrrLmeVrssuuyzcXYlYq1ev1plnnqmWLVsqOTlZu+22m+bOnRvubkUcr9erCRMmqHv37s5+6tGjh2699dY613KOBTNnztTIkSOdVVbs9TZlypRK19s+uvHGG9W+fXtn3w0fPlyLFy9WLKptX5WUlOiaa65xXoOpqalOm7PPPttZcTBW1XVsVfSXv/zFafPggw8qFgWzrxYsWKDjjjvOKS5vx9g+++yjFStWKNbUta9yc3P117/+1Vk0yd6z+vfvryeeeCJs/Y1UBK4N7Ntvv9WTTz6p3XffPdxdiVhbt27VgQceqPj4eH344YeaP3++7rvvPjVv3jzcXYs4d999tx5//HE9+uijzpu//X7PPffokUceUazLy8vTwIED9dhjj1V7ve2nhx9+2Hnj/+abb5wPzBEjRqiwsFCxprZ9lZ+fr++//975gmQ/33rrLS1cuNAJNGJVXcdWwNtvv62vv/7aCURiVV37aunSpRoyZIj69u2rGTNm6KeffnKOtaSkJMWauvbVFVdcoWnTpumll15y3u9t8MsC2XfffTfkfY1oVg4LDSMnJ8ffq1cv/8cff+wfOnSo/9JLLw13lyLSNddc4x8yZEi4uxEVjjnmGP/48eMrbTvppJP8Y8aMCVufIpG9lb399tvlv/t8Pn+7du38//rXv8q3bdu2zZ+YmOh/9dVX/bGs6r6qzpw5c5x2y5cv98e6mvbXqlWr/B07dvT/8ssv/q5du/ofeOABf6yrbl+ddtpp/jPPPDNsfYqmfbXrrrv6b7nllkrb9txzT/8NN9wQ4t5FNkZcG9DFF1+sY445xjkliZrZt8e9995bp556qtq0aaM99thDTz/9dLi7FZEGDx6s6dOna9GiRc7vP/74o7744gsdddRR4e5aRFu2bJnWrVtX6bVopyn3228/ffXVV2HtWzTIyspyTmU2a9Ys3F2JSD6fT2eddZauuuoq7brrruHuTkTvp/fff1+9e/d2znbY+729BmtLvYj193v7fLRUOottP/vsM+e9/4gjjgh31yIKgWsDmTRpknOa7c477wx3VyLe77//7pz+7tWrlz766CNdeOGFuuSSS/T888+Hu2sR59prr9Xpp5/unGaz1AoL8u300ZgxY8LdtYhmQatp27Ztpe32e+A6VM9SKSzndfTo0crIyAh3dyKSpezExcU571uo2YYNG5y8TZv3ceSRR+q///2vTjzxRJ100kn6/PPPw929iGMpYJbXajmuCQkJzj6ztIKDDz443F2LKHHh7kBTsHLlSl166aX6+OOPYzJvZ2e+hduI6x133OH8bsHYL7/84uQijh07Ntzdiyivv/66Xn75Zb3yyivOyM68efOcwNVy6thXaGg2UWvUqFHOaI99ucT2vvvuOz300EPOQIWNSqP293pz/PHH6/LLL3f+P2jQIM2ePdt5vx86dGiYexh5gavlTNuoa9euXZ3JXHYm197vOZP7J0ZcG+iNzL5Z7rnnns63cLvYt0mbGGL/t5nh+JPN8rZvlRX169cvJmeZ1sVORQZGXW3Wt52etA8ARvZr165dO+fn+vXrK2233wPXofqgdfny5c6XcEZbqzdr1izn/b5Lly7l7/e2z6688kp169Yt3N2LKK1atXL2D+/3dSsoKND111+v+++/36k8YBO8bWLWaaedpnvvvTfc3YsojLg2gGHDhunnn3+utO2cc85xTu/aKTePxxO2vkUiqyhgs5Yrsjwe+4aJ7Wd8u92Vv1/a8RQYyUD1rHyYBaiWH2wjPCY7O9upLmCpKag+aLVyYZZXZ2XqUD378lh19MvyN227ve/jT3a620pf8X4f3GvQLrzf143AtQGkp6drwIABlbZZ6R1786+6HXJGDC0J3VIF7MNyzpw5euqpp5wLKrNv3rfffrszumOpAj/88IPzjXz8+PGKdZY7t2TJkkoTsiyVokWLFs7+spSK2267zcmltkDWSvDYKbcTTjhBsaa2fWVnQE455RTn1PfUqVOdM0SBPGC73oKPWFPXsVU1sLf8c/ui1KdPH8WauvaVnTWyUUPL0zz00EOdck/vvfeeUxor1tS1ryx1wvaX1XC1wN7O3L7wwgvOez4qCHdZg6aKcli1e++99/wDBgxwyhP17dvX/9RTT4W7SxEpOzvbOY66dOniT0pK8u+yyy5OaZSioiJ/rPvss8+ckjJVL2PHji0viTVhwgR/27ZtneNs2LBh/oULF/pjUW37atmyZdVeZxf7u1hU17FVVSyXwwpmXz3zzDP+nj17Ou9hAwcO9E+ZMsUfi+raV2vXrvWPGzfO36FDB2df9enTx3/fffc572X4k8v+qRjIAgAAAJGIyVkAAACICgSuAAAAiAoErgAAAIgKBK4AAACICgSuAAAAiAoErgAAAIgKBK4AAACICgSuAAAAiAoErgAQxVwul6ZMmRLubgBASBC4AkA9eL1eDR48WCeddFKl7VlZWercubNuuOGGsPUNAJoaAlcAqAePx6OJEydq2rRpevnll8u3/+1vf1OLFi30z3/+M6z9A4CmhMAVAOqpd+/euuuuu5xgde3atXrnnXc0adIkvfDCC0pISKj2b66//nrtt99+220fOHCgbrnlFuf/3377rQ4//HC1atVKmZmZGjp0qL7//vsa+zFjxgwndWDbtm3l2+bNm+ds++OPP8q3ffHFFzrooIOUnJzsjApfcsklysvLK7/+3//+t3r16qWkpCS1bdtWp5xyyk7vGwBoSASuANAALGi1oPOss87S+eefrxtvvNH5vSZjxozRnDlztHTp0vJtv/76q3766SedccYZzu85OTkaO3asE2h+/fXXTjB59NFHO9t3lt3fkUceqZNPPtm5r9dee825/b/+9a/O9XPnznUCWQueFy5c6IwkH3zwwTt9fwDQkFx+v9/foLcIADHqt99+U79+/bTbbrs5I6NxcXG1th80aJATQE6YMKF8FPbTTz91gtTq+Hw+NWvWTK+88oqOPfZYZ5uNpr799ts64YQTnBHXQw89VFu3bnXaBUZc99hjDy1btkzdunXT//3f/znpDU8++WT57VrgaqO5Nur6wQcf6JxzztGqVauUnp7egHsHAOqPEVcAaCDPPvusUlJSnCDRAr+62KirBaHGxhBeffVVZ1vA+vXrdd555zkjrZYqkJGRodzcXK1YsWKn+/jjjz86OblpaWnllxEjRjhBsfXbUhO6du2qXXbZxRk9trzd/Pz8nb4/AGhIBK4A0ABmz56tBx54QFOnTtW+++6rc8891wlGazN69GjndLyNztrfr1y5Uqeddlr59ZYmYCOmDz30kHO9/b9ly5YqLi6u9vbc7rK39Ir3W1JSUqmNBb4XXHCBc1uBiwWzixcvVo8ePZxRVuuPBdHt27cvT3momDcLAOFS+3ksAECdbERy3LhxuvDCC51T9d27d3fSBZ544glnW006derknKK3Uc2CggJntLNNmzbl13/55ZfORCnLazUW2G7atKnG22vdurXz0yaINW/e3Pm/BaYV7bnnnpo/f7569uxZ4+1YisPw4cOdi1VFsLQDS2GoWvILAEKNEVcAqKfrrrvOGeW0ygLGcknvvfdeXX311ZVm81fHUgOsAsEbb7xRKU3AWIrAiy++qAULFuibb75xrrdKADWxYNSqBNx0003OCOr777+v++67r1Kba665xhm9tclYFtRaO6uCEJicZSPGDz/8sHPd8uXLncoIlkbQp0+feuwhAGgYBK4AUA+ff/65HnvsMT333HNOfmuAnY63hQnqShmwUlObN292Rm1tglVFzzzzjDPRykZJLd/UZvtXHJGtKj4+3jnFb5PEdt99d91999267bbbKrWx7dbnRYsWOSWxbOKWpQN06NDBud5GV9966y0ddthhzkQzGzW229x1113rsZcAoGFQVQAAAABRgRFXAAAARAUCVwAAAEQFAlcAAABEBQJXAAAARAUCVwAAAEQFAlcAAABEBQJXAAAARAUCVwAAAEQFAlcAAABEBQJXAAAARAUCVwAAACga/D9AjsMUDCEjNQAAAABJRU5ErkJggg==",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,6))\n",
    "sns.scatterplot(data=df, x=\"x\", y=\"y\", hue=\"dataset\", s=70)\n",
    "plt.title(\"Anscombe’s Quartet – All Datasets Overlaid\")\n",
    "plt.xlabel(\"X values\")\n",
    "plt.ylabel(\"Y values\")\n",
    "plt.legend(title=\"Dataset\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c22671e7-e41d-4c5c-a6f6-9426cf87f175",
   "metadata": {},
   "source": [
    "## Overlaid Compaison Plot\n",
    "This graphs showcases the points of data overlaid across each other with the same X and Y axis'. They are colour coded for a clearer view and for the reader to be able to distinguish where data points are coming from. \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "e0cd04f2-3efc-44a6-ba7d-d0a3ad64229b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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B8o1rfG+/ud//mf62Mc+VqwXmrofkLUSkI0rrrgiukpXuHXt2rU0ZcqUBuvecsstxWNnnXVWg+UHHXRQqUOHDqUZM2bULRs/fnypY8eOpfvuu6904403Fs+74IIL1tiee+65p1j38ssvL/3rX/8qzZkzp/Tb3/62tOWWWxbbePDBB4v1Zs6cWawX7S/bYYcdShtvvHHpjTfeqFv26KOPFu0YPXp03bIf/vCHxXPjNdZk6dKlxWsOGTKk9M4779QtnzZtWvEap5122grvZbmNa/LQQw8V60+fPr24v3z58tJmm21WOuGEExqsV97XPn36lN5888265bfeemux/Pbbby/uz5s3r7gf+7cq//73v0sbbLBBaZdddmmwP+XtN3Wfx4wZUyw755xz6pZFO9Zbb73i87ruuuvqlj/99NPFut/73vdWeM8++tGPFtstO++884rlsY/hrbfeKm244Yalr3zlKw3a/Oqrr5Z69uy5wvLGRo4cWdpuu+1KTRXHVLz+6px//vlFW2+77bYG+9T4+Cof2/GzbNGiRSu83qRJk4r3btasWSu8z6eccsoK6++7776lLbbYYoXlv/rVr4pj//7772+w/OKLLy5e689//nPdsm7duhXbgGqkVAAyFKd2o2cublOnTi1mFYhemKi9LPvd735X9JZGz1V9UToQvbH1ZyGInrPo0YsesmOPPbY4Ld34easTPYLRCxoDseJ0cPlUZv063PpeeeWV9MgjjxSnOGPEd9n222+f9t5776Lt70dM0xS9lrEPMZq8LNoU5RAfZFR59ExGHXG81yF6Fr/0pS+l6667bqXlFPFYnOIui1PmoTzbQvR2R89t9FKu7DR7iM83em3j1Hf9/Slv//3ucxwrZdFzGr3P0eMasyyUxbJ4bGWzQ0RPX7nXOEQPZJSplD+3aHcMiIpe6TgzUL7F8Rin8OMswOrEdqNXsnFpxZrEexWn5Ven/His21TxmZXFMR77FOU08fsUvciNxfuytqLHNnpZ4zOr/56Vy27W9J5BtVAqABmK04T1Q2EEhB133LE4hRu1ixGI4tRtBMnGf8jjj2OIx8ti/csvv7w41Rzh54orrqgLRmvjtNNOK4JZBJM4nR/baFxvW195243rGsvti5KG9zOYZXWvG4EgTsO+HxFMI6BGaK1ftxshbPLkyemuu+5Kn/3sZxs8J+og6yuH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ji1C6x577JEeeOCBtP7666dp06alPffcM7U3EVq79+5X6WYAAFS1Jg/O+vWvf52GDRtW/P/2229PM2fOTE8//XQ66aST0ne/+92WaCMAADQ9uL7++utpk002Kf7/u9/9Lh188MFpm222SUcffXRRMgAAAG0iuNbU1KQnn3yyKBO444470t57710sX7RoUVpnnXVaoo0AAND0GtejjjoqHXLIIalfv36pQ4cOafjw4cXyv/71r2nQoEEt0UYAAGh6cD399NPTkCFD0uzZs4syga5duxbLo7f1lFNOaYk2AgDA+5sO66CDDip+Ll68uG7ZmDFjmq9VAADwQWtco7b1zDPPTJtuumlxlaEXXnihWD5x4sR02WWXNfXlAACgZYLr2WefnaZMmZLOO++81KVLl7rlUT7wy1/+sqkvBwAALRNcr7rqqnTJJZekUaNGNZhFIOZ2jflcAQCgTQTXl19+OW299dYrLF++fHlxeVQAAGgTwXXw4MHp/vvvX+kVtXbcccfmahcAAHywWQVOO+20YgaB6HmNXtabbropPfPMM0UJwbRp05r6cgAA0DI9riNHjky33357uvPOO1O3bt2KIPvUU08Vy8pX0QIAgDYxj+vuu++epk+f3uyNAQDaj3nz5qWFCxemajJ37twGP6tJ9+7dU69evdpecAUAWFNoPeecc6p24PbUqVNTtencuXOaMGFCi4bXJgfXjh07pg4dOqz2AgXA+/P6wqWVbgKtwOdMNYie1gitB22zddp4/fUq3Rxa2GuL3km/fnZG8bm3qeB68803N7gfB+U//vGPdOWVV6YzzjijOdsGVeemx1+rdBMAmlWE1v7du1e6GbQTnd7P4KzGDjrooLTddtul66+/Po0dO7a52gZV58ChG6eNuv/fFelovz2uvqQAVLDG9eMf/3g65phjUnOKsoPTTz+9qBN59dVXU//+/dORRx6ZTj311NWWK0CuIrT279m10s0AgPYbXN955530k5/8JG266aapOZ177rnpoosuKsoQokf3oYceSkcddVTq2bNnOv7445t1WwAAtLPgGgW39Xs7S6VSeuutt9L666/f7CPoHnjggaI0Yd999y3ub7nllunaa69Nf/vb35p1OwAAtMPg+uMf/7hBcI1ZBvr27Zt22WWXZh9Ftttuu6VLLrkkPfvss2mbbbZJjz76aPrTn/6Uzj///FU+Z8mSJcWtbMGCBc3aJgAAMgmuUWPaWk455ZQieA4aNCits846Rc3r2WefnUaNGrXK50yaNMnsBgCrYVL46tIak8JDmwqujz322Fq/4Pbbb5+ayw033JCuvvrqdM011xQ1ro888kg68cQTi0FaY8aMWelzxo8fn775zW/W3Y/gW1tb22xtAsiZSeFNCg/tPrjusMMORXlA1LOuTqzTnBcgOPnkk4te10MPPbS4P3To0DRr1qyiV3VVwbVr167FDYBVTwrff/jWqUtvk8K3d0vffCfNubPlJ4WHNhVcZ86cmSph0aJFRQ1tfVEysHz58oq0B6C9iNC6Xt9ulW4GQPMH1y222CJVwogRI4qa1s0337woFYgrdMXArKOPProi7QEAIMN5XJ988sn04osvpqVLG15ze//990/N5ac//WmaOHFiOvbYY9Nrr71W1LZ+9atfTaeddlqzbQMAgHYaXF944YX0hS98IT3++OMN6l7LU2Q1Z43rBhtskC644ILiBgBAdWtYQLoWTjjhhDRgwICiBzQuOvDEE0+k++67L+20007p3nvvbZlWAgBQ9Zrc4/qXv/wl3X333WmjjTYqBk7F7ZOf/GQx0j8uwxp1qAAAUPEe1ygFiFP4IcLrnDlz6gZwPfPMM83eQAAAeF89rkOGDCkuvRrlAnGZ1/POOy916dKluDTrhz70Ie8qAABtI7ieeuqp6e233y7+//3vfz/tt99+affdd099+vRJ119/fUu0EQAAmh5c99lnn7r/b7311unpp59Ob775ZnFFjvLMAgAAUPEa17jOc7nHtax3795CKwAAbSu4nnTSSammpiYdfvjh6Xe/+12zztsKAADNFlxfeeWVdN111xU9rIccckjq169fOu6449IDDzzQ1JcCAICWC66dOnUqBmRdffXVxUUIfvzjH6d//vOfaa+99kpbbbVVU18OAABaZnBWfXHlrBisNW/evDRr1qz01FNPfZCXAwCA5g2uixYtSjfffHPR63rXXXel2tradNhhh6Vf//rXqT1aNP/1SjeBVuBzBoB2FlwPPfTQNG3atKK3NWpcJ06cmHbdddfUnj37l9sq3QQAgKrX5OC6zjrrpBtuuKEoEYj/V4Ntdt0/rd9zo0o3g1bocfUlBQDaUXCN8oBqE6G1e+9+lW4GAEBVa/KsAgAAUAmCKwAA7Su4zpkzp2VbAgAAzRFct9tuu3TNNdes7eoAAFCZ4Hr22Wenr371q+nggw9Ob775ZvO2AgAAmiu4Hnvssemxxx5Lb7zxRho8eHC6/fbb1/apAADQutNhDRgwIN19993pZz/7WTrwwAPTtttumzp1avgSDz/88AdvFQAAfNB5XGfNmpVuuumm1KtXrzRy5MgVgisAALSEJqXOSy+9NI0bNy4NHz48PfHEE6lv374t0igAAHjfwfVzn/tc+tvf/laUCYwePXptnwYAAK0bXJctW1YMztpss82aZ8sAANASwXX69OlNeV0AAGhWLvkKAEAWBFcAALIguAIAkAXBFQCALLh6AEAVWjLvnUo3gVbgc6a9EVwBqtAr02dUugkATSa4AlShfntvnbr2Wq/SzaAVelx9SaE9EVwBqlCE1vX6dqt0M6gC/1qkXKEa/KuVPmfBFQBoMTc+q8eX5iO4AgAt5uBttk5911eWUg09rje2wpcUwRUAaDERWvt3717pZtBOmMcVAIAsCK4AAGRBcAUAIAuCKwAAWRBcAQDIguAKAEAWBFcAALIguAIAkAXBFQCALAiuAABkQXAFACALnSrdAOD/vP720ko3gVbgcwZ4fwRXaAO6d++eOnfulG567LVKN4VWEp93fO4ArD3BFdqAXr16pQkTvpsWLlyYqsncuXPT1KlT0xFHHJFqampSNYnQGp87AGtPcIU2IkJMtQaZCK21tbWVbgYAbZzBWQAAZEFwBQAgC4IrAABZEFwBAMiC4AoAQBYEVwAAsiC4AgCQBcEVAIAsCK4AAGRBcAUAIAuCKwAAWRBcAQDIguAKAEAWOlW6ATlYNP+NSjeBVuBzBoC2TXBdje7du6fOnTunZ/9ya6WbQiuJzzs+dwCg7RFcV6NXr15pwoQJaeHChZVuSquaO3dumjp1ajriiCNSTU1NqiYRWuNzh/Zu6ZvvVLoJtAKfM+2N4LoGEWKqNchEaK2tra10M4AWOJM0584ZlW4KrcSZJNoTwRWgijiT5EwS5ExwBagyziQ5kwS5ElwBgBbz2iJ1ttXgtVb6nAVXAKDF6ql//ax66mrRuRXqqQVXAKDZqadWT90SBFcAoEWop1ZPXXWXfH355ZeLby19+vRJ6623Xho6dGh66KGHKt0sAABaWZvucZ03b176xCc+kfbaa6/0+9//PvXt2zc999xzVfvtDQCgmrXp4HruuecW3exXXHFF3bIBAwZUtE0AAFRGmy4VuO2229JOO+2UDj744LTxxhunHXfcMV166aWrfc6SJUvSggULGtwAAMhfmw6uL7zwQrrooovSwIED0x/+8If09a9/PR1//PHpyiuvXOVzJk2alHr27Fl3UxgNANA+tOngunz58vSRj3wknXPOOUVv6zHHHJO+8pWvpIsvvniVzxk/fnyaP39+3W327Nmt2mYAAKowuPbr1y8NHjy4wbJtt902vfjii6t8TteuXVOPHj0a3AAAyF+bDq4xo8AzzzzTYNmzzz6btthii4q1CQCAymjTwfWkk05K//M//1OUCsyYMSNdc8016ZJLLknHHXdcpZsGAEAra9PBdeedd04333xzuvbaa9OQIUPSmWeemS644II0atSoSjcNAIBW1qbncQ377bdfcQMAoLq16R5XAAAoE1wBAMiC4AoAQBYEVwAAsiC4AgCQBcEVAIAsCK4AAGRBcAUAIAuCKwAAWRBcAQDIguAKAEAWBFcAALIguAIAkAXBFQCALAiuAABkQXAFACALgisAAFkQXAEAyILgCgBAFjpVugEAVIelS5emuXPnVmTb5e1WavuhpqYmdenSpWLbh/ZAcAWgVURonDx5ckXbMHXq1Ipte9y4cam2trZi24f2QHAFoNV6HCO8VfP+Ax+M4ApAq4jT5HocgQ/C4CwAALIguAIAkAWlAgBAu2IGi5p2O4OF4AoAtCtmsBjXbuvJBVcAoF0xg0VNaq8EVwCgXTGDRftlcBYAAFkQXAEAyILgCgBAFgRXAACyILgCAJAFwRUAgCwIrgAAZEFwBQAgC4IrAABZEFwBAMiC4AoAQBYEVwAAsiC4AgCQBcEVAIAsCK4AAGRBcAUAIAuCKwAAWRBcAQDIQqdKNwCorKVLl6a5c+dWZNvl7VZq+6GmpiZ16dKlYtsHYO0JrlDlIjROnjy5om2YOnVqxbY9bty4VFtbW7HtA7D2BFeoctHjGOGtmvcfgDwIrlDl4jS5HkcAcmBwFgAAWRBcAQDIguAKAEAWBFcAALIguAIAkAXBFQCALAiuAABkQXAFACALgisAAFkQXAEAyILgCgBAFgRXAACyILgCAJAFwRUAgCwIrgAAZEFwBQAgC4IrAABZ6FTpBrByS5cuTXPnzq3ItsvbrdT2Q01NTerSpUvFtg8AtD2CaxsVoXHy5MkVbcPUqVMrtu1x48al2traim0fAGh7BNc2KnocI7xV8/4DANQnuLZRcZpcjyMAwP8xOAsAgCwIrgAAZEFwBQAgC4IrAABZEFwBAMhCVsH1Bz/4QerQoUM68cQTK90UAABaWTbB9cEHH0y/+MUv0vbbb1/ppgAAUAFZBNeFCxemUaNGpUsvvTT16tWr0s0BAKACsgiuxx13XNp3333T8OHD17jukiVL0oIFCxrcAADIX5u/ctZ1112XHn744aJUYG1MmjQpnXHGGS3eLgAAWleb7nGdPXt2OuGEE9LVV1+d1l133bV6zvjx49P8+fPrbvEaAADkr0OpVCqlNuqWW25JX/jCF9I666xTt2zZsmXFzAIdO3YsygLqP7YyUSrQs2fPIsT26NGjFVoNAEBTrG1ea9OlAp/5zGfS448/3mDZUUcdlQYNGpS+853vrDG0AgDQfrTp4LrBBhukIUOGNFjWrVu31KdPnxWWAwDQvrXpGlcAAMiix3Vl7r333ko3AQCACtDjCgBAFgRXAACyILgCAJAFwRUAgCwIrgAAZEFwBQAgC4IrAABZEFwBAMiC4AoAQBYEVwAAsiC4AgCQBcEVAIAsCK4AAGRBcAUAIAuCKwAAWRBcAQDIguAKAEAWBFcAALIguAIAkAXBFQCALAiuAABkQXAFACALgisAAFkQXAEAyILgCgBAFgRXAACyILgCAJCFTpVuAFCdli9fnp5//vm0YMGC1KNHj7TVVluljh19lwZg1QRXoNU9+uij6dZbb01vvvlm3bLevXunkSNHpmHDhlW0bQC0XYIr0OqhdcqUKWnw4MFp9OjRqV+/fumVV15J06dPL5YfeeSRwisAK+W8HNCq5QHR0xqhdezYsWnLLbdMXbt2LX7G/Vh+2223FesBQGOCK9BqoqY1ygP23nvvFepZ4/7w4cPTG2+8UawHAI0JrkCriYFYIcoDVqa8vLweANQnuAKtJmYPCFHTujLl5eX1AKA+wRVoNTHlVcweEAOxGtexxv0777wz9enTp1gPABoTXIFWE3WsMeXVk08+mS677LI0c+bMtHjx4uJn3I/l+++/v/lcAVipDqVSqZTasaiV69mzZ5o/f77Tj9BGrGwe1+hpjdBqKiyA6rNgLfOaeVyBVhfhdOjQoa6cBUCTCK5ARURIHThwYKWbAUBGdG8AAJAFwRUAgCwIrgAAZEFwBQAgC4IrAABZEFwBAMiC4AoAQBYEVwAAsiC4AgCQBcEVAIAsCK4AAGRBcAUAIAuCKwAAWeiU2rlSqVT8XLBgQaWbAgDASpRzWjm3VW1wfeutt4qftbW1lW4KAABryG09e/Zc5eMdSmuKtplbvnx5mjNnTtpggw1Shw4dKt2cbL71RNCfPXt26tGjR6WbQzvmWKO1ONZoLY619yfiaITW/v37p44dO1Zvj2vs/GabbVbpZmQpfuH80tEaHGu0FscarcWx1nSr62ktMzgLAIAsCK4AAGRBcGUFXbt2Td/73veKn9CSHGu0FscarcWx1rLa/eAsAADaBz2uAABkQXAFACALgisAAFkQXAEAyILgykodeeSR6YADDqh0M6iSY8zxRmsfY445mkv5WBoxYkT63Oc+t9J17r///uLqnY899lirt6+9EVwBAD6gsWPHpunTp6eXXnpphceuuOKKtNNOO6Xtt9++Im1rTwRXAIAPaL/99kt9+/ZNU6ZMabB84cKF6cYbbyyCLR+c4AoA8AF16tQpjR49ugiu9afIj9C6bNmydNhhh1W0fe2F4AoA0AyOPvro9Pzzz6c//vGPDcoEvvjFL6aePXtWtG3theAKANAMBg0alHbbbbd0+eWXF/dnzJhRDMxSJtB8BFcAgGYSIfU3v/lNeuutt4re1q222irtsccelW5WuyG4AgA0k0MOOSR17NgxXXPNNemqq64qygdiKiyaR6dmeh0AgKrXvXv39KUvfSmNHz8+LViwoJjnleajxxUAoJnLBebNm5f22Wef1L9//0o3p13pUKo/ZwMAALRRelwBAMiC4AoAQBYEVwAAsiC4AgCQBcEVAIAsCK4AAGRBcAUAIAuCKwAAWRBcAQDIguAK0IziuuQdOnQobp07d041NTVp7733Tpdffnlavnz5Wr/OlClT0oYbbpgq0f4DDjig1bcLsDYEV4Bm9rnPfS698sor6Z///Gf6/e9/n/baa690wgknpP322y+99957lW4eQLYEV4Bm1rVr17TJJpukTTfdNH3kIx9JEyZMSLfeemsRYqMnNZx//vlp6NChqVu3bqm2tjYde+yxaeHChcVj9957bzrqqKPS/Pnz63pvTz/99OKxX/3qV2mnnXZKG2ywQbGNww8/PL322mt12543b14aNWpU6tu3b1pvvfXSwIED0xVXXFH3+OzZs9MhhxxS9Ob27t07jRw5sgjYIbZx5ZVXFm0tbzfaAtBWCK4AreDTn/50GjZsWLrpppuK+x07dkw/+clP0hNPPFGExbvvvjt9+9vfLh7bbbfd0gUXXJB69OhR9NzG7Vvf+lbx2LvvvpvOPPPM9Oijj6ZbbrmlCJ1xer9s4sSJ6cknnyxC8lNPPZUuuuiitNFGG9U9d5999ilC7/3335/+/Oc/p+7duxc9xEuXLi22EaG23GMct2gLQFvRqdINAKgWgwYNSo899ljx/xNPPLFu+ZZbbpnOOuus9LWvfS39/Oc/T126dEk9e/YsejyjV7W+o48+uu7/H/rQh4rwu/POOxe9tRFCX3zxxbTjjjsWvbLl1y67/vrrizrbX/7yl8Vrh+iNjd7X6Fn97Gc/W/TSLlmyZIXtArQFelwBWkmpVKoLjHfeeWf6zGc+U5QTRA/ol7/85fTGG2+kRYsWrfY1/v73v6cRI0akzTffvHjeHnvsUSyPwBq+/vWvp+uuuy7tsMMORQ/uAw88UPfc6KWdMWNG8bwIuXGLcoHFixen559/vkX3HaA5CK4ArSRO3Q8YMKA4vR8Dtbbffvv0m9/8pgijF154YbFOnLJflbfffrs41R8lBFdffXV68MEH080339zgeZ///OfTrFmz0kknnZTmzJlThONymUH0yn70ox9NjzzySIPbs88+W9TKArR1SgUAWkHUsD7++ONFoIygGqfsJ0+eXNS6hhtuuKHB+lEusGzZsgbLnn766aJX9gc/+EExoCs89NBDK2wrBmaNGTOmuO2+++7p5JNPTj/60Y+KgWJRLrDxxhsX4XdlVrZdgLZCjytAM4sa0VdffTW9/PLL6eGHH07nnHNOMXo/ellHjx6dtt5662Kg1E9/+tP0wgsvFDMFXHzxxQ1eI2pTo4f0rrvuSq+//npRQhDlAREsy8+77bbbioFa9Z122mnFrABREhADv6ZNm5a23Xbb4rGYbSAGakVbYnDWzJkzi9rW448/Pr300kt124063GeeeabYbrQToK0QXAGa2R133JH69etXhMAYoX/PPfcUg6giUK6zzjrF7AIxHda5556bhgwZUpz2nzRpUoPXiNH8MVjrS1/6UtGDet555xU/YzqtG2+8MQ0ePLjoeY2e1Poi2I4fP74oQ/jUpz5VbC9qXsP666+f7rvvviIAH3jggUWgHTt2bFHjWu6B/cpXvpI+/OEPF4O7Ynsx8wBAW9GhFKMFAACgjdPjCgBAFgRXAACyILgCAJAFwRUAgCwIrgAAZEFwBQAgC4IrAABZEFwBAMiC4AoAQBYEVwAAsiC4AgCQcvD/AW7BQFIqW88lAAAAAElFTkSuQmCC",
      "text/plain": [
       "<Figure size 800x600 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(8,6))\n",
    "sns.boxplot(data=df, x=\"dataset\", y=\"y\", hue=\"dataset\", palette=\"pastel\", legend=False)\n",
    "plt.title(\"Box Plot of Anscombe’s Quartet\")\n",
    "plt.xlabel(\"Dataset\")\n",
    "plt.ylabel(\"Y values\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "819c24b5-69de-4bfd-845d-712a85e71986",
   "metadata": {},
   "source": [
    "## Box Plot\n",
    "Boxes look to average out at around the value of $y = 7.5$ excet for Dataset II, showing how the line is also extending quite shorter in height as there is not much variance compared to the other datasets."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "529ab044-2007-493f-a122-b767d1d7fe36",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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JRGRtYOatWvXOv8+cuSIjBo1Svr27Su33HKLLvv+++9lwYIFMmnSJP9sKVEh03FoXpqVdZyamBiOU0NEFIQKNPdT+/bt5e9//7s89NBDbsvff/99naV73bp1Eow49xMVBEcUJiKy8YSWERERsmPHDqlbt67b8v3790vjxo0lJSVFghGDGiIiIuvJ6/m7QIPvVa1aVd54440sy998801dR0RERGSJLt0vvvii3H///fLll1/KTTfdpMs2bdokv/zyC6dJICIiooAoUE3NXXfdpU1NXbt2ld9//11v+BvLsI6IiIiosBUop8aqmFNDRERk3/N3npuffvrpJ2nQoIFOiYC/c3LjjTfmb2uJiIiIrlKegxr0asKkldHR0fo3BuLzVsmD5Rx8j4iIiII2qDl8+LBOWmn+TURERGTJoKZ69epe/yYiIiKybO8nTIfw+eefO++PGDFC54Zq1aqVHD161JfbR0REROS/oGbixIlSsmRJ55xPr7zyikydOlUqVKggQ4YMkVCSnpEum+M2yxeHvtB/cZ+IiIgsMvje8ePHpU6dOvr3smXL5IEHHpB//OMfcuutt0rbtm0lVKw5ukYmb5os8SnxzmUxETEyquUo6VC9Q0C3jYiIKNQUqKamdOnScvr0af171apVcuedd+rfJUqUkD/++ENCJaAZum6oW0ADCSkJuhzriYiIKMiDGgQxmKUbN9dRhH/++WepUaOG2B2amFBDY0jWLu3msimbprApioiIKNiDmjlz5sgtt9wiiYmJOtdT+fLldfnWrVvloYceErvblrAtSw2NZ2ATlxKn5YiIiCiIc2rQ0wnJwZ7Gjx8voSAxJdGn5YiIiChANTWwYcMGeeSRR7Qb94kTJ3TZwoUL5b///a/YXcWIij4tR0RERAEKatDk1KlTJ+3WvW3bNrl48aIux0RT6O5td02jm2ovJ4c4vK7H8tiIWC1HREREQRzUTJgwQebNmydvvPGGFC1a1LkcXboR5NhdeFi4dtsGz8DGvD+y5UgtR0REREEc1Ozbt09at26dZTmmBT979qyEAoxDM7PtTImOiHZbjhocLOc4NURERBZIFI6NjZUDBw5k6b6NfJpatWpJqEDg0q5qO+3lhKRg5NCgyakwamjQXTwQr0tERGSroGbAgAEyaNAgefvtt8XhcMjJkyd1uoRnnnlGxo4dK6EEgUSL2BaF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Yp/PAUVERER+Y563c+sgFFJBzblz5/RfTJVARERE1juPY2RhW01oWVAZGRk61UKZMmXE4fDdxHHmnFLHjx8PiekXQm1/Q3Gfub/2xv21t2Qb7i9CFQQ0lStXznFGgZCqqcEb4Tl4ny/hy2OXL1BehNr+huI+c3/tjftrb5E229+camhMTBQmIiIiW2BQQ0RERLbAoMYHihcvLuPGjdN/Q0Go7W8o7jP31964v/ZWPMT2N2QThYmIiMi+WFNDREREtsCghoiIiGyBQQ0RERHZAoMaIiIisgUGNT40efJkHal48ODBYlcnTpyQRx55RMqXLy8lS5aUhg0bypYtW8SO0tPT5dlnn5WaNWvqvtauXVuef/75XOcesYr169dL165ddYROfG+XLVvmth77OXbsWKlUqZLuf4cOHeSXX34Ru+7z5cuXZeTIkfqdLlWqlJbp3bu3jkJu18/Y1eOPP65lZs2aJXbe3z179si9996rA7nhc27RooUcO3ZM7Li/58+fl6eeekoHncVvuH79+jJv3jyxMwY1PrJ582Z57bXX5MYbbxS7OnPmjNx6661StGhR+fLLL2X37t0yY8YMueaaa8SOpkyZInPnzpVXXnlFD4S4P3XqVJk9e7bYwYULF6RRo0YyZ84cr+uxry+//LIeBH/88Uc9AXTq1ElSU1PFjvuckpIi27Zt00AW/37yySeyb98+PQHa9TM2LV26VH744Qc9OVpZbvt78OBBue2226RevXqybt06+emnn/TzLlGihNhxf4cOHSorV66U9957T49huOBGkLN8+XKxLXTppqtz7tw5o27dusbq1auNNm3aGIMGDTLsaOTIkcZtt91mhIq7777b6N+/v9uy7t27G7169TLsBoeCpUuXOu9nZGQYsbGxxrRp05zLzp49axQvXtz4z3/+Y9hxn73ZtGmTljt69Khh1/399ddfjSpVqhi7du0yqlevbrz44ouGHXjb3549exqPPPKIYUfe9veGG24w/v3vf7sta9q0qfGvf/3LsCvW1PjAk08+KXfffbdWz9sZovvmzZvLgw8+KNHR0dKkSRN54403xK5atWolX3/9tezfv1/v79ixQ/773/9Kly5dxO4OHz4scXFxbt9pVNffdNNN8v3330uoSEpK0mr9smXLil0n+X300Udl+PDhcsMNN4idYV8///xzufbaa7XGEccwfJ9zapKzwzFs+fLlmjaAuGft2rV6POvYsaPYFYOaq/TBBx9oVfWkSZPE7g4dOqTNMXXr1pWvvvpKBg4cKE8//bQsWLBA7GjUqFHyt7/9Tauq0eSGIA7Vt7169RK7Q0ADMTExbstx31xnd2hmQ47NQw89ZKtJAV2hSbVIkSL6O7a7hIQEzTFB7mPnzp1l1apV8te//lW6d+8u3377rdjR7NmzNY8GOTXFihXT/UZTVevWrcWuQmqWbl/DtO6DBg2S1atXW7ZNNr9XOqipmThxot7HSX7Xrl2ac9GnTx+xmw8//FAWLVok77//vl7Fbt++XYMa5B3YcX9J3JKGe/TooVe3COTtaOvWrfLSSy/pRRlqo0Lh+AX33XefDBkyRP9u3LixbNy4UY9hbdq0ETsGNT/88IPW1lSvXl0Ti9GygGOYXVsWWFNzlQcFRP9NmzbVqx3cEPEjuRJ/o/eMnaAXDKJ+V9dff71lew7kBlXyZm0NesSgmh4Hw1ColYuNjdV/4+Pj3ZbjvrnO7gHN0aNH9YLFrrU0GzZs0ONXtWrVnMcv7PMzzzwjNWrUELupUKGC7mOoHMP++OMPGTNmjMycOVN7SKETC5KEe/bsKdOnTxe7Yk3NVWjfvr3s3LnTbVm/fv20uQLV1uHh4QHbNn9Azyf0BnGF9llcAdgResOEhbnH/fhMzSs+O0M3dgQvyCnC1SwkJydrLyg0O9o9oEHXdeQfYOgCu0KQ7nm1jlwTLMdxzG7Q/ILu26FyDLt8+bLeQu0YxqDmKpQpU0YaNGjgtgzdXnEg9FxuB6ilQOIZmp9w4N+0aZO8/vrrerMjXN288MILeiWL5qf//e9/etXTv39/sQPkFxw4cMAtORhNbOXKldN9RlPbhAkTNIcKQQ66vqLaulu3bmLHfUZN5AMPPKDNMStWrNCaVjN/COtxUrTbZ+wZtCF3DMHsddddJ1aU2/6i9hU1FcgpadeunXZ3/uyzz7R7tx33t02bNrrPGKMGgRtaEt599109jtlWoLtf2Y2du3TDZ599ZjRo0EC79tarV894/fXXDbtKTk7Wz7JatWpGiRIljFq1amlXyIsXLxp2sHbtWu0G6nnr06ePs1v3s88+a8TExOjn3b59e2Pfvn2GXff58OHDXtfhhsfZ8TP2ZPUu3XnZ37feesuoU6eO/qYbNWpkLFu2zLDr/p46dcro27evUblyZd3f6667zpgxY4b+tu3Kgf8FOrAiIiIiulpMFCYiIiJbYFBDREREtsCghoiIiGyBQQ0RERHZAoMaIiIisgUGNURERGQLDGqIiIjIFhjUEBERkS0wqCEiIiJbYFBDREREtsCghogsKzExUSdgxCSrpo0bN+rkk5hhnIhCC+d+IiJL++KLL3TmcAQzmF26cePGct9999l7JmIi8opBDRFZ3pNPPilr1qyR5s2by86dO2Xz5s1SvHjxQG8WERUyBjVEZHl//PGHNGjQQI4fPy5bt26Vhg0bBnqTiCgAmFNDRJZ38OBBOXnypGRkZMiRI0cCvTlEFCCsqSEiS7t06ZK0bNlSc2mQUzNr1ixtgoqOjg70phFRIWNQQ0SWNnz4cPn4449lx44dUrp0aWnTpo1ERUXJihUrAr1pRFTI2PxERJa1bt06rZlZuHChREZGSlhYmP69YcMGmTt3bqA3j4gKGWtqiIiIyBZYU0NERES2wKCGiIiIbIFBDREREdkCgxoiIiKyBQY1REREZAsMaoiIiMgWGNQQERGRLTCoISIiIltgUENERES2wKCGiIiIbIFBDREREdkCgxoiIiISO/j/Ow5hLUoDfQQAAAAASUVORK5CYII=",
      "text/plain": [
       "<Figure size 640x480 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "for dataset in df['dataset'].unique():\n",
    "    data_subset = df[df['dataset'] == dataset]\n",
    "    x = data_subset['x']\n",
    "    y = data_subset['y']\n",
    "    m = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean())**2).sum()\n",
    "    b = y.mean() - m * x.mean()\n",
    "    y_pred = m * x + b\n",
    "    residuals = y - y_pred\n",
    "    plt.scatter(x, residuals, label=f'Dataset {dataset}')\n",
    "\n",
    "plt.axhline(0, color='black', linestyle='--')\n",
    "plt.xlabel('x')\n",
    "plt.ylabel('Residuals')\n",
    "plt.title(\"Combined Residual Plot for Anscombe's Quartet\")\n",
    "plt.legend()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0dc76e12-5a24-45db-8fcf-432182f45716",
   "metadata": {},
   "source": [
    "## Residual Plot\n",
    "The point of the Residual Plot tells us about the actal y compared to the predicted y value. You wil see the points are mostly scattered around the $y = 0$ line, with some outliers here and there. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "906edc5e-6afe-48af-8977-25fc0566d7bb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.plotly.v1+json": {
       "config": {
        "plotlyServerURL": "https://plot.ly"
       },
       "data": [
        {
         "mode": "markers",
         "name": "Dataset I",
         "type": "scatter",
         "x": {
          "bdata": "CggNCQsOBgQMBwU=",
          "dtype": "i1"
         },
         "xaxis": "x",
         "y": {
          "bdata": "FK5H4XoUIEDNzMzMzMwbQFK4HoXrUR5AH4XrUbieIUApXI/C9aggQOxRuB6F6yNA9ihcj8L1HEAK16NwPQoRQK5H4XoUriVASOF6FK5HE0C4HoXrUbgWQA==",
          "dtype": "f8"
         },
         "yaxis": "y"
        },
        {
         "mode": "markers",
         "name": "Dataset II",
         "type": "scatter",
         "x": {
          "bdata": "CggNCQsOBgQMBwU=",
          "dtype": "i1"
         },
         "xaxis": "x2",
         "y": {
          "bdata": "SOF6FK5HIkBI4XoUrkcgQHsUrkfheiFACtejcD2KIUCF61G4HoUiQDMzMzMzMyBAhetRuB6FGEDNzMzMzMwIQMP1KFyPQiJACtejcD0KHUD2KFyPwvUSQA==",
          "dtype": "f8"
         },
         "yaxis": "y2"
        },
        {
         "mode": "markers",
         "name": "Dataset III",
         "type": "scatter",
         "x": {
          "bdata": "CggNCQsOBgQMBwU=",
          "dtype": "i1"
         },
         "xaxis": "x3",
         "y": {
          "bdata": "16NwPQrXHUAUrkfhehQbQHsUrkfheilAcT0K16NwHEA9CtejcD0fQK5H4XoUriFAUrgehetRGECPwvUoXI8VQM3MzMzMTCBArkfhehSuGUDsUbgehesWQA==",
          "dtype": "f8"
         },
         "yaxis": "y3"
        },
        {
         "mode": "markers",
         "name": "Dataset IV",
         "type": "scatter",
         "x": {
          "bdata": "CAgICAgICBMICAg=",
          "dtype": "i1"
         },
         "xaxis": "x4",
         "y": {
          "bdata": "UrgehetRGkAK16NwPQoXQNejcD0K1x5ArkfhehSuIUBxPQrXo/AgQClcj8L1KBxAAAAAAAAAFUAAAAAAAAApQD0K16NwPRZApHA9CtejH0CPwvUoXI8bQA==",
          "dtype": "f8"
         },
         "yaxis": "y4"
        }
       ],
       "layout": {
        "annotations": [
         {
          "font": {
           "size": 16
          },
          "showarrow": false,
          "text": "Dataset I",
          "x": 0.225,
          "xanchor": "center",
          "xref": "paper",
          "y": 1,
          "yanchor": "bottom",
          "yref": "paper"
         },
         {
          "font": {
           "size": 16
          },
          "showarrow": false,
          "text": "Dataset II",
          "x": 0.775,
          "xanchor": "center",
          "xref": "paper",
          "y": 1,
          "yanchor": "bottom",
          "yref": "paper"
         },
         {
          "font": {
           "size": 16
          },
          "showarrow": false,
          "text": "Dataset III",
          "x": 0.225,
          "xanchor": "center",
          "xref": "paper",
          "y": 0.375,
          "yanchor": "bottom",
          "yref": "paper"
         },
         {
          "font": {
           "size": 16
          },
          "showarrow": false,
          "text": "Dataset IV",
          "x": 0.775,
          "xanchor": "center",
          "xref": "paper",
          "y": 0.375,
          "yanchor": "bottom",
          "yref": "paper"
         }
        ],
        "height": 600,
        "template": {
         "data": {
          "bar": [
           {
            "error_x": {
             "color": "#2a3f5f"
            },
            "error_y": {
             "color": "#2a3f5f"
            },
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "bar"
           }
          ],
          "barpolar": [
           {
            "marker": {
             "line": {
              "color": "#E5ECF6",
              "width": 0.5
             },
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "barpolar"
           }
          ],
          "carpet": [
           {
            "aaxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "baxis": {
             "endlinecolor": "#2a3f5f",
             "gridcolor": "white",
             "linecolor": "white",
             "minorgridcolor": "white",
             "startlinecolor": "#2a3f5f"
            },
            "type": "carpet"
           }
          ],
          "choropleth": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "choropleth"
           }
          ],
          "contour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "contour"
           }
          ],
          "contourcarpet": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "contourcarpet"
           }
          ],
          "heatmap": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "heatmap"
           }
          ],
          "histogram": [
           {
            "marker": {
             "pattern": {
              "fillmode": "overlay",
              "size": 10,
              "solidity": 0.2
             }
            },
            "type": "histogram"
           }
          ],
          "histogram2d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2d"
           }
          ],
          "histogram2dcontour": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "histogram2dcontour"
           }
          ],
          "mesh3d": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "type": "mesh3d"
           }
          ],
          "parcoords": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "parcoords"
           }
          ],
          "pie": [
           {
            "automargin": true,
            "type": "pie"
           }
          ],
          "scatter": [
           {
            "fillpattern": {
             "fillmode": "overlay",
             "size": 10,
             "solidity": 0.2
            },
            "type": "scatter"
           }
          ],
          "scatter3d": [
           {
            "line": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatter3d"
           }
          ],
          "scattercarpet": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattercarpet"
           }
          ],
          "scattergeo": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergeo"
           }
          ],
          "scattergl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattergl"
           }
          ],
          "scattermap": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermap"
           }
          ],
          "scattermapbox": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scattermapbox"
           }
          ],
          "scatterpolar": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolar"
           }
          ],
          "scatterpolargl": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterpolargl"
           }
          ],
          "scatterternary": [
           {
            "marker": {
             "colorbar": {
              "outlinewidth": 0,
              "ticks": ""
             }
            },
            "type": "scatterternary"
           }
          ],
          "surface": [
           {
            "colorbar": {
             "outlinewidth": 0,
             "ticks": ""
            },
            "colorscale": [
             [
              0,
              "#0d0887"
             ],
             [
              0.1111111111111111,
              "#46039f"
             ],
             [
              0.2222222222222222,
              "#7201a8"
             ],
             [
              0.3333333333333333,
              "#9c179e"
             ],
             [
              0.4444444444444444,
              "#bd3786"
             ],
             [
              0.5555555555555556,
              "#d8576b"
             ],
             [
              0.6666666666666666,
              "#ed7953"
             ],
             [
              0.7777777777777778,
              "#fb9f3a"
             ],
             [
              0.8888888888888888,
              "#fdca26"
             ],
             [
              1,
              "#f0f921"
             ]
            ],
            "type": "surface"
           }
          ],
          "table": [
           {
            "cells": {
             "fill": {
              "color": "#EBF0F8"
             },
             "line": {
              "color": "white"
             }
            },
            "header": {
             "fill": {
              "color": "#C8D4E3"
             },
             "line": {
              "color": "white"
             }
            },
            "type": "table"
           }
          ]
         },
         "layout": {
          "annotationdefaults": {
           "arrowcolor": "#2a3f5f",
           "arrowhead": 0,
           "arrowwidth": 1
          },
          "autotypenumbers": "strict",
          "coloraxis": {
           "colorbar": {
            "outlinewidth": 0,
            "ticks": ""
           }
          },
          "colorscale": {
           "diverging": [
            [
             0,
             "#8e0152"
            ],
            [
             0.1,
             "#c51b7d"
            ],
            [
             0.2,
             "#de77ae"
            ],
            [
             0.3,
             "#f1b6da"
            ],
            [
             0.4,
             "#fde0ef"
            ],
            [
             0.5,
             "#f7f7f7"
            ],
            [
             0.6,
             "#e6f5d0"
            ],
            [
             0.7,
             "#b8e186"
            ],
            [
             0.8,
             "#7fbc41"
            ],
            [
             0.9,
             "#4d9221"
            ],
            [
             1,
             "#276419"
            ]
           ],
           "sequential": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ],
           "sequentialminus": [
            [
             0,
             "#0d0887"
            ],
            [
             0.1111111111111111,
             "#46039f"
            ],
            [
             0.2222222222222222,
             "#7201a8"
            ],
            [
             0.3333333333333333,
             "#9c179e"
            ],
            [
             0.4444444444444444,
             "#bd3786"
            ],
            [
             0.5555555555555556,
             "#d8576b"
            ],
            [
             0.6666666666666666,
             "#ed7953"
            ],
            [
             0.7777777777777778,
             "#fb9f3a"
            ],
            [
             0.8888888888888888,
             "#fdca26"
            ],
            [
             1,
             "#f0f921"
            ]
           ]
          },
          "colorway": [
           "#636efa",
           "#EF553B",
           "#00cc96",
           "#ab63fa",
           "#FFA15A",
           "#19d3f3",
           "#FF6692",
           "#B6E880",
           "#FF97FF",
           "#FECB52"
          ],
          "font": {
           "color": "#2a3f5f"
          },
          "geo": {
           "bgcolor": "white",
           "lakecolor": "white",
           "landcolor": "#E5ECF6",
           "showlakes": true,
           "showland": true,
           "subunitcolor": "white"
          },
          "hoverlabel": {
           "align": "left"
          },
          "hovermode": "closest",
          "mapbox": {
           "style": "light"
          },
          "paper_bgcolor": "white",
          "plot_bgcolor": "#E5ECF6",
          "polar": {
           "angularaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "radialaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "scene": {
           "xaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "yaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           },
           "zaxis": {
            "backgroundcolor": "#E5ECF6",
            "gridcolor": "white",
            "gridwidth": 2,
            "linecolor": "white",
            "showbackground": true,
            "ticks": "",
            "zerolinecolor": "white"
           }
          },
          "shapedefaults": {
           "line": {
            "color": "#2a3f5f"
           }
          },
          "ternary": {
           "aaxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "baxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           },
           "bgcolor": "#E5ECF6",
           "caxis": {
            "gridcolor": "white",
            "linecolor": "white",
            "ticks": ""
           }
          },
          "title": {
           "x": 0.05
          },
          "xaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          },
          "yaxis": {
           "automargin": true,
           "gridcolor": "white",
           "linecolor": "white",
           "ticks": "",
           "title": {
            "standoff": 15
           },
           "zerolinecolor": "white",
           "zerolinewidth": 2
          }
         }
        },
        "title": {
         "text": "Anscombe's Quartet"
        },
        "width": 800,
        "xaxis": {
         "anchor": "y",
         "domain": [
          0,
          0.45
         ]
        },
        "xaxis2": {
         "anchor": "y2",
         "domain": [
          0.55,
          1
         ]
        },
        "xaxis3": {
         "anchor": "y3",
         "domain": [
          0,
          0.45
         ]
        },
        "xaxis4": {
         "anchor": "y4",
         "domain": [
          0.55,
          1
         ]
        },
        "yaxis": {
         "anchor": "x",
         "domain": [
          0.625,
          1
         ]
        },
        "yaxis2": {
         "anchor": "x2",
         "domain": [
          0.625,
          1
         ]
        },
        "yaxis3": {
         "anchor": "x3",
         "domain": [
          0,
          0.375
         ]
        },
        "yaxis4": {
         "anchor": "x4",
         "domain": [
          0,
          0.375
         ]
        }
       }
      },
      "image/png": "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"
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "from plotly.subplots import make_subplots\n",
    "import plotly.graph_objects as go\n",
    "\n",
    "fig = make_subplots(\n",
    "    rows=2, cols=2, \n",
    "    subplot_titles=[\"Dataset I\", \"Dataset II\", \"Dataset III\", \"Dataset IV\"]\n",
    ")\n",
    "\n",
    "datasets = [\"I\", \"II\", \"III\", \"IV\"]\n",
    "positions = [(1,1), (1,2), (2,1), (2,2)]\n",
    "\n",
    "for ds, pos in zip(datasets, positions):\n",
    "    subset = df[df[\"dataset\"] == ds]\n",
    "    fig.add_trace(\n",
    "        go.Scatter(x=subset[\"x\"], y=subset[\"y\"], mode=\"markers\", name=f\"Dataset {ds}\"),\n",
    "        row=pos[0], col=pos[1]\n",
    "    )\n",
    "\n",
    "fig.update_layout(height=600, width=800, title_text=\"Anscombe's Quartet\")\n",
    "fig.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "30012633-e302-4c3d-af6e-0b21da88c411",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig.write_html(\"anscombe_quartet.html\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7dd10085-eeb3-44d8-bf4d-307c773266b7",
   "metadata": {},
   "source": [
    "## Interpretation:\n",
    "All four datasets have similar statistics, their scatter plots show different patterns. \n",
    "\n",
    "Dataset 1 looks like it is fully random, but when drawing a line of best fit, it looks like a $y = x$ linear relationship.\n",
    "\n",
    "Dataset 2 shows a slight curve pattern overall, mimicing a $y=-x^2$ relationship\n",
    "\n",
    "Dataset 3 shows a typical linear relationship of $y = -x$ except for one outlier\n",
    "\n",
    "Dataset 4 shows that almost all x-values are the same except for one outlier\n",
    "\n",
    "These graphs - scatter plots, regression lines, resdiual plots and box plots have been used because summary statistics by themselves can be misleading. All four datasets have the same mean, variance, and correlation, but the residual plots reveal differences in patterns, nonlinearity, and outliers. \n",
    "\n",
    "These graphs desmontrate why visualizing data is cruicial before coming to a conclusion. \n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3239cd0e-9586-4501-948b-fe1bdad9b0e4",
   "metadata": {},
   "source": [
    "## Conclusion:\n",
    "In conclusion, the residual plots of Anscombe’s quartet demonstrate that similar summary statistics do not guarantee similar data behavior. While all four datasets share the same mean, variance, and linear regression equation $y = mx + b$, the residuals reveal very different patterns: Dataset I fits the linear model well, Dataset II shows a nonlinear trend, and Datasets III and IV are affected by outliers. This highlights the importance of visualizing data using different types of graphs—scatter plots, regression lines, and residual plots before interpreting results. Overall, residual analysis is a powerful tool for assessing model fit, detecting nonlinearity, and identifying influential points that can skew conclusions."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
