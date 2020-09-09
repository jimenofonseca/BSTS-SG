# BSTS-SG 1.0
![GitHub license](https://img.shields.io/github/license/JIMENOFONSECA/BSTS-SG) ![Repo Size](https://img.shields.io/github/repo-size/JIMENOFONSECA/BSTS-SG) ![Lines](https://github.com/JIMENOFONSECA/BSTS-SG/blob/image-data/badge-lines-of-code.svg)

 Bayesian Structural Timeseries Model for the Causal Impact Analysis of Behavioral 
 Interventions on Energy Consumption in Singapore
 
 ## How does it work

 This is an ensemble model combining multi-physics and data-driven building energy consumption models
 into a Bayesian Structural Timeseries Model.
 
 ![summary](https://github.com/JIMENOFONSECA/BSTS-SG/blob/master/images/summary.PNG)
 
 More information is coming soon...

## Installation

- Clone this repository
- Install dependencies in setup.py

  - [`EnthalpyGradients==1.0`](https://pypi.org/project/EnthalpyGradients/)
  - `numpy`
  - `pandas`
  - `Scikit-learn==0.20.0`
  - `pycausalimpact==0.1.1`
  - `cityenergyanalyst==3.0`

## FAQ

- Where are the results stored? A: the results are inside the results folder / final_results.csv
- Where is the raw observed data? A: It is propriety. Hourly weather observations were purchased from [`NEA`](https://www.nea.gov.sg/weather), daily electricity consumption was directly obtained via the REST API of the manufacturer [`SMAPEE`](https://smappee.atlassian.net/wiki/spaces/DEVAPI/overview).
- Where is the raw synthetic data? A: It is open-source. GIS and BIM data is obatined with the open-source software [`City Energy Analyst`](https://cityenergyanalyst.com/)


## Cite

J. A. Fonseca, BSTS-SG 1.0. An ensemble timeseries model for the evaluation of energy conservation strategies in Singaporean households. Report – Cooling Singapore, ETH Zurich. 2020 
