# Case Study: Pricing for the Meal Delivery Routing Problem

This directory is supplementary material for our work submitted for review to Transportation Research: Part C: 

[Pricing for Multi-modal Pickup and Delivery Problems with
Heterogeneous Users](google.com) Mark Beliaev, Negar Mehr, Ramtin Pedarsani

All relevant citations for methods used are found in the paper's list of references.

This README contains 4 sections:

[I. Requirements](#i.-installation)

List of the requirements needed to run the code, as well as instructions on how to setup the required environemnt.

[II. Contents](#ii.-contents)

Summary of the sub-directories and files found within this project.

## I. Installation 

We recommend using pacakge manager [pip](https://pip.pypa.io/en/stable/) as well as 
[conda](https://www.anaconda.com/products/individual) to install the relative packages:

- python-3.9.16 [python](https://www.python.org/downloads/release/)
- numpy-1.24.2 [numpy](https://numpy.org/devdocs/release/1.24.2-notes.html)
- pandas-1.5.3 [pandas](https://github.com/mechmotum/cyipopt)
- cyipopt-1.2.0 [cyipopt](https://github.com/mechmotum/cyipopt)

The instaces used for our first case study are saved in **/instances** and can be downloaded directly from [grubhub instances](https://github.com/grubhub/mdrplib) along with their documentation.

The instances used for our second case study are stored remotely and loaded in **/notes/chicago_data.ipynb**, where they are converted to the format required for our use. The converted data is saved at **/results/all.p**. The raw **.csv** data can be downloaded directly from [chicago instances](https://data.cityofchicago.org/Transportation/Transportation-Network-Providers-Trips-2023-/n26f-ihde), whcih was accessed July 24 2023.


## II. Contents
The folder **\instances** folder contains data for the meal delivery problem. 

The folder **\utils** contains classes and functions required to run and evaluate our results. 

- **\utils\nlp_mdrp.py** contains the class which initializes our non-linear program to be solved with "ipopt" interface.

- **\utils\convert.py** contains the function responsible for loading instance paramaters from the files provided by  the [grubhub instances](https://github.com/grubhub/mdrplib). These parameters are converted to our desired format in order to specify the non-linear program and solve it. 

The jupyter notebooks provided **case.ipynb, chicago_case.ipynb** load the corresponding instance used in the paper and run the ip-opt algorithm on it to find the solution. These solutions are then evaluated, and the resulting tables correspond to the results found in the paper. Note that the convert_instrance() methods found in **\utils\convert.py** is seeded, so the results should be identical to the ones reported. We also note that the results for prices in the first table, when only cars are present, are irrelevant in the notebook as the comparing latencies between modalities when only one transportation mode is present is irrelevant. The prices are simply equal to the operational cost of cars, making the Nash Eq. trivial. We ignore these and report the true values in the paper since solving the NLP with only one modality is trivial. Lastly, the solution for the second case study provides two additional cells for checking the Nash Eq. both explicitly (directly checking the inequalities), and implicitly (checking for the condition that is derived in the Appendix of the paper.) A count of zero represents that all conditions were satisfied, where the break statement was used to check this condition for both settings

