# Case Study: Pricing for the Meal Delivery Routing Problem

This directory is supplementary material for our work presented at "": 

[Paper Title](google.com) Authors, , .s

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

The instaces used for our case study and saved in **\instances** can be found in [grubhub instances](https://github.com/grubhub/mdrplib) along with their documentation.

## II. Contents

The folder **\utils** contains classes and functions required to run and evaluate our results. 

The folder **\results** contains saved results after running our main script. 

The folder **\instances** folder contains data for the meal delivery problem. 

- **\utils\nlp_mdrp.py** contains the class which initializes our non-linear program to be solved with "ipopt" interface.
- **\utils\load_instance.py** contains the function responsible for loading instance paramaters from the files provided by  the [grubhub instances](https://github.com/grubhub/mdrplib). These parameters are converted to our desired format in order to specify the non-linear program and solve it. 

The script **run.py** loads the corresponding instance and runs the ip-opt algorithm on it to find the solution. The solution along with the paramters are saved in **\results** for evaluation.

The script **eval.py** loads the corresponding solution and evaluates it by providing relevant statistics such as prices, average travel time, cost, etc. 

