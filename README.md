# Crabby 
Release Date: **TBD**
Training data gathered between: 27th March 2024 - 13th August 2024

# TODO:
- [ ] Call the test set the validation set, so that we’re clear
- [ ] Make yaml file with the requirements
- [ ] Call the Update models the Day N models 
- [ ] Number the notebooks in this directory
- [ ] Cross reference all relevant notebooks in the Overview notebook
- [ ] Overview notebook:
    - [ ] Recall at rank K define
    - [ ] Add AXES to all my recall at rank K plots
    - [ ] Permutation importance explanation
    - [ ] Explain: real scores, gal scores, Good, PM, Garbage, Galactic (with caveats)
    - [ ] Explain the features and cross reference the data notebooks
    - [ ] Add conclusion about the importance of the LC update features:
        - [ ] DET N Total and others that are useless are tricky features trying to make up for the many caveats associated wiht using the non forced data
        - [ ] in future iteration we will look at how far down the VRA scores we can go to calculate the FP and use it to improve the model. 
- [ ] Policy notebook -> Write down in words the number of good lost (in the pie chart numbers too small) 
- [ ] features\_update:
    - [ ] Rename
    - [ ] Finish the intro 


# Overview
##  Quick Summary

The **Virtual Research Assistant** is a bot (or set of bots) that help ATLAS eyeballers by ordering the alerts in the eyeball list, removing the crappiest objects, and [TBW] sending automatic triggers for transients within 100 Mpc to be followed up with the **[BLABLA telescope - @stephen add text]**.
**Crabby** is a family or version of VRA models created using data gathered between 27th March and 13th August 2024.
This is the first public release of the VRA data and models, previous iterations (Arin, BMO) are not available online but can be requested (it's not particularly interesting but you can have it). 

### Who is this repo for and how to use it.
* 1) **Users (eyeballers) who want to understand the models and its limitation**: You probably wnat to focus on the `Overview` and the `Key_transients` notebooks. 
* 2) **Scientist who want to understand the method**: In addition to the high level summary notebooks you might want to check the hyper parameter tuning codes and our metric to chose the best model (the **Area under the Recall at rank K**).
* 3) **Anyone who wants to reproduce the results**: Run everything and see if works. If it doesn't please send us an email. 

### Requirements:
```
matplotlib, numpy, pandas, scikit-leanrn, joblib, atlasvras, atlasapiclient
```

# Description
[TBW]
## Cleaning the Data and extracting the features


## Training
### Hyperparameter tuning
- what HP we're trying out
- how we're doing it (say what the python and bash scripts are)
- say how we're picking our favourite

### VRA policies: auto-garbaging and eyeballing strategy





## Directory structure

```
data/
| clean_data_csv/
| features_and_labels_csv/
| | day1/
| | update/
| figures/
| summary_plots.ipynb
| clean_data.ipynb
| features_update.ipynb  
| features_day1.ipynb 
| schema_doc.html 
| schema.json
| vra_with_decisions.csv
|
train_scoring_models/
| 

```
