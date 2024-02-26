# Road-Casualty-statistics
This project aims to analyze driving accident data to identify patterns and factors contributing to accidents and to provide solutions for reducing them. This analysis and package is developed based on this kaggle dataset:  <br>  
https://www.kaggle.com/datasets/juhibhojani/road-accidents-data-2022/data
This version only supports LogisticRegression classifier. In future versions svm and RF classifiers are going to be implemented.  <br>
To see more details about different features including non_informatives, categorical features, numerical features, target column, features in which -1 meaning null and  features in which 9 meaning null you can refere to the config file. Note that these information are for the full analysis. For pedestrians, car passengers and bus passengers which we have EDA for them separately, you have to redefine these features.

## Exploratory Data Analysis
The notebooks containing full version of data cleaning and EDA phase are stored in notebooks folder.  <br>  
This folder is containing the EDA of full data, pedestrians, car passengers and bus passengers.  <br>  
You can see the documentation for this research in this link:
https://docs.google.com/document/d/1VrPaA3A7noqAaNo_dyCth6nNfKxIQz8r2JCokE3iqjQ/edit#heading=h.dm3sgcgvgd4g

Also the presentation file is accesible in this link:
https://docs.google.com/presentation/d/1HvPwdv6Sc2e23845jWx6VvSTV-9G6lcnM8CwZiaO-LM/edit#slide=id.ge9090756a_2_12

## Run The ML Pipeline
In order to run the entire ML pipeline, you can run this command:

```shell
python pipeline.py --clean --train --evaluate
```
If you want to ignore specific steps, don't pass the corresponding option. Also you can specify some more details with these arguments:  <br>  
--config: default = config.yaml,  <br>  
          help = path to the config file  <br>  
--model_name: default = model,  <br>  
              help = The name used for saving the trained model  <br>  
--model: default = LogisticRegression,  <br>  
          help = choose one of the sklearn classification models  <br>  
--classifier_package: default = sklearn.linear_model,  <br>  
                      help = choose one of the sklearn classification models  <br>  

## Get Predictions Using Saved Model
In order to get predictions out of a unlabeled new data stored in a csv file, you can run this command:

```shell
python inference.py 
```
You can specify some more details with these arguments:  <br>  
--config_path: default = config.yaml  <br>  
               help = path to config file  <br>  
--model_path: default = run/models/model  <br>  
                    help = path to saved ML model  <br>  
--data_path: default = data/test.csv  <br>  
                    help = path to test csv file  <br>  
