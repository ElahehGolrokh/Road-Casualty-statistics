# Road-Casualty-statistics
This project aims to analyze driving accident data to identify patterns and factors contributing to accidents and to provide solutions for reducing them.
This version only supports LogisticRegression classifier. In future versions svm and RF classifiers are going to be implemented.  <br>  
The notebooks containing full version of data cleaning and EDA phase are stored in notebooks folder.
This folder is containing the EDA of full data, pedestrians, car passengers and bus passengers.

```shell
python pipeline.py --clean --train --evaluate
```
You can specify some more details with these arguments:  <br>  
--config: default = config.yaml,  <br>  
          help = path to the config file  <br>  
--model_name: default = model,  <br>  
              help = The name used for saving the trained model  <br>  
--model: default = LogisticRegression,  <br>  
          help = choose one of the sklearn classification models  <br>  
--classifier_package: default = sklearn.linear_model,  <br>  
                      help = choose one of the sklearn classification models  <br>  

You can see the documentation for this research in this link:
https://docs.google.com/document/d/1VrPaA3A7noqAaNo_dyCth6nNfKxIQz8r2JCokE3iqjQ/edit#heading=h.dm3sgcgvgd4g

Also the presentation file is accesible in this link:
https://docs.google.com/presentation/d/1HvPwdv6Sc2e23845jWx6VvSTV-9G6lcnM8CwZiaO-LM/edit#slide=id.ge9090756a_2_12
