# Census Classifier

Observation: This Bank context is fictitious. Created by me to simulate a real problem that might be solved using Data Science technics

It's known that to provide a good prediction about the risk of offering credit to the bank's customers is necessary to analyze lots of features that impact this problem. One of the most important features that might serve to the prediction model is the income of each people, how much the person earns for sure is a feature that brings a piece of good information to make a good analysis and afford a good model.

This project has its focus on providing a response about the income of the customers,  through a machine learning classifier algorithm to make these predictions. For Helping the bank to make better analyzes, and to choose what customers to provide more credit to or not. 
objective: Predict whether income exceeds $50K/yr based on census data.

Summarized Steps:

- 1. Load
- 2. Problem Hypothesis
- 3. EDA
- 4. Filtering variables
- 5. Data Preparation
- 6. Machine Learning
- 7. Tune
- 8. Understanding The Result
- 9. Deploy

I took this Dataset from: https://archive.ics.uci.edu/ml/datasets/adult

Attribute Information:

Listing of attributes:

- **Income**: >50K, <=50K.

- **age**: continuous.
- **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- **fnlwgt**: continuous.
- **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- **education-num**: continuous.
- **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- **sex:** Female, Male.
- **capital-gain:** continuous.
- **capital-loss:** continuous.
- **hours-per-week:** continuous.
- **native-country:** United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
