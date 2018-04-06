This repository is mainly about machine learning and data mining
====
This repository includes four main parts'

	An example of data mining of Diabetes dataset
	An example of using keras for kaggle otto classification
	An example of using MXNET for kaggle dog competition
	An example of using Sk-learn SVM classifier
		
The first practice is the Diabetes 130-US hospitals for years 1999-2008 Data-Set
----

The dataset represents 10 years (1999-2008) of clinical care at 130 US hospitals and integrated delivery networks. It includes over 50 features representing patient and hospital outcomes. Information was extracted from the database for encounters that satisfied the following criteria.The data contains such attributes as patient number, race, gender, age, admission type, time in hospital, medical specialty of admitting physician, number of lab test performed, HbA1c test result, diagnosis, number of medication, diabetic medications, number of outpatient, inpatient, and emergency visits in the year before the hospitalization, etc.

	(1) It is an inpatient encounter (a hospital admission).
	(2) It is a diabetic encounter, that is, one during which any kind of diabetes was entered to the system as a diagnosis.
	(3) The length of stay was at least 1 day and at most 14 days.
	(4) Laboratory tests were performed during the encounter.
	(5) Medications were administered during the encounter.

This dataset is complicated and unbalanced. Let's do it!
----
Look at the label:`eadmitted`,we visualize it by groups
![](https://github.com/BoXiao123/Machine-Learning-and-Data-Mining-Practice/raw/master/img/Figure_1.png)

there are three types of labels. We just regroup it to two labels.

	data['readmitted'] = pd.Series([0 if val == 'NO' else 1 for val in data['readmitted']])

Then we look at the feature:`age`, we visualize it by groups

![](https://github.com/BoXiao123/Machine-Learning-and-Data-Mining-Practice/raw/master/img/Figure_2.png)

We find it really unbalance,then regroup this feature.

![](https://github.com/BoXiao123/Machine-Learning-and-Data-Mining-Practice/raw/master/img/Figure_3.png)

It is better now.

There are two many features in this dataset, but many of features are noisy. We use random forest to evaluate the importance of each features and rank them.

![](https://github.com/BoXiao123/Machine-Learning-and-Data-Mining-Practice/raw/master/img/Figure_4.png)

Ok, we just select Top 10 features for the following procedure. We have washed this dataset and could apply our machine learning algorithm. 

Using k-means for clustering and ROC for evaluation

![](https://github.com/BoXiao123/Machine-Learning-and-Data-Mining-Practice/raw/master/img/Figure_5.png)

![](https://github.com/BoXiao123/Machine-Learning-and-Data-Mining-Practice/raw/master/img/Figure_7.png)

The second practice is kaggle otto classification
-----
The Otto Group is one of the world’s biggest e-commerce companies, with subsidiaries in more than 20 countries, including Crate & Barrel (USA), Otto.de (Germany) and 3 Suisses (France). We are selling millions of products worldwide every day, with several thousand products being added to our product line.

A consistent analysis of the performance of our products is crucial. However, due to our diverse global infrastructure, many identical products get classified differently. Therefore, the quality of our product analysis depends heavily on the ability to accurately cluster similar products. The better the classification, the more insights we can generate about our product range.

![](https://github.com/BoXiao123/Machine-Learning-and-Data-Mining-Practice/raw/master/img/11.jpg)

For this competition, there have provided a dataset with 93 features for more than 200,000 products. The objective is to build a predictive model which is able to distinguish between our main product categories.

We are going to build a simple ANN to do this work. The structure is shown below

![](https://github.com/BoXiao123/Machine-Learning-and-Data-Mining-Practice/raw/master/img/12.png)


The Fourth practice is using SVM
----
The SVM is an important classify in machine learning, we need to practice how to use the SVC in sklearn lib. You can see all codes in SVM.py

First we create some fake data for classification

![](https://github.com/BoXiao123/Machine-Learning-and-Data-Mining-Practice/raw/master/img/Figure_8.png)

Then we use 10 fold cross validation to choose paramater gamma

![](https://github.com/BoXiao123/Machine-Learning-and-Data-Mining-Practice/raw/master/img/Figure_9.png)

We can adopt SVC and draw the hyperplane

![](https://github.com/BoXiao123/Machine-Learning-and-Data-Mining-Practice/raw/master/img/Figure_10.png)



