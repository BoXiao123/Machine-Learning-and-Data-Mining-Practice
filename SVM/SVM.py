import numpy as np
import scipy as sp
import sklearn
import matplotlib.pyplot as plt
ID = 7023839
np.random.seed(ID)
clustercentres = np.array([[0.5,0.65],[0.35,0.4]])
centreshifts = np.random.random((2,2))*0.02
clustercentres = clustercentres + centreshifts
print clustercentres
noisescale = 0.25
nclasses = 2
npoints = 100
data = np.zeros((nclasses*npoints,2), dtype=float)
classes = np.zeros(nclasses*npoints, dtype=int)
for i in range(nclasses):
    for j in range(npoints):
        randomshift = np.random.random((2,))
        data[i*npoints+j,:] = clustercentres[i,:] + noisescale*randomshift[0]*np.array([np.sin(randomshift[1]*2*np.pi),np.cos(randomshift[1]*2*np.pi)])
        classes[i*npoints+j] = i
#Produce a scatter plot of your data with a different colour for each class
class0=data[np.where(classes==0)]
class1=data[np.where(classes==1)]
class0_x,class0_y=class0[:,0],class0[:,1]
class1_x,class1_y=class1[:,0],class1[:,1]
centers_x,centers_y=clustercentres[:,0],clustercentres[:,1]
plt.figure(figsize=(8,4))
plt.scatter(class0_x, class0_y, label='$class0$', color='red', linewidth=3)
plt.scatter(class1_x, class1_y ,label='$class1$', linewidth=3)
plt.scatter(centers_x, centers_y ,label='$centers$',color='black', linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,1)
plt.ylim(0,1)
plt.legend()
plt.show()
plt.close('all')
#Split the data into a training set and a test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data,classes, test_size=0.2, random_state=42)
#Train a support vector machine (svm) classifier using the training data
#Use values of kernel='rbf', C=100.0 and gamma=1.0. [Hint: you will need to import something from sklearn]
from sklearn.svm import SVC
cls=SVC(kernel='rbf',C=100.0,gamma=1.0)
cls.fit(X_train,y_train)
#Calculate the fraction of training points correctly classified by your SVM
train_predict=cls.predict(X_train)
print 'the fraction of correct predicted points in train sets: ',cls.score(X_train,y_train)
#Calculate the fraction of test points correctly classified by your SVM
print 'the fraction of correct predicted points in test sets: ',cls.score(X_test,y_test)
#Obtain measures of performance for different values of gamma
gammas=[1e0,1e1,1e2,1e3,1e4,1e5]
train_score=[]
test_score=[]
for gamma in gammas:
    cls=SVC(kernel='rbf',C=100.0,gamma=gamma)
    cls.fit(X_train, y_train)
    train_score.append(cls.score(X_train,y_train))
    test_score.append(cls.score(X_test, y_test))
print train_score,test_score
#Produce a plot of the performance of the svm against the training and test data as a function of the value of  gamma
figure=plt.figure()
ax=figure.add_subplot(1,1,1)
ax.plot(gammas,train_score,label='Training score',marker='+')
ax.plot(gammas,test_score,label='Testing score',marker='o')
ax.set_title('SVM-gammas')
ax.set_xscale('log')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel('score')
ax.set_ylim(0,1.05)
ax.legend(loc='best',framealpha=0.5)
plt.show()
#Which value of gamma is optimal
'''
the value of gamma should be set to 1e0 or 1e1. when we train a classifier,we would like to
choose one performs well in both train and test dataset. the high performance in train-sets 
and low performance in test-sets resluts to overfitting. and an overfittinf classifier is 
hard to generalize. 

'''
#Fit your optimal svm and plot the decision boundary
cls=SVC(kernel='rbf',C=100.0,gamma=1.0)
cls.fit(data,classes)
h = .02  # step size in the mesh
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = cls.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
plt.scatter(class0_x, class0_y, label='$class0$', color='red', linewidth=3)
plt.scatter(class1_x, class1_y ,label='$class1$', linewidth=3)
plt.scatter(centers_x, centers_y ,label='$centers$',color='black', linewidth=3)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()