# -*- coding: utf-8 -*-
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
'''
VARIABLE DESCRIPTIONS:
survival        Survival
                (0 = No; 1 = Yes)
pclass          Passenger Class
                (1 = 1st; 2 = 2nd; 3 = 3rd)
name            Name
sex             Sex
age             Age
sibsp           Number of Siblings/Spouses Aboard
parch           Number of Parents/Children Aboard
ticket          Ticket Number
fare            Passenger Fare
cabin           Cabin
embarked        Port of Embarkation
                (C = Cherbourg; Q = Queenstown; S = Southampton)
'''

fname = 'C:/Users/YJ/Documents/1) Learning/Udacity - Data Analyst/Submissions/003/titanic_data.csv'
#fname = 'C:/Users/jenni/Documents/titanic_data.csv'
fhand = pd.read_csv(fname)
print fhand.count()

'Probability of survival of a passenger given his personal details'

def sigmoid(z):
    return 1/(1+np.exp(-z))
def norm(myseries):
    return (myseries -myseries.mean())/myseries.std(ddof=0)
#insert an random normal distributed variable for each NaN value
def agerand(x):
    if math.isnan(x):
        x=np.random.normal(0,age.std(ddof=0))
    return x

#data cleaning.  creating new binary variables as well as normalising the continuous variables 
survived = fhand.Survived
pclass1 = fhand.Pclass ==1
pclass2 = fhand.Pclass ==2
pclass3 = fhand.Pclass ==3
sex = fhand.Sex=='male'
age = norm(fhand.Age)
sibsp = norm(fhand.SibSp)
parch = norm(fhand.Parch)
embarkedC = fhand.Embarked=='C'
embarkedQ = fhand.Embarked=='Q'
embarkedS = fhand.Embarked=='S'
fare = norm(fhand.Fare)

#plot original normalised age graph 
fig = plt.figure()
DirtyHist = fig.add_subplot(2,1,1)
age.hist(bins = 20)
DirtyHist.set_title('Histogram of Age of Titanic Passengers')
DirtyHist.set_xlabel('Normalised Age')
DirtyHist.set_ylabel('Frequency')
DirtyHist.grid(1)
#against new normalised age graph and see if they look very different
plt.subplots_adjust(hspace = 0.4)
CleanHist = fig.add_subplot(2,1,2)
age = age.apply(agerand)
CleanHist.hist(age,bins = 20)
CleanHist.set_title('Histogram of Clean Age of Titanic Passengers')
CleanHist.set_xlabel('Normalised Age')
CleanHist.set_ylabel('Frequency')
CleanHist.grid(1)

variablenames= ['pclass1','pclass2','pclass3','sex','age','sibsp','parch','embarkedC','embarkedQ','embarkedS','fare']

# put all data back into DataFrame and remove 2 NaNs from embarked which reduce the data from 891 to 889
X = np.array([survived,pclass1,pclass2,pclass3,sex,age,sibsp,parch,embarkedC,embarkedQ,embarkedS,fare]).transpose()
X = pd.DataFrame(X,columns = ['survived','pclass1','pclass2','pclass3','sex','age','sibsp','parch','embarkedC','embarkedQ','embarkedS','fare'])
Details_of_Data= X.describe()

#-----------------------------------------------------------------------------------------------------------------------------------------
class myplot():
    
    def mypie(self,data,mylabels,mytitle,autopct):
        fig = plt.figure()
        newsub = fig.add_subplot(1,1,1)
        newsub.pie(data,labels=mylabels,autopct = autopct)
        newsub.set_title(mytitle)

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

#plot passenger class pie chart
a=myplot()
data = [pclass1.sum(),pclass2.sum(),pclass3.sum()]
mylabels = ['First Class','Second Class', 'Third Class']
title = 'Pie chart showing split of passenger tickets'
a.mypie(data,mylabels,title,make_autopct(data))

#plot port of embarkation pie chart
b=myplot()
data = [embarkedC.sum(),embarkedQ.sum(),embarkedS.sum()]
mylabels = ['Cherbourg','Queenstown', 'Southampton']
title = '''Pie chart showing split passenger's port of embarkation'''
b.mypie(data,mylabels,title,make_autopct(data))

#Normalised Age of Titanic Passengers
fig1 = plt.figure()
CleanHist = fig1.add_subplot(1,1,1)
CleanHist.hist(age,bins = 20)
CleanHist.set_title('Histogram of Normalised Age of Titanic Passengers')
CleanHist.set_xlabel('Normalised Age')
CleanHist.set_ylabel('Frequency')

#Fare paid by passengers
fig2 = plt.figure()
fareplot = fig2.add_subplot(1,1,1)
fareplot.hist(fhand.Fare,bins = 50)
fareplot.set_title('Histogram of Fare paid by Titanic Passengers')
fareplot.set_xlabel('Fare')
fareplot.set_ylabel('Frequency')


#Pie chart of survivors split by class
c=myplot()
data = [pclass1.where(survived==1).sum(),pclass2.where(survived==1).sum(),pclass3.where(survived==1).sum()]
mylabels = ['First Class','Second Class', 'Third Class']
title = 'Pie chart showing survivors split by class'
c.mypie(data,mylabels,title,make_autopct(data))

#Pie chart of survivors split by sex
c=myplot()
data = [sex.where(survived==1).sum(),sex.where(survived==0).sum()]
mylabels = ['Male','Female']
title = 'Pie chart showing survivors split by class'
c.mypie(data,mylabels,title,make_autopct(data))

horiz = fhand.Sex.where(fhand.Survived==1)
horiz = horiz.replace(1,'Male').replace(0,'Female')
df = pd.crosstab(fhand.Pclass,horiz)/(survived==1).sum()
df.plot(kind='bar',title = 'Ratio of survivors in each class according to sex')

#-----------------------------------------------------------------------------------------------------------------------------------------


#Training set
Xtrain = np.matrix(X.ix[:int(len(X)*0.6),1:])
ytrain = np.matrix(X.ix[:int(len(X)*0.6),0]).transpose()

#Cross validation to find the best lambda and alpha
Xcross = np.matrix(X.ix[int(len(X)*0.6):int(len(X)*0.8),1:])
ycross = np.matrix(X.ix[int(len(X)*0.6):int(len(X)*0.8),0]).transpose()

#Test set to see performance of theta
Xtest = np.matrix(X.ix[int(len(X)*0.8):,1:])
ytest = np.matrix(X.ix[int(len(X)*0.8):,0]).transpose()

#cost function of logistic regression
def cost(X1,y1,theta,lambda1):
    m = len(X1)
    h = sigmoid(X1.dot(theta))
    reg = lambda1 * 1 / 2 / m * theta.transpose().dot(theta)
    return sum((np.multiply(y1,log(h))+ np.multiply(1-y1,log(1-h)))/-m) + sum(reg)

#gradient descent of each step
def gradient_descent(X1,y1,theta,alpha,lambda1):
    m = len(X1)
    h = sigmoid(X1.dot(theta))
    reg = lambda1 / m * theta
    theta = theta - alpha * (1./m * X1.transpose().dot((h-y1)) + reg)
    return theta

#random initialisation of theta.
def randtheta(X):
    return np.matrix(np.zeros(X.shape[1]).reshape(X.shape[1],1))

def runs(iterations,Xtrain,ytrain,Xcross,ycross,theta,alpha,lambda1,lambda2):
    train = [None] * iterations
    cross = [None] * iterations
    for i in xrange(iterations):
        train[i] = cost(Xtrain,ytrain,theta,lambda1)
        cross[i] = cost(Xcross,ycross,theta,lambda2)
        theta = gradient_descent(Xtrain, ytrain, theta,alpha,lambda1)    
    return theta,train[iterations-1],cross[iterations-1]   


#finding the ideal alpha
alphalist = [i/10. for i in xrange(8,15)]
alphaerror = []
theta = randtheta(Xtrain)
lambda1 = 0
lambda2 = 0
for i in alphalist:
    alphaerror.append(runs(100,Xtrain,ytrain,Xcross,ycross,theta,i,lambda1,lambda2)[2])
alpha = alphalist[np.array(alphaerror).argmin()]
print('The optimal alpha  is %f ') %alpha  

#Finding the ideal lambda using the test set and checking it against the cross validation set
lambdalist = [i/10. for i in xrange(1,10)]
lambdaerror = []
lambda2=0
for i in lambdalist:
    lambdaerror.append(runs(100,Xtrain,ytrain,Xcross,ycross,theta,alpha,i,lambda2)[2])
#
lambda1 = lambdalist[np.array(lambdaerror).argmin()]
print('The optimal lambda  is %f ') %lambda1 

#initialising graphs
fig = plt.figure()
plotalpha = fig.add_subplot(2,1,1)
plotlambda = fig.add_subplot(2,1,2)
fig.subplots_adjust(hspace=0.4)
plotalpha.plot(alphalist,alphaerror)
plotalpha.set_title('Alpha that minimises Cross Validation Error')
plotalpha.set_xlabel('Alpha')
plotalpha.set_ylabel('Cross Validation Error')
plotalpha.annotate('Chosen Alpha',
                    xy = (alpha,min(alphaerror)),
                    xytext = (alpha,min(alphaerror)+0.0002),
                    arrowprops=dict(facecolor='black') )

plotlambda.plot(lambdalist,lambdaerror)
plotlambda.set_title('Lambda that minimises Cross Validation Error')
plotlambda.set_xlabel('Lambda')
plotlambda.set_ylabel('Cross Validation Error')
plotlambda.annotate('Chosen Lambda',
                    xy = (lambda1,min(lambdaerror)),
                    xytext = (lambda1,min(lambdaerror)+0.00001),
                    arrowprops=dict(facecolor='black') )

#looking for optimal theta 
theta=(runs(10000,Xtrain,ytrain,Xcross,ycross,theta,alpha,lambda1,lambda2))[0]
prediction = np.array(sigmoid(Xtest.dot(theta))>=0.5 )
answer = np.array(ytest)

accuracy = (answer==prediction).sum()/float(len(prediction))
print('''The model has a {0:.2f}% of predicting whether a passenger will live or die''').format(accuracy * 100)

theta = pd.DataFrame(theta,index = variablenames,columns = ['Theta'])
print theta

def genMeanProb(X,theta,cat,value):
    df = pd.DataFrame(X, columns = variablenames)
    if cat == 'pclass1':
        df[cat]=value
        df.pclass2 = 0
        df.pclass3 = 0
    elif cat == 'pclass2':
        df[cat]=value
        df.pclass1 = 0
        df.pclass3 = 0
    elif cat == 'pclass3':
        df[cat]=value
        df.pclass1 = 0
        df.pclass2 = 0
    elif cat == 'embarkedC':
        df[cat] = value
        df.embarkedQ=0
        df.embarkedS=0
    elif cat == 'embarkedQ':
        df[cat] = value
        df.embarkedC=0
        df.embarkedS=0
    elif cat == 'embarkedS':
        df[cat] = value
        df.embarkedQ=0
        df.embarkedC=0
    else:
        df[cat] = value
    test = np.matrix(df)
    prediction = sigmoid(test.dot(theta))
    return prediction.mean()
    
print 'Predicted probability of a male surviving a marine disaster : {0:.2f}%'.format(genMeanProb(X,theta,'sex',1)*100)
print 'Predicted probability of a female surviving a marine disaster : {0:.2f}%'.format(genMeanProb(X,theta,'sex',0)*100)
print 'Predicted probability of a first class passenger surviving a marine disaster  : {0:.2f}%'.format(genMeanProb(X,theta,'pclass1',1)*100)
print 'Predicted probability of a second class passenger surviving a marine disaster : {0:.2f}%'.format(genMeanProb(X,theta,'pclass2',1)*100)
print 'Predicted probability of a third class passenger surviving a marine disaster  : {0:.2f}%'.format(genMeanProb(X,theta,'pclass3',1)*100)
print 'Predicted probability of a passenger boarding from Chebourg and surviving a marine disaster  : {0:.2f}%'.format(genMeanProb(X,theta,'embarkedC',1)*100)
print 'Predicted probability of a passenger boarding from Queenstown and surviving a marine disaster  : {0:.2f}%'.format(genMeanProb(X,theta,'embarkedQ',1)*100)
print 'Predicted probability of a passenger boarding from Southampton and surviving a marine disaster  : {0:.2f}%'.format(genMeanProb(X,theta,'embarkedS',1)*100)

if True:
    plt.close('all')
    
print X.groupby(['sex'])['survived'].mean()
print X.groupby(['sex'])['survived'].count()
print X.groupby(['pclass1','pclass2','pclass3','sex'])['survived'].count()
print X.groupby(['pclass1','pclass2','pclass3','sex'])['survived'].mean()*100

def ttest(X,x,y):
    #test if mean of x==1 is significiantly different from mean of x==0 for y
    mean_xy=X.groupby([x])[y].mean()
    count_xy = X.groupby([x])[y].apply(len)
    std_xy=X.groupby([x])[y].std(ddof=0)
    tstatistics = (mean_xy[1]-mean_xy[0])/std_xy[0]/std_xy[1]
    ddof = count_xy[0]-1+count_xy[1]-1
    return [tstatistics,ddof]

[tstat,ddof] = ttest(X,'sex','survived')
print ('''Null hypothesis states that the mean probability of survival of a male is the same as the probability of survival of a female passenger \n H0 : µ1 = 	µ0''')
print ('''Alternate hypothesis states that the mean probability of survival of a male is different from the probability of a female passenger \n H0 : µ1 =/= µ0''') 
print '''The t statistic of the difference in mean probability of male survivors and female survivors is %f with a degree of freedom of %f''' %(tstat,ddof)
print '''As tstat is less than that of tcritical (99.5 percent with degree of freedom of 1000 = -2.813), \n
thus we will reject the null hypothesis and can conclude tha the probability of survival for a male passenger is significiantly lower than that of a female'''

print X.groupby(['sex'])['survived'].mean()
print X.groupby(['pclass1','pclass2','pclass3'])['survived'].mean()
print X.groupby(['survived'])['age'].mean()

