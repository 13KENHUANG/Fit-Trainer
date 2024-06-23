import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression,RidgeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
df=pd.read_csv('coords.csv')
#----------------------#
classes = ['biceps', 'Shoulder']

def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
#------------------------#
#print(df.tail())

x=df.drop('class',axis=1) #feature
y=df['class'] #target
#print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1234)

#print(y_train)

pipelines={
    'lr':make_pipeline(StandardScaler(),LogisticRegression()),
    'rc':make_pipeline(StandardScaler(),RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(),RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(),GradientBoostingClassifier()),
}

#print(list(pipelines.values())[0])

fit_models={}
for algo,pipeline in pipelines.items():
    model=pipeline.fit(x_train.values,y_train)
    fit_models[algo]=model

for algo,model in fit_models.items():
    yhat=model.predict(x_test.values)
    print(confusion_matrix(y_test,yhat))
    cm=confusion_matrix(y_test,yhat)
    plot_confusion_matrix(cm,'cm.png',title='confusion matrix')
    print(algo,accuracy_score(y_test,yhat))


#print(fit_models)
with open('body_language.pkl','wb') as f:
    pickle.dump(fit_models['rf'],f)




# print(fit_models['rf'].predict(x_test))
# print(y_test)