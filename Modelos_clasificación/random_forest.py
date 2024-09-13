from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

df = pd.read_csv('df_output\datos_codifcadobalanceados.csv')
variables_principales= ['INSE','EDAD'
                        ,'ESTU_DEDICACIONLECTURADIARIA','ESTU_DEDICACIONINTERNET','COLE_JORNADA','COLE_COD_DANE_ESTABLECIMIENTO']
df_y = df[['PUNT_GLOBAL']]
def y_cambio(i):
    if i<=279:
        return 0
    elif i >=280 and i <= 359:
        return 1
 
    
    elif i >=360:
        return 2

df_y['PUNT_GLOBAL'] = df_y['PUNT_GLOBAL'].apply(y_cambio)
Y= df_y['PUNT_GLOBAL']
X = df[variables_principales]

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X,Y,test_size=0.2,random_state=2)

ran_forest = RandomForestClassifier(n_estimators=19,random_state=2016,min_samples_leaf=6,)

ran_forest.fit(Xtrain,Ytrain)

preds = ran_forest.predict(Xtest)
print(accuracy_score(Ytest, preds))
print('********************+')
print(ran_forest.score(Xtest,Ytest))

# con random forest genera un accuracy de 0.7191