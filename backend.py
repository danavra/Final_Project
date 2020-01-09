import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.externals import joblib
from matplotlib import pyplot
from sklearn.utils import shuffle


def preprocessAnsersLabels(df,labels):
    columnsNamesArr = df.columns.values
    listOfColumnNames = list(columnsNamesArr)
    strLabels=list()
    for i in labels:
        strLabels.append(listOfColumnNames[i])
    if('%' in df[df['Answer'][0]][0]):
        precentageDict = {
            "0-10%": 0.1,
            "11-20%": 0.2,
            "21-30%": 0.3,
            "31-40%": 0.4,
            "41-50%": 0.5,
            "51-60%": 0.6,
            "61-70%": 0.7,
            "71-80%": 0.8,
            "81-90%": 0.9,
            "91-100%": 1.0,
        }
        for label in strLabels:
            df[label]=df[label].apply(lambda x:precentageDict[x])
    return df


def dropAnswersLabels(df, labels):
    columnsNamesArr = df.columns.values
    listOfColumnNames = list(columnsNamesArr)
    strLabels = list()
    for i in labels:
        strLabels.append(listOfColumnNames[i])
    for label in strLabels:
        df=df.drop([label],axis=1)
    return df

def preProcessData(df,labels):
    numOfAnswers = df.groupby(['Answer']).size()
    totalNumOfAnswer = len(df)
    df['same_answer_actual'] = df['Answer'].apply(lambda x: numOfAnswers[x] / totalNumOfAnswer)
    df=preprocessAnsersLabels(df,labels)
    df['estimated_ability_myanswer'] = df.apply(lambda x: x['same_answer_actual'] - float(x[x['Answer']]), axis=1)
    df['estimated_ability_allanswers'] = df.apply(lambda x: calc_estimated_ability_allanswers(x,df,numOfAnswers,totalNumOfAnswer), axis=1)
    df['arrogance'] = df.apply(lambda x: ((x['Confidence'] - 1) / 10) / (9 * x[x['Answer']]), axis=1)
    df['estimation_compare'] = df.apply(lambda x: x[x['Answer']] - (df[x['Answer']].sum() / totalNumOfAnswer), axis=1)
    df=dropAnswersLabels(df,labels)
    return df

def calc_estimated_ability_allanswers(row,df,numOfAnswers,totalNumOfAnswer):
    estimated_ability_allanswers=0
    for answer in df.Answer.unique():
        estimated_ability_allanswers+=abs(row[answer]-numOfAnswers[answer]/totalNumOfAnswer)
    return estimated_ability_allanswers/len(df.Answer.unique())
def buildModel(df):
    X = df.drop(['index','Class', 'same_answer_actual','Answer'], axis=1)
    y = df['Class'].apply(lambda x: True if x == 'Solver' else False)
    model = RandomForestClassifier(n_estimators=200, random_state=5,class_weight={True:3,False:1})
    model.fit(X, y)
    return model
def getFeatureImportance(model,X_data):
    importances = list(model.feature_importances_)
    features = list(X_data.columns)
    feature_importance = [(feature, round(importance, 2)) for feature, importance in zip(features, importances)]
    feature_importance = sorted(feature_importance, key=lambda x: x[1])
    return feature_importance
def predictWithModel(df,model):
    X = df.drop(['index','Class', 'same_answer_actual','Answer'], axis=1)
    y = df['Class'].apply(lambda x: True if x == 'Solver' else False)
    pred = model.predict(X)
    predict_proba = model.predict_proba(X)
    # score = metrics.accuracy_score(y, pred)
    # print("accuracy:   %0.3f" % score)
    # matrix = metrics.confusion_matrix(y, pred)
    # print("confusion matrix:")
    # print(matrix)
    return pred,predict_proba
def TestAgainstMajorityRule(pred_proba,df):
    majority=df.Answer.mode()[0]
    maxpred = 0
    i=0
    index=0
    for prob in pred_proba:
        if (maxpred < prob[1]):
            maxpred = prob[1]
            index = i
        i += 1
    modelAnswer=df.loc[ index , : ]
    return majority,modelAnswer['Answer']
def checkPredictionsValues(numOfGroups,file,labels,modelPath):
    df = pd.read_csv(file, index_col=None)
    df = preProcessData(df, labels)
    #df = shuffle(df)
    sizeOfEachGroup=int(len(df)/numOfGroups)+1
    answersList=list()
    actualAnswer = df[df['Class'] == 'Solver'].Answer.unique()[0]
    for i in range(0,numOfGroups):
        if(i==0):
            part_df=df[:sizeOfEachGroup]
        elif(i==numOfGroups-1):
            part_df = df[i*sizeOfEachGroup:]
        else:
            part_df = df[i*sizeOfEachGroup:(i+1)*sizeOfEachGroup]
        part_df=part_df.reset_index()
        #part_df=preProcessData(part_df,labels)
        loaded_model = joblib.load(modelPath)
        pred,pred_proba=predictWithModel(part_df,loaded_model)
        majority, modelAnswer=TestAgainstMajorityRule(pred_proba,part_df)
        answerDict={
            "groupID":i+1,
            "majority":majority,
            "model":modelAnswer,
            "ground_true":actualAnswer
        }
        answersList.append(answerDict)
    return answersList


def seperateToGroups(numOfGroups, df,labels):
    sizeOfEachGroup = int(len(df) / numOfGroups) + 1
    listDf=list()
    for i in range(0, numOfGroups):
        if (i == 0):
            part_df = df[:sizeOfEachGroup]
        elif (i == numOfGroups - 1):
            part_df = df[i * sizeOfEachGroup:]
        else:
            part_df = df[i * sizeOfEachGroup:(i + 1) * sizeOfEachGroup]
        part_df = part_df.reset_index()
        part_df=preProcessData(part_df,labels)
        listDf.append(part_df)
    return pd.concat(listDf,ignore_index=True,axis=0)

def createModel(file,pathToModel,modelName,labels):
    df = pd.read_csv(file, index_col=None)
    #df = seperateToGroups(20, df,labels)
    secondProb=df[400:]
    df=preProcessData(df[:400],labels)
    model=buildModel(df)
    joblib.dump(model, pathToModel+'/'+modelName)
    return secondProb


if __name__ == '__main__':
    #part_df=createModel('cr.csv','/Users/itzikvais/PycharmProjects/final_project','rf3Model.sav',[3,4,5,6,7,8,9,10])
    answers=checkPredictionsValues(3,'puzzle.csv',[2,3,4,5,6,7],'rfModel.sav')
    major=0
    model=0
    for answer in answers:
        print(answer)
        if(answer['majority']==answer['ground_true']):
            major+=1
        if (answer['model'] == answer['ground_true']):
            model+=1
    x=['majority','our model']
    y=[major/5,model/5]
    pyplot.bar(x,height=y)
    pyplot.show()






