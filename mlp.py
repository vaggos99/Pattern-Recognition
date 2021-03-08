import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import KFold
import sqlite3
from sqlite3 import Error
import sys
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except Error as e:
        print(e)

    return conn

def makeDataset(conn):
    """
    Query all odds for every match from Match table an the attributes of the
    teams of the match
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT t.home_team_goal ,t.away_team_goal,t1.buildUpPlaySpeed, t1.buildUpPlayPassing, t1.chanceCreationPassing,  t1.chanceCreationCrossing,  t1.chanceCreationShooting,  t1.defencePressure,"+
    "t1.defenceAggression,  t1.defenceTeamWidth, t2.buildUpPlaySpeed, t2.buildUpPlayPassing, t2.chanceCreationPassing,  t2.chanceCreationCrossing,  t2.chanceCreationShooting,  t2.defencePressure,  t2.defenceAggression,"+
    "t2.defenceTeamWidth,t.B365H, t.B365D, t.B365A, t.BWH, t.BWD,  t.BWA, t.IWH, t.IWD,  t.IWA,  t.LBH, t.LBD,  t.LBA  FROM [Match] t INNER JOIN "+
"(SELECT *FROM Team_Attributes ta INNER JOIN (SELECT team_api_id , MAX(date) AS MaxDateTime FROM Team_Attributes GROUP BY team_api_id) groupedtt ON ta.team_api_id = groupedtt.team_api_id AND ta.date = groupedtt.MaxDateTime ) t1 "+ 
 " ON t1.team_api_id=t.home_team_api_id  INNER JOIN  (SELECT *FROM Team_Attributes ta INNER JOIN (SELECT team_api_id , MAX(date) AS MaxDateTime FROM Team_Attributes GROUP BY team_api_id) groupedtt "+
"ON ta.team_api_id = groupedtt.team_api_id AND ta.date = groupedtt.MaxDateTime ) t2  ON t2.team_api_id=t.away_team_api_id "
"WHERE home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL AND B365H IS NOT NULL AND B365D IS NOT NULL AND B365A IS NOT NULL AND IWH IS NOT NULL AND IWD IS NOT NULL AND IWA "+
"IS NOT NULL AND BWH IS NOT NULL AND BWD IS NOT NULL AND BWA IS NOT NULL AND LBH IS NOT NULL AND LBD IS NOT NULL AND LBA IS NOT NULL  group by t.id ")
    data = cur.fetchall()
 
    data=np.array(data)
    x=data[:,2:len(data[0])]
    y=data[:,0:2]
    return x,y
    
def multilayer_perceptron(x,goals):
    x=np.array(x)
    goals=np.array(goals)
    mean_taining_score=0
    mean_success_rate=0
    kfold = KFold(10)
    i=1
    #10-fold cross validation
    for train_index, test_index in kfold.split(x):
        x_train, x_test=x[train_index],x[test_index]
        goals_train, goals_test=goals[train_index],goals[test_index]
        d_train=[]
        #calculate the result from the score for training and testing data
        for goal in goals_train:
            if goal[0]>goal[1]:
                d_train.append('H')
            elif goal[0]<goal[1]:
                d_train.append('A')
            else :
                  d_train.append('D')
       
        d_test=[]
        for goal in goals_test:
            if goal[0]>goal[1]:
                d_test.append('H')
            elif goal[0]<goal[1]:
                d_test.append('A')
            else :
                d_test.append('D')

        #trasform the desired output to binary vector
        y_dense = LabelBinarizer().fit_transform(d_train)
        y_dense_test = LabelBinarizer().fit_transform(d_test) 
        #train mlp
        clf = MLPClassifier(solver='sgd', alpha=1e-5,max_iter=1000,hidden_layer_sizes=(80, 40, 20), random_state=1)
        clf.fit(x_train, y_dense)
        #add the result of training score
        mean_taining_score+=clf.score(x_train, y_dense)
        #test the mlp
        y_predict=clf.predict(x_test)
        #print(y_predict)
        #print(len(x_test)," ",len(y_dense_test))
        print(i,")Successfull predictions rate",clf.score(x_test, y_dense_test))
        mean_success_rate+=clf.score(x_test, y_dense_test)
        i+=1
    mean_taining_score/=10
    mean_success_rate/=10
    mean_success_rate *=100
    print("Mean training score:", mean_taining_score)
    print("Successfull predictions mean rate", mean_success_rate,"% \n")


def main():
    np.set_printoptions(threshold=sys.maxsize)
    database = "database.sqlite"

    # create a database connection
    conn = create_connection(database)
    if conn:
        print(" making Dataset...")
        x,goals=makeDataset(conn)
        print(" training and testing multilayer perceptron...")
        multilayer_perceptron(x,goals)

if __name__ == '__main__':
    main()
