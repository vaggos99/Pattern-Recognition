import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
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

def select_b365(conn):
    """
    Query all the rows for B365 
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT B365H, B365D, B365A FROM Match Where home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL AND B365H IS NOT NULL AND B365D IS NOT NULL AND B365A IS NOT NULL  ")

    odds = cur.fetchall()
    cur.execute("SELECT home_team_goal, away_team_goal FROM Match Where home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL AND B365H IS NOT NULL AND B365D IS NOT NULL AND B365A IS NOT NULL  ;")
    goals = cur.fetchall()
 
    return odds,goals

def select_IW(conn):
    """
    Query all the rows for IW 
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT IWH, IWD, IWA FROM Match Where home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL AND IWH IS NOT NULL AND IWD IS NOT NULL AND IWA IS NOT NULL  ")

    odds = cur.fetchall()
    cur.execute("SELECT home_team_goal, away_team_goal FROM Match Where home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL AND IWH IS NOT NULL AND IWD IS NOT NULL AND IWA IS NOT NULL  ;")
    goals = cur.fetchall()
 
    return odds,goals

def select_BW(conn):
    """
    Query all the rows for BW 
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT BWH, BWD, BWA FROM Match Where home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL AND BWH IS NOT NULL AND BWD IS NOT NULL AND BWA IS NOT NULL  ")

    odds = cur.fetchall()
    cur.execute("SELECT home_team_goal, away_team_goal FROM Match Where home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL AND BWH IS NOT NULL AND BWD IS NOT NULL AND BWA IS NOT NULL  ;")
    goals = cur.fetchall()
 
    return odds,goals
def select_LB(conn):
    """
        Query all the rows for LB 
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT  LBH,LBD,LBA FROM Match Where home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL AND LBH IS NOT NULL AND LBD IS NOT NULL AND LBA IS NOT NULL  ")

    odds = cur.fetchall()
    cur.execute("SELECT home_team_goal, away_team_goal FROM Match Where home_team_goal IS NOT NULL AND away_team_goal IS NOT NULL AND LBH IS NOT NULL AND LBD IS NOT NULL AND LBA IS NOT NULL  ;")
    goals = cur.fetchall()
 
    return odds,goals
def least_squares(x,goals):
        x=np.array(x)
        goals=np.array(goals)
        train_score=0
        error=0
        mean_success_rate=0
        mean_r2_score=0
        #10-fold cross validation
        kfold = KFold(10)
        for train_index, test_index in kfold.split(x):
            x_train, x_test=x[train_index],x[test_index]
            goals_train, goals_test=goals[train_index],goals[test_index]
            d_train=[]
            for goal in goals_train:
                if goal[0]>goal[1]:
                    d_train.append([1,0,0])
                elif goal[0]<goal[1]:
                    d_train.append([0,0,1])
                else :
                    d_train.append([0,1,0])
           
            d_test=[]
            for goal in goals_test:
                if goal[0]>goal[1]:
                    d_test.append([1,0,0])
                elif goal[0]<goal[1]:
                    d_test.append([0,0,1])
                else :
                    d_test.append([0,1,0])
                    
         
            d_test=np.array(d_test)

            
            # Create linear regression object
            regr = linear_model.LinearRegression()
            
            # Train the model using the training sets
            regr.fit(x_train, d_train)
            # Make predictions using the testing set
            y_pred = regr.predict(x_test)
            # the result is the index of the max value in the vector 
            exp_results=[]
            for test in d_test:
                exp_results.append(np.argmax(test, axis=0))
                
            pred_results=[]
            for y in y_pred:
                pred_results.append(np.argmax(y, axis=0))
                
                
                
            # The coefficients
            #print('Coefficients: \n', regr.coef_)
            # The mean squared error
            #print('Mean squared error: %.2f'% mean_squared_error(d_test, y_pred))
            # The coefficient of determination: 1 is perfect prediction
            #print('Coefficient of determination: %.2f'% r2_score(d_test, y_pred))
            train_score+=regr.score(x_train, d_train)

            success_rate=0
            for i in range(len(pred_results)):
                if pred_results[i]==exp_results[i]:
                    success_rate=success_rate+1
            success_rate=success_rate/len(pred_results)*100
            
            mean_success_rate +=success_rate
            mean_r2_score +=r2_score(d_test, y_pred)
            error+=mean_squared_error(d_test, y_pred)
            #print("Successfull predictions", success_rate,"% \n")
        train_score/=10
        mean_success_rate /=10
        mean_r2_score /=10
        error /=10
        print('Train score: ' ,train_score)
        print('Mean squared error: %.2f'% error)
        print('Coefficient of determination: %.2f'% mean_r2_score)
        print("Successfull predictions mean rate", mean_success_rate,"% \n")

def graph(array):
    array=np.array(array)
    x=array[:,0]
    y=array[:,1]
    z=array[:,2]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    pnt3d=ax.scatter(x,y,z,c=z)
    cbar=plt.colorbar(pnt3d)
    cbar.set_label("Values (units)")
    plt.show()
     
def main():
    np.set_printoptions(threshold=sys.maxsize)
    database = "database.sqlite"

    # create a database connection
    conn = create_connection(database)
    if conn:
        print("1. Computing for Β365 odds and results...")
        x,goals=select_b365(conn)
        graph(x)
        least_squares(x,goals)
        print("2. Computing for BW odds and results...")
        x,goals=select_BW(conn)
        graph(x)
        least_squares(x,goals)
        print("3. Computing for IW odds and results...")
        x,goals=select_IW(conn)
        graph(x)
        least_squares(x,goals)
        print("4. Computing for LB odds and results...")
        x,goals=select_LB(conn)
        graph(x)
        least_squares(x,goals)

if __name__ == '__main__':
    main()
