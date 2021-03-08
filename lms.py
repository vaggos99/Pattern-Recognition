import sqlite3
from sqlite3 import Error
import padasip as pa
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
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
def lms_home(x,goals):
    x=np.array(x)
    goals=np.array(goals)
        
    d_train=[]
    for goal in goals:
        if goal[0]>goal[1]:
            d_train.append(1)
        else :
            d_train.append(-1)
       
    
    f_w = pa.filters.FilterNLMS(n=3, mu=0.1, w="random")
    y, e, w = f_w.run(d_train, x)
    #print("Weights for home win")
    #print(w[len(w)-1,:])
    return f_w
def lms_draw(x,goals):
    x=np.array(x)
    goals=np.array(goals)
  
        
    d_train=[]
    for goal in goals:
        if goal[0]==goal[1]:
            d_train.append(1)
        else :
            d_train.append(-1)
       
    
    f_d= pa.filters.FilterNLMS(n=3, mu=0.1, w="random")
    y, e, w = f_d.run(d_train, x)
    #print("Weights for draw ")
    #print(w[len(w)-1,:])
    return f_d

def lms_away(x,goals):
    x=np.array(x)
    goals=np.array(goals)

        
    d_train=[]
    for goal in goals:
        if goal[0]<goal[1]:
            d_train.append(1)
        else :
            d_train.append(-1)
       
    
    f_a = pa.filters.FilterNLMS(n=3, mu=0.1, w="random")
    y, e, w = f_a.run(d_train, x)
    #print("Weights for away win")
    #print(w[len(w)-1,:])
    return f_a



def lms_predict(f_w,f_d,f_a,x):
    x=np.array(x)
    y_pred=[]
    y_r=[]
    for x_test in x:
        r=[]
        y_w= f_w.predict(x_test)
        y_d= f_d.predict(x_test)
        y_a= f_a.predict(x_test)
        r.append(y_w)
        r.append(y_d)
        r.append(y_a)
        if max(r)==y_w:
            y_pred.append('H')
        elif max(r)==y_d:
            y_pred.append('D')
        elif max(r)==y_a:
            y_pred.append('A')
        
            
        y_r.append(r)

    return y_pred

def train_test(x,goals):
    mean_train_score=0
    mean_success_rate=0
    kfold = KFold(10)
    for train_index, test_index in kfold.split(x):
        x_train, x_test=x[train_index],x[test_index]
        goals_train, goals_test=goals[train_index],goals[test_index]

        f_w=lms_home(x_train,goals_train)
        f_d=lms_draw(x_train,goals_train)
        f_a=lms_away(x_train,goals_train)
        train_score=lms_predict(f_w,f_d,f_a,x_train)
        pred_results=lms_predict(f_w,f_d,f_a,x_test)

        exp_results_train=[]
        for goal in goals_train:
            if goal[0]>goal[1]:
                exp_results_train.append('H')
            elif goal[0]<goal[1]:
                exp_results_train.append('A')
            else:
                exp_results_train.append('D')

            
        exp_results=[]
        for goal in goals_test:
            if goal[0]>goal[1]:
                exp_results.append('H')
            elif goal[0]<goal[1]:
                exp_results.append('A')
            else:
                exp_results.append('D')
                
        success_rate=0
        for i in range(len(train_score)):
            if train_score[i]==exp_results_train[i]:
                success_rate=success_rate+1
        success_rate=success_rate/len(train_score)*100
        #print("Successfull predictions", success_rate,"% \n")
        mean_train_score +=success_rate  
         
        success_rate=0
        for i in range(len(pred_results)):
            if pred_results[i]==exp_results[i]:
                success_rate=success_rate+1
        success_rate=success_rate/len(pred_results)*100
        #print("Successfull predictions", success_rate,"% \n")
        mean_success_rate +=success_rate
    mean_train_score/=10       
    mean_success_rate /=10
    
    print("Mean training score:", mean_train_score,"%")
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
        x=np.array(x)
        graph(x)
        goals=np.array(goals)
        train_test(x,goals)

        print("2. Computing for BW odds and results...")
        x,goals=select_BW(conn)
        x=np.array(x)
        graph(x)
        goals=np.array(goals)
        train_test(x,goals)
        
        print("3. Computing for IW odds and results...")
        x,goals=select_IW(conn)
        x=np.array(x)
        graph(x)
        goals=np.array(goals)
        train_test(x,goals)
        
        print("4. Computing for LB odds and results...")
        x,goals=select_LB(conn)
        x=np.array(x)
        graph(x)
        goals=np.array(goals)
        train_test(x,goals)
        
if __name__ == '__main__':
    main()
