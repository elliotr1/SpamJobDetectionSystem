import random
import multiprocessing as mp
import pandas as pd
from Models.Naive_Bayes.Bernoulli.Model_Builder.model import naive_bayes_mixed_data, naive_bayes_func
import json
import mysql.connector

from Models.Naive_Bayes.Bernoulli.Testing.testing import spamVariance


class CreateSharedColumns:
    _column_names: list
    _list_of_dfs: list

    def __init__(self, column_names: list, list_of_dfs: list):
        self._column_names = column_names
        self._list_of_dfs = list_of_dfs

    def returnReducedDatasets(self):
        return [df[self._column_names].dropna() for df in self._list_of_dfs]



seen_spam_and_ham_dataset = pd.read_csv("./Datasets/newActiveLearningModel.csv")
dataset = seen_spam_and_ham_dataset.dropna()
file = open("config.json", "r")
jsonObj = json.load(file)
con = mysql.connector.connect(
    database=jsonObj["SQLSettings"]["database"],
    host=jsonObj["SQLSettings"]["host"],
    user=jsonObj["SQLSettings"]["user"],
    password=jsonObj["SQLSettings"]["password"]
)
curs = con.cursor()
curs.execute("SELECT title, description, requirements, company_profile FROM jobs WHERE length(company_profile) > 0")
results = curs.fetchall()
unseen_data = pd.DataFrame(columns=["title", "description", "requirements", "company_profile"])
for row in results:
    title, description, requirements, company_profile = row
    unseen_data = unseen_data._append(
        {"title": title, "description": description, "requirements": requirements, "company_profile": company_profile},
        ignore_index=True)
con.close()
curs.close()

# scrapReedJobs()

df1 = pd.read_csv("Datasets/data job posts.csv").dropna()
columns = dataset.columns.tolist()
columns.remove("fraudulent")

sharedColumnsObj = CreateSharedColumns(column_names=columns, list_of_dfs=[df1])
unseen_dfs = sharedColumnsObj.returnReducedDatasets()
# recursiveNN(dataset)
unseen_datasets_being_used = [{"table_name": "reed scraped data", "dataset": unseen_dfs[0]}]
# {"table_name": "Datasets/data job posts.csv", "dataset": unseen_dfs[1]},
# {"table_name": "Datasets/Jobs.csv", "dataset": unseen_dfs[0]}]




def NaiveBayesModel():
    print(len(seen_spam_and_ham_dataset))
    for i in range(10):
         naive_bayes_mixed_data(dataset,unseen_datasets_being_used,  .999)
         print(spamVariance)
if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=2)
    p1 = mp.apply_async(target=scrapReedJobs, args=None)
    p2 = mp.apply_async(target=NaiveBayesModel, args=None)
    pool.close()
    pool.join()
    print("Process has ended")
