import pandas as pd
import copy
import numpy as np
import json


demographics = dict()
groups = []

def init():
    sex = ["male", "female", "unknown"]
    race = ["white", "black", "hispanic", "non-hispanic", "asian/pacific", "multiracial", "hawaiian", "native-american", "unknown"]
    for s in sex:
        for r in race:
            demographics[s + "-" + r] = len(demographics)
    groups = [i for i in range(len(demographics))]

def dataset1():
    df = pd.read_csv('datasets/shooting/NYPD_Shooting_Incident_Data__Historic_.csv')
    df["PERP_SEX"].replace({"M": "male", "F": "female", "U": "unknown"}, inplace=True)
    df["PERP_SEX"].replace(np.nan, "unknown", regex=True, inplace=True)
    df["VIC_SEX"].replace({"M": "male", "F": "female", "U": "unknown"}, inplace=True)
    df["VIC_SEX"].replace(np.nan, "unknown", regex=True, inplace=True)
    df["PERP_RACE"] = df["PERP_RACE"].str.lower().replace({"black hispanic": "black", "white hispanic": "hispanic"}, regex=True, inplace=True)
    df["PERP_RACE"].replace(np.nan, "unknown", regex=True, inplace=True)
    df["VIC_RACE"] = df["VIC_RACE"].str.lower().replace({"black hispanic": "black", "white hispanic": "hispanic"}, inplace=True)
    df["VIC_RACE"].replace(np.nan, "unknown", regex=True, inplace=True)
    df = df.rename(columns={"PERP_SEX": "officer_sex", "PERP_RACE": "officer_race", "VIC_SEX": "subject_sex", "VIC_RACE": "subject_race"})
    df = df.reset_index()
    df['id'] = df.index
    df.to_csv("datasets/clean_shooting/NYPD_Shooting_Incident_Data__Historic_.csv", index=False)

    # convert into algorithm input data
    dataset = dict()
    dataset["id"] = "NYPD_Shooting_Incident_Data__Historic_.csv" 
    dataset["groups"] = groups
    dataset["cost"] = 50
    tuples = [[row["id"], str(row["officer_sex"]) + "-" + str(row["officer_race"]), str(row["subject_sex"]) + "-" + str(row["subject_race"])] for index, row in df.iterrows()]         
    dataset["data"] = [[row[0], demographics[row[1]], demographics[row[2]]] for row in tuples] 
    return dataset 


def dataset2():
    df = pd.read_csv('datasets/shooting/SPD_Officer_Involved_Shooting__OIS__Data_Seattle.csv')
    df["Officer Gender"] = df["Officer Gender"].str.lower()
    df["Officer Race"] = df["Officer Race"].str.lower().replace({"hispanic/latio": "hispanic", "multi-racial": "multiracial", r"asian": "asian/pacific",r"black or african american": "black",  r"black": "black", "american indian/alaska native": "AI/AN", "two or more races": "multiracial", "none": "unknown", "not specified": "unknown", "nat hawaiian/oth pac islander": "national hawaiian/other pacific islander"}, regex=True, inplace=True)
    df["Officer Race"].replace(np.nan, "unknown", regex=True, inplace=True)
    df["Subject Gender"] = df["Subject Gender"].str.lower()
    df["Subject Race"] = df["Subject Race"].str.lower().replace({"hispanic/latio": "hispanic", "multi-racial": "multiracial", r"asian": "asian/pacific",r"black or african american": "black",  r"black": "black", "american indian/alaska native": "AI/AN", "two or more races": "multiracial", "none": "unknown", "not specified": "unknown", "nat hawaiian/oth pac islander": "national hawaiian/other pacific islander"}, regex=True, inplace=True)
    df["Subject Race"].replace(np.nan, "unknown", regex=True, inplace=True)
    df = df.rename(columns={"Officer Gender": "officer_sex", "Officer Race": "officer_race", "Subject Gender": "subject_sex", "Subject Race": "subject_race"})
    df = df.reset_index()
    df['id'] = df.index
    df.to_csv("datasets/clean_shooting/SPD_Officer_Involved_Shooting__OIS__Data_Seattle.csv", index=False)

    # convert into algorithm input data
    dataset = dict()
    dataset["id"] = "SPD_Officer_Involved_Shooting__OIS__Data_Seattle.csv"
    dataset["groups"] = groups
    dataset["cost"] = 50
    tuples = [[row["id"], str(row["officer_sex"]) + "-" + str(row["officer_race"]), str(row["subject_sex"]) + "-" + str(row["subject_race"])] for index, row in df.iterrows()]
    dataset["data"] = [[row[0], demographics[row[1]], demographics[row[2]]] for row in tuples] 
    return dataset 



ds = []
init()
d = dataset1()
d2 = dataset2()
ds.append(copy.deepcopy(d2))
for i in range(10):
    d["id"] = "test" + str(i)
    ds.append(copy.deepcopy(d))
json.dump(ds, open("datasets/clean_shooting/shooting_datasets.json", "w"))

