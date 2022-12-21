import pandas as pd
import numpy as np

import xgboost as xgb

import actual_modelling
import prep
import misc
import calculus

import wraps

import util
import metrics

df = pd.read_csv("data.csv")

df = df.rename(columns = {'Alessin_Adamant_Angel_Deck_A_Count':"Angels",
       'Bold_Battalion_Deck_A_Count':"Battalions",
       'Dreadwing_Darkfire_Dragon_Deck_A_Count':"Dragons",
       'Evil_Emperor_Eschatonus_Empyreal_Envoy_of_Entropic_End_Deck_A_Count':"Emperors",
       'Gentle_Guard_Deck_A_Count':"Guards",
       'Horrible_Hooligan_Deck_A_Count':"Hooligans",
       'Kindly_Knight_Deck_A_Count':"Knights",
       'Lilac_Lotus_Deck_A_Count':"Lotuses",
       'Murderous_Minotaur_Deck_A_Count':"Minotaurs",
       'Patchy_Pirate_Deck_A_Count':"Pirates",
       'Sword_of_Shadows_Deck_A_Count':"Swords",
       'Virtuous_Vigilante_Deck_A_Count':"Vigilantes"})

trainDf = df[:250000].reset_index(drop=True)
valDf = df[250000:300000].reset_index(drop=True)
testDf = df[300000:].reset_index(drop=True)

print(trainDf)
print(valDf)
print(testDf)

cats=[]
conts=["Angels",
       "Battalions",
       "Dragons",
       "Emperors",
       "Guards",
       "Hooligans",
       "Knights",
       "Lotuses",
       "Minotaurs",
       "Pirates",
       "Swords",
       "Vigilantes"]

dtrain = xgb.DMatrix(trainDf[cats+conts], label=trainDf["Deck_A_Win?"])
dtest = xgb.DMatrix(testDf[cats+conts], label=testDf["Deck_A_Win?"])

bst = xgb.train({"max_depth":3, 'objective':'reg:logistic'}, dtrain, 1000)

testDf["Predicted"] = bst.predict(dtest)
testDf["Actual"] = testDf["Deck_A_Win?"]

print("GINI: " + str(metrics.get_gini(testDf,"Predicted","Actual")))
p, a = metrics.get_Xiles(testDf,"Predicted","Actual")


print("DECILES")
print([util.round_to_sf(x) for x in p])
print([util.round_to_sf(x) for x in a])

