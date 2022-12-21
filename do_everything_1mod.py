import pandas as pd
import numpy as np

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

models = wraps.prep_cratio_models(df, 'Deck_A_Win?', cats, conts, 1, contTargetPts=10, edge=0.002)

#models = wraps.train_cratio_models(trainDf, 'Deck_A_Win?', 1000, [0.3], models, prints="verbose")
#print(models)

models = [{'BASE_VALUE': 1.0027725621633006, 'featcomb': 'mult', 'conts': {'Angels': [[0, 0.9070446865888991], [1, 1.0581141701470254], [2, 1.1135884891409038], [3, 1.2232000640229348], [7, 0.400744835583624]], 'Battalions': [[0, 1.0184199921748895], [1, 0.9984465660322378], [2, 0.9898750847312999], [3, 0.9965545718443076], [7, 0.9249940082858935]], 'Dragons': [[0, 0.9822768034891098], [1, 1.036259790549373], [2, 1.0372488199701946], [3, 1.0395823218408293], [7, 0.7320602861651603]], 'Emperors': [[0, 0.9703155541025104], [1, 1.0187145877106947], [2, 1.0816510390444498], [3, 1.0868075774263501], [7, 0.6409158267197335]], 'Guards': [[0, 0.9860371494437449], [1, 1.0284721918372306], [2, 1.04973393519919], [3, 1.0007478358717259], [7, 0.8652347340201542]], 'Hooligans': [[0, 0.9688774003194145], [1, 1.0565104870290958], [2, 1.0468822569063208], [3, 1.0267416668582765], [7, 0.7724538229385922]], 'Knights': [[0, 1.016681910414473], [1, 1.0298906891710542], [2, 0.9794947110892162], [3, 0.973291826834164], [7, 0.728269000546184]], 'Lotuses': [[0, 0.9102459151816232], [1, 1.0803146993191213], [2, 1.1350074302564144], [3, 1.1284227547747483], [7, 0.4719565119699651]], 'Minotaurs': [[0, 0.9916473457043048], [1, 1.0304209223890493], [2, 1.0248864769376702], [3, 1.0165186252631913], [7, 0.7989939373671913]], 'Pirates': [[0, 0.936696877267718], [1, 1.0538816494485028], [2, 1.1154244153040933], [3, 1.0784556375176164], [7, 0.7530375666456856]], 'Swords': [[0, 0.9914923104789036], [1, 1.0624046667287765], [2, 1.0320386989574877], [3, 0.97531213343662], [7, 0.42609260928118087]], 'Vigilantes': [[0, 0.9737087843415911], [1, 1.0106552191795382], [2, 1.0721814536849423], [3, 1.0651794523849538], [7, 0.8968710018521692]]}}]

testDf["Predicted"] = misc.predict(testDf, models, calculus.Cratio_mlink)
testDf["Actual"] = testDf["Deck_A_Win?"]



print("GINI: " + str(metrics.get_gini(testDf,"Predicted","Actual")))
p, a = metrics.get_Xiles(testDf,"Predicted","Actual")


print("DECILES")
print([util.round_to_sf(x) for x in p])
print([util.round_to_sf(x) for x in a])


wraps.viz_cratio_models(models, "1_model")

