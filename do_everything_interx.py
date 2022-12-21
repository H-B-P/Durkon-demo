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

models[0] = prep.add_contcont_to_model(models[0], df, 'Pirates','Swords', contTargetPts1=10, edge1=0.002,contTargetPts2=10, edge2=0.002, replace=True)
models[0] = prep.add_contcont_to_model(models[0], df, 'Emperors','Lotuses',contTargetPts1=10, edge1=0.002,contTargetPts2=10, edge2=0.002, replace=True)

#models = wraps.train_cratio_models(trainDf, 'Deck_A_Win?', 1000, [0.3], models, prints="verbose")

models = [{'BASE_VALUE': 1.0027725621633006, 'featcomb': 'mult', 'conts': {'Angels': [[0, 0.911275458608], [1, 1.0648246648955624], [2, 1.118926231667518], [3, 1.2216388205547835], [7, 0.37460564688690284]], 'Battalions': [[0, 1.024096328479612], [1, 1.004260502495537], [2, 0.9943279412925194], [3, 0.9974697500154455], [7, 0.8817517209405422]], 'Dragons': [[0, 0.9872246719442789], [1, 1.0431281448766072], [2, 1.0406398195080273], [3, 1.0399317804216788], [7, 0.6941147654180095]], 'Guards': [[0, 0.9915512044378721], [1, 1.0338325197201101], [2, 1.0543174089267409], [3, 1.002231498435091], [7, 0.8248273852025277]], 'Hooligans': [[0, 0.9737217280534993], [1, 1.0624377669666105], [2, 1.0535075844586295], [3, 1.0262140174157022], [7, 0.7329721658648278]], 'Knights': [[0, 1.0219555494688404], [1, 1.036316234863753], [2, 0.9832403441770542], [3, 0.9725293963203314], [7, 0.6894320458039913]], 'Minotaurs': [[0, 0.9965089947992033], [1, 1.0368708319469342], [2, 1.0296488894263323], [3, 1.0171634057030536], [7, 0.7600833342768597]], 'Vigilantes': [[0, 0.978513138096482], [1, 1.0174622039522128], [2, 1.078170900217874], [3, 1.0649062224263863], [7, 0.8560096112406732]]}, 'contconts': {'Pirates X Swords': [[0, [[0, 1.0171905836950443], [1, 0.9690004861420108], [2, 0.8621654908152188], [3, 0.756515915855678], [7, 0.22693089525013355]]], [1, [[0, 1.0009699105064158], [1, 1.112771164982719], [2, 1.0879826416100276], [3, 1.0238439978019565], [7, 0.7366235498843865]]], [2, [[0, 0.976234986819078], [1, 1.1543218936079216], [2, 1.2999238603333452], [3, 1.4008549028888377], [7, 1.2343562908309855]]], [3, [[0, 0.8976641972714365], [1, 1.1611137630520139], [2, 1.359425973090799], [3, 1.558361243712887], [7, 1.5264564819720874]]], [7, [[0, 0.6009454709313534], [1, 0.8927590192396703], [2, 1.142614362246356], [3, 1.859020986176159], [7, 6.138130745244824]]]], 'Emperors X Lotuses': [[0, [[0, 0.9555163870679769], [1, 1.0154451755369724], [2, 0.9928936368511502], [3, 0.9678442383275163], [7, 0.2601507400918256]]], [1, [[0, 0.9020978400996463], [1, 1.0843924745855646], [2, 1.1755651751937142], [3, 1.10227815234559], [7, 0.7701221302166223]]], [2, [[0, 0.881177517425315], [1, 1.1831004991931044], [2, 1.3400063831304518], [3, 1.459569257404306], [7, 1.1849490852706444]]], [3, [[0, 0.8509747433297554], [1, 1.1939239430379762], [2, 1.444652899340143], [3, 1.5418215122070502], [7, 2.867461206662624]]], [7, [[0, 0.4371789870952982], [1, 0.8657738211332737], [2, 1.2133181113742622], [3, 1.547041104571734], [7, 3.2292429040793715]]]]}}]

print(models)

testDf["Predicted"] = misc.predict(testDf, models, calculus.Cratio_mlink)
testDf["Actual"] = testDf["Deck_A_Win?"]

print("GINI: " + str(metrics.get_gini(testDf,"Predicted","Actual")))
p, a = metrics.get_Xiles(testDf,"Predicted","Actual")


print("DECILES")
print([util.round_to_sf(x) for x in p])
print([util.round_to_sf(x) for x in a])


wraps.viz_cratio_models(models, "interx_model")

