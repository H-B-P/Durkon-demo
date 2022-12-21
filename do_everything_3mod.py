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

models = wraps.prep_cratio_models(df, 'Deck_A_Win?', cats, conts, 3, contTargetPts=10, edge=0.002)

#models = wraps.train_cratio_models(df, 'Deck_A_Win?', 50, [0.3, 0,0], models, prints="verbose")
#print(models)
#models = wraps.train_cratio_models(trainDf, 'Deck_A_Win?', 100, [0, 0.3,0], models, prints="verbose")
#print(models)
#models = wraps.train_cratio_models(trainDf, 'Deck_A_Win?', 150, [0, 0,0.3], models, prints="verbose")
#print(models)
#models = wraps.train_cratio_models(trainDf, 'Deck_A_Win?', 1000, [0.3, 0.3,0.3], models, prints="verbose")
#print(models)

models = [{'BASE_VALUE': 0.5013862810816503, 'featcomb': 'mult', 'conts': {'Angels': [[0, 0.8566845306339685], [1, 1.0460660828453552], [2, 1.1026638683283434], [3, 1.185688260343649], [7, 1.270732888443457]], 'Battalions': [[0, 1.056657634364542], [1, 0.9676337135597828], [2, 0.8946584833819948], [3, 0.8642116549979556], [7, 0.8857556666995149]], 'Dragons': [[0, 0.8736825886704804], [1, 1.010028497474458], [2, 1.1562623169537185], [3, 1.194492788356963], [7, 1.005855809762596]], 'Emperors': [[0, 0.8020354252001985], [1, 1.001466549803797], [2, 1.219599252339933], [3, 1.344695824653331], [7, 1.031912554385071]], 'Guards': [[0, 1.1358739471788988], [1, 0.9642597382059249], [2, 0.8045916326091148], [3, 0.5961279123767858], [7, 0.11890159430393918]], 'Hooligans': [[0, 1.0688718212548138], [1, 1.024180677175137], [2, 0.8696621390628827], [3, 0.7355439805016745], [7, 0.37547223346490255]], 'Knights': [[0, 1.0993956574527213], [1, 0.9925216578166856], [2, 0.8310604098137544], [3, 0.7002218055335818], [7, 0.26365139293219314]], 'Lotuses': [[0, 0.1], [1, 0.8750007212891608], [2, 1.787445663522918], [3, 2.4152914428138006], [7, 0.8010975538347461]], 'Minotaurs': [[0, 0.9547586321960302], [1, 1.0152769521780545], [2, 1.0282093286834952], [3, 1.0643205829512732], [7, 0.8551677517957792]], 'Pirates': [[0, 1.1016269361458066], [1, 0.9906257499721745], [2, 0.8451912007406932], [3, 0.6763248971341319], [7, 0.19520398561179444]], 'Swords': [[0, 1.1144696762428705], [1, 1.0349697997989789], [2, 0.8848374919764962], [3, 0.6537628921196101], [7, 0.1]], 'Vigilantes': [[0, 1.012521607937143], [1, 0.9813963139809851], [2, 1.0072089661358359], [3, 0.9300949506630837], [7, 0.7540961346409585]]}}, {'BASE_VALUE': 0.33425752072110015, 'featcomb': 'mult', 'conts': {'Angels': [[0, 1.002474793284652], [1, 1.0809984509303305], [2, 1.0702524829189974], [3, 1.1354387904386094], [7, 0.151426551643235]], 'Battalions': [[0, 1.2148877217585163], [1, 0.956153117784342], [2, 0.7908176025767706], [3, 0.6190262094449092], [7, 0.24149299401746363]], 'Dragons': [[0, 1.0347710442457398], [1, 1.0489377359927659], [2, 1.0145588357996362], [3, 1.0465085735995754], [7, 0.8947742956974166]], 'Emperors': [[0, 1.0964959494124495], [1, 1.0131243939639067], [2, 0.9336102458109667], [3, 0.9971307416112362], [7, 0.5488741376400936]], 'Guards': [[0, 1.065086250865433], [1, 1.0561216363532935], [2, 1.04035350791568], [3, 0.8652980314172274], [7, 0.7076253823875253]], 'Hooligans': [[0, 0.8157906926504751], [1, 1.0490400044668118], [2, 1.2371169057351399], [3, 1.4626920593729997], [7, 1.2719563729796524]], 'Knights': [[0, 1.1058472652397346], [1, 1.0754636921396978], [2, 0.9102331908596639], [3, 0.811269840831851], [7, 0.32012723874593263]], 'Lotuses': [[0, 1.2815771243853833], [1, 1.0090457361564784], [2, 0.5583256022594184], [3, 0.23849289067529872], [7, 0.1]], 'Minotaurs': [[0, 0.9737326581015163], [1, 1.072339660281252], [2, 1.1138608859509993], [3, 1.0647066534459275], [7, 1.1596270207202388]], 'Pirates': [[0, 0.6869027716128305], [1, 1.0633926354411105], [2, 1.412194490727617], [3, 1.5255529532908854], [7, 1.3234644137494624]], 'Swords': [[0, 0.517121241853042], [1, 1.0383350799108009], [2, 1.4270058719350494], [3, 1.90619141620682], [7, 0.8125371874173465]], 'Vigilantes': [[0, 1.0711697802556326], [1, 1.0545813993703361], [2, 1.0452706381265677], [3, 0.866501305520701], [7, 0.4325751958817395]]}}, {'BASE_VALUE': 0.16712876036055008, 'featcomb': 'mult', 'conts': {'Angels': [[0, 0.9165087605725241], [1, 1.1211300199890393], [2, 1.2665968902472238], [3, 1.4685511275911145], [7, 0.3146311388564101]], 'Battalions': [[0, 0.7784524659941277], [1, 1.0847870552753651], [2, 1.3253095886664723], [3, 1.5880115557061159], [7, 1.8348197511936117]], 'Dragons': [[0, 1.105257852375573], [1, 1.1190744718945798], [2, 1.044311735309171], [3, 0.9986717423157337], [7, 0.6310038383225789]], 'Emperors': [[0, 1.0536007301431203], [1, 1.0954112926711026], [2, 1.2065971652452987], [3, 1.0694096651278586], [7, 0.6971310587244811]], 'Guards': [[0, 0.7946817746789385], [1, 1.084126588878865], [2, 1.3232352730034145], [3, 1.583027118511733], [7, 1.741915262260946]], 'Hooligans': [[0, 1.1036123026655467], [1, 1.146479398629146], [2, 1.058749776899102], [3, 0.8931984246306178], [7, 0.7063675652408101]], 'Knights': [[0, 0.8884440208087603], [1, 1.060291884424536], [2, 1.2479401711726723], [3, 1.4732463172453278], [7, 1.712851708908205]], 'Lotuses': [[0, 1.2593356863382288], [1, 1.1646921288565097], [2, 0.7548098794913102], [3, 0.2313796925804882], [7, 0.1]], 'Minotaurs': [[0, 1.127276250211969], [1, 1.0832806578194942], [2, 1.014025400386246], [3, 1.024568483562993], [7, 0.6508393457862599]], 'Pirates': [[0, 1.1005527329805138], [1, 1.1234107744322972], [2, 1.0607840378522615], [3, 0.9828540910822654], [7, 0.6681309647401493]], 'Swords': [[0, 1.3909013160871193], [1, 1.0681387826697015], [2, 0.6593191592635058], [3, 0.1], [7, 0.1]], 'Vigilantes': [[0, 0.8903575830476689], [1, 1.0478615702567324], [2, 1.2366997029791968], [3, 1.4963442343732052], [7, 1.7545437622428113]]}}]

testDf["Predicted"] = misc.predict(testDf, models, calculus.Cratio_mlink)
testDf["Actual"] = testDf["Deck_A_Win?"]

print("GINI: " + str(metrics.get_gini(testDf,"Predicted","Actual")))
p, a = metrics.get_Xiles(testDf,"Predicted","Actual")


print("DECILES")
print([util.round_to_sf(x) for x in p])
print([util.round_to_sf(x) for x in a])


wraps.viz_cratio_models(models, "3_model")
