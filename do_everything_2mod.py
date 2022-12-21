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

models = wraps.prep_cratio_models(df, 'Deck_A_Win?', cats, conts, 2, contTargetPts=10, edge=0.002)

#models = wraps.train_cratio_models(df, 'Deck_A_Win?', 100, [0.3, 0], models, prints="verbose")
#print(models)
#models = wraps.train_cratio_models(trainDf, 'Deck_A_Win?', 100, [0, 0.3], models, prints="verbose")
#print(models)
#models = wraps.train_cratio_models(trainDf, 'Deck_A_Win?', 1000, [0.3, 0.3], models, prints="verbose")
#print(models)

models = [{'BASE_VALUE': 0.6685150414422003, 'featcomb': 'mult', 'conts': {'Angels': [[0, 0.8524166992638966], [1, 1.056072346200526], [2, 1.1757452372236539], [3, 1.3317438937821244], [7, 0.605132393679308]], 'Battalions': [[0, 0.9074881352605864], [1, 1.0062714241930393], [2, 1.1001174487416099], [3, 1.2201941823422842], [7, 1.3241757356954014]], 'Dragons': [[0, 0.972197375283779], [1, 1.0415698409328136], [2, 1.0530701201546877], [3, 1.0585753315903297], [7, 0.8062792372281614]], 'Emperors': [[0, 0.9058698299855423], [1, 1.029021609413727], [2, 1.1614001150212727], [3, 1.1934955617395457], [7, 0.8589384971824693]], 'Guards': [[0, 0.9371478606367379], [1, 1.0328195734436931], [2, 1.0984185977851602], [3, 1.120737818787692], [7, 1.0332106188031265]], 'Hooligans': [[0, 1.0367838509117662], [1, 1.0612479591923136], [2, 0.9568936528226617], [3, 0.8504672380113516], [7, 0.6205531157126367]], 'Knights': [[0, 0.9561923793672202], [1, 1.0185163856792003], [2, 1.0571902145459628], [3, 1.1353537272777965], [7, 1.020563169577311]], 'Lotuses': [[0, 0.7982217676812082], [1, 1.0972451711890925], [2, 1.2633767212339249], [3, 1.2923817377332179], [7, 0.6329480476900181]], 'Minotaurs': [[0, 1.0136391667808362], [1, 1.0150420906555861], [2, 0.9850650239373595], [3, 1.03523446030743], [7, 0.7668727289693406]], 'Pirates': [[0, 1.047230823923881], [1, 1.0310717240451777], [2, 0.9635998682565224], [3, 0.8831111831537574], [7, 0.5599514579167826]], 'Swords': [[0, 1.273438252769557], [1, 0.9923350055998588], [2, 0.6066050369981921], [3, 0.1], [7, 0.1]], 'Vigilantes': [[0, 0.9014050613166084], [1, 0.9935402764867474], [2, 1.1289404506820713], [3, 1.246542636272343], [7, 1.2877589244356933]]}}, {'BASE_VALUE': 0.33425752072110015, 'featcomb': 'mult', 'conts': {'Angels': [[0, 1.0227252623810392], [1, 1.0909268168138695], [2, 1.0399742969296417], [3, 1.0866265592828626], [7, 0.1]], 'Battalions': [[0, 1.2039364170610585], [1, 0.9817784929434088], [2, 0.8104719818831331], [3, 0.6465320500226163], [7, 0.2713146410425944]], 'Dragons': [[0, 1.0319305872069875], [1, 1.0607642051756332], [2, 1.0500716119232847], [3, 1.0510372216163668], [7, 0.729608307611789]], 'Emperors': [[0, 1.1017169331683248], [1, 1.0254666414226628], [2, 0.9763370038370647], [3, 0.9449058153979931], [7, 0.3237927520727546]], 'Guards': [[0, 1.0950213220282212], [1, 1.0464964228535876], [2, 1.0005915255706008], [3, 0.8366751384881725], [7, 0.6708807889600171]], 'Hooligans': [[0, 0.8629791716429938], [1, 1.0553102137833132], [2, 1.2227590221222409], [3, 1.3839804261812019], [7, 1.182290761015496]], 'Knights': [[0, 1.138065193374833], [1, 1.0673146125156618], [2, 0.8725787425712003], [3, 0.7345843500924586], [7, 0.28583941769732396]], 'Lotuses': [[0, 1.112789629021284], [1, 1.05651894829088], [2, 0.9272092576015744], [3, 0.8660185281041457], [7, 0.21308773517084345]], 'Minotaurs': [[0, 0.9815509035612944], [1, 1.0893165268250775], [2, 1.1342889924862742], [3, 1.030528361595817], [7, 1.0099196024322892]], 'Pirates': [[0, 0.749512826511836], [1, 1.0812961260455116], [2, 1.3822674686769882], [3, 1.4385531033641765], [7, 1.2022560373535687]], 'Swords': [[0, 0.3601993321035487], [1, 0.9781705661056576], [2, 1.5042831757340442], [3, 2.1036904580990297], [7, 0.6694415054688967]], 'Vigilantes': [[0, 1.1095162439713153], [1, 1.0512146955934698], [2, 0.9946434216873553], [3, 0.7857150046603212], [7, 0.2973500842526371]]}}]

testDf["Predicted"] = misc.predict(testDf, models, calculus.Cratio_mlink)
testDf["Actual"] = testDf["Deck_A_Win?"]

print("GINI: " + str(metrics.get_gini(testDf,"Predicted","Actual")))
p, a = metrics.get_Xiles(testDf,"Predicted","Actual")


print("DECILES")
print([util.round_to_sf(x) for x in p])
print([util.round_to_sf(x) for x in a])


wraps.viz_cratio_models(models, "2_model")

