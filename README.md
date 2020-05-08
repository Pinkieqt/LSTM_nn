# Detekce kašle pomocí neuronových sítí

Student: Dušan Kučeřík, KUC0277


## Příprava dat

Většinu videí jsem natočil na svůj osobní telefon. Zkoušel jsem i verzi s GoPro, ovšem mám starší model (ročník 2014) a kvalita podle mě nebyla dostatečná pro správnou detekci obličejových bodů pomocí knihovny OpenPose, jelikož jsem musel zmenšit rozlišení pro OpenPose, aby to má grafická karta zvládla v rozumném čase.

Jednotlivá videa jsem po natočení upravil pomocí editovacího softwaru, ořízl do čtvercové podoby (800x800 px), upravil světelné podmínky/stíny a doostřil.

Video jsem poté "rozsekal" pomocí skriptu [framesFromVideo.py](https://github.com/Pinkieqt/LSTM_nn/blob/master/OpenPose/Preparing/framesFromVideo.py) na jednotlivé snímky.

Poté jsem vygeneroval jednotlivé obličejové body pro jeden zdroj snímků do .CSV souboru pomocí knihovny OpenPose. ([C++ kod](https://github.com/Pinkieqt/LSTM_nn/blob/master/OpenPose/MyFaceImplementation.cpp))

Tento .CSV soubor jsem načetl pomocí [DatasetParseru.py](https://github.com/Pinkieqt/LSTM_nn/blob/master/Dataset_Parser.py), přes který jsem vypočítal jednotlivé vzdálenosti D1-D14 a uložil je pro každý snímek do dalšího separátního .CSV souboru.

Jednotlivé vzdálenosti D1-D14 jsou zvýrazněny na následném obrázku.

![openpose keypoints](https://github.com/Pinkieqt/LSTM_nn/blob/master/Media/keypoints_face.png)

Tímto postupem jsem získal veškeré potřebné data z natočeného videa.

## Trénování

###### Trénovací data: [Front view](https://github.com/Pinkieqt/LSTM_nn/blob/master/Final_datasets/iphone_new.csv), [Side view](https://github.com/Pinkieqt/LSTM_nn/blob/master/Final_datasets/iphone_side.csv), [Normalized and combined Front and Side view](https://github.com/Pinkieqt/LSTM_nn/blob/master/Final_datasets/train_front_and_side.csv)
###### Testovací data: [Front view test](https://github.com/Pinkieqt/LSTM_nn/blob/master/Final_datasets/iphone_front_test.csv), [Front view test 2](https://github.com/Pinkieqt/LSTM_nn/blob/master/Final_datasets/iphone_front_test_2.csv), [Front view test (only normal data)](https://github.com/Pinkieqt/LSTM_nn/blob/master/Final_datasets/iphone_normal.csv), [GoPro Side test](https://github.com/Pinkieqt/LSTM_nn/blob/master/Final_datasets/gopro.csv)

Model jsem trénoval na datech ze souboru [iphone_new.csv](https://github.com/Pinkieqt/LSTM_nn/blob/master/Final_datasets/iphone_new.csv), kdy byla kamera umístěna na palubní desce přesně v rovině s obličejem. Po mnoha pokus-omyl jsem jakž takž dosáhl nějakého rozumného výsledku. Ovšem po testování tohoto modelu na datech z [iphone_side.csv](https://github.com/Pinkieqt/LSTM_nn/blob/master/Final_datasets/iphone_side.csv), kde kamera natáčela z úhlu, neprokazoval žádné rozumné výsledky.

Zkusil jsem tedy všechny data normalizovat. Normalizoval jsem každý obrázek a jeho obličejové body/data pomocí vzorce *normalisedData = (data - mean) / var*. 

Data před normalizací: (modré = pohled zepředu, oranžové = pohled z boku)
![before normalization](https://github.com/Pinkieqt/LSTM_nn/blob/master/Media/data%20before%20normalization.png)
A data po normalizaci: (modré = pohled zepředu, oranžové = pohled z boku)
![after normalization](https://github.com/Pinkieqt/LSTM_nn/blob/master/Media/data%20after%20normalization.png)
Ovšem po normalizaci, spojení těchto dvou datasetů a následném trénování modelu jsem získával někdy rozumný odhad, ale spíše ne zrovna rozumný odhad.

Odhad na testovacím videu po normalizaci, spojení a natrénování:
![normalized data predict](https://github.com/Pinkieqt/LSTM_nn/blob/master/Media/normalized%20data.png)

Nakonec jsem usoudil, že trénovací data (i po normalizaci) nemohou pocházet z úplně jiných zdrojů, nýbrž velmi podobných.

Natrénovaný model na ne-normalizovaných datech z datasetu [iphone_new.csv](https://github.com/Pinkieqt/LSTM_nn/blob/master/Final_datasets/iphone_new.csv) jsem tedy nechal odhadovat na testovacím videu s tímto výsledkem:
![enter image description here](https://github.com/Pinkieqt/LSTM_nn/blob/master/Media/final%20test_2.png)



