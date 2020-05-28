# Zkouška mé metody na datech z VŠB

Zkusil jsem svou metodu na datech z následujích složek:
train 1 - složka _133148
train 2 - složka _132752
train 3 - složka _131842
test 1 - 131413

Zkusil jsem jak boční úhel tak pohled zepředu, ale výsledky nebyly moc pozitivní. Často nerozpoznal kašel vůbec, jestli jsem na jednom modelu naměřil kolem 20% úspěšnosti rozpoznání kašle, tak zase při klasickém otáčení hlavy měl procenta o dost vyšší a ostatní aktivita na tom byla podobně jako kašel.
Tak jsem zkusil hledat v datech proč tomu tak je, jelikož na mých datech lze zřetelně vidět rozdíl mezi normálními a kašlajícími částmi snímků (lze vidět na obrázcích níže)...

![mojetrenovacidata](https://github.com/Pinkieqt/LSTM_nn/blob/master/VSBTrain/Moje%20trenovaci%20data.png)
![mojetrenovacidata](https://github.com/Pinkieqt/LSTM_nn/blob/master/VSBTrain/Mojetrenovacidata_D10_D12.png)

Na tomto rozdílu jsem tedy předtím taky pracoval a podle toho se snažil ty data vyhodnotit.

Ovšem trénovací data, která jste mi zaslal, jsou viditelně o mnoho komplexnější. 
Rozdíly mezi kašláním a normální aktivitou tam nejsou tak viditelné, což bude hlavní důvod proč má metoda na těchto datech asi nefungovala.

![train1](https://github.com/Pinkieqt/LSTM_nn/blob/master/VSBTrain/Train_1_990_1035.png)
![train1](https://github.com/Pinkieqt/LSTM_nn/blob/master/VSBTrain/Train_1_D10_D12.png)
![train3](https://github.com/Pinkieqt/LSTM_nn/blob/master/VSBTrain/Train_3.png)
![train3](https://github.com/Pinkieqt/LSTM_nn/blob/master/VSBTrain/Train_3_D10_D12.png)
