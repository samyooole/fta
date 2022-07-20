import pandas as pd

df = pd.read_csv('core_excels/totaPlus_MandatoryPredict.csv')
text=df.text.astype('str')
writePath = 'scripts/model/rob/tolabel.txt'
text.to_csv(writePath, header=None, index=None, sep=' ', mode='a')