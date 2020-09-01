import pandas as pd
import pickle as pk

df = pd.read_csv(".../data/valnew.csv")
df.info()
df.dropna(inplace = True)

target = df['Survived']
del(df["Survived"])

filename='model.pkl'
outfile=open(filename,'wb')
pk.dump(df,outfile)
outfile.close()

infile = open(filename,'rb')
new_df = pk.load(infile)
infile.close()

predictions = new_df.predict(new_df)
# Reassign target (if it was present) and predictions.
train_df["prediction"] = predictions
train_df["target"] = target

ok = 0
for i in df.iterrows():
    if (i[1]["target"] == i[1]["prediction"]):
        ok = ok + 1

print("accuracy is", ok / df.shape[0])

