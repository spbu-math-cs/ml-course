import pandas as pd
from glob import glob
from split import StratifiedGroupKFold

data_path = '/datasets/train_classifier/uw-madison-gi-tract-image-segmentation/'
anno_path = 'train.csv'

df = pd.read_csv(anno_path)
df['mask_path'] = df['mask_path'].apply(lambda x: x.split("train")[1])
df['image_path'] = df['image_path'].apply(lambda x: x.split("train")[1])
df['segmentation'] = df.segmentation.fillna('')
df['rle_len'] = df.segmentation.map(len) # length of each rle mask
df['mask_path'] = df.mask_path.str.replace('.png', '.npy')
print(df.iloc[0]['mask_path'])
print(df.iloc[0]['image_path'])

df2 = df.groupby(['id'])['segmentation'].agg(list).to_frame().reset_index() # rle list of each id
df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(sum).to_frame().reset_index()) # total length of all rles of each id

df = df.drop(columns=['segmentation', 'class', 'rle_len'])
df = df.groupby(['id']).head(1).reset_index(drop=True)
df = df.merge(df2, on=['id'])
df['empty'] = (df.rle_len==0) # empty masks
# print(df['height'].head(50))
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2022)
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups=df["case"])):
    df.loc[val_idx, 'fold'] = int(fold)

df.to_csv('train_folds.csv', index=False)
print(df.head(50))
