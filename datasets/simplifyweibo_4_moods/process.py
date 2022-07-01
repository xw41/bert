import pandas as pd
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    df = pd.read_csv('simplifyweibo_4_moods.csv')
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=28, shuffle=True)
    df_train.to_csv('train.tsv', sep='\t', index=False)
    df_test.to_csv('test.tsv', sep='\t', index=False)
