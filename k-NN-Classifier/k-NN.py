#######################################################
# Creator: Filipy                                     #
# LinkedIn: Filipy S. Furtado/The most beautiful guy  #
# Instagram: sf.filipy                                #
# Discord: furth__                                    #
#######################################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def gender(x):
    if x == 'Female': return 1
    else: return 0


def eletrocardiogram(x):
    if x == 'Normal': return 0
    elif x == 'ST': return 1


def pain(x):
    if x == 'TA': return 1
    elif x == 'ATA': return 2
    elif x == 'NAP': return 3
    else: return 0


def angina(x):
    if x == 'Y': return 1
    else: return 0


def slope(x):
    if x == 'Flat': return 1
    else: return 0


#Inefficient algorithm
def l2_distance(dataf):
    num_test = dataf.shape[0]
    dist = np.zeros(num_test, dtype=dataf.dtype)
    for i in range(num_test):
        l2 = np.sum(np.abs(df - dataf), axis=1)
        min_index = np.argmin(l2)
        dist [i] = df_dp[min_index]
    return dist


#transform = lambda x: np.sum(np.asarray(x)) / len(x)

with open('heart.csv', 'r') as f:
    df = pd.read_csv(f)
    f.close()

shape_df = df.shape
df['ST_Slope'] = df['ST_Slope'].apply(slope)
df['ExerciseAngina'] = df['ExerciseAngina'].apply(angina)
df['ChestPainType'] = df['ChestPainType'].apply(pain)
df['Sex'] = df['Sex'].apply(gender)
df['RestingECG'] = df['RestingECG'].apply(eletrocardiogram)
df_dp = df['HeartDisease']
df.drop(['HeartDisease'], axis=1, inplace=True)

#Insert Dataframes
"""dists = l2_distance(df_dp[:])
fig, (axs_1, axs_2, axs_3) = plt.subplots(3, 1)
axs_1.plot(df, c=(0.132,0.71,0.255))
axs_1.set_title('Features mean')
axs_2.scatter(("Quantity of dependent variables", "Features."), (dists[0:,], dists[:0,]))
axs_2.set_title('Dists Graphic')
axs_3.scatter(df, df_dp, c=(0.2321, 0.125, 0.333), alpha=0.2)
axs_3.set_title('Linear comparisson')"""

plt.show()