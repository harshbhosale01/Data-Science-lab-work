import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.load_dataset("tips")
sns.boxplot(df["total_bill"]).set_title(" ")
sns.boxplot(x=df['tip'],y=df['smoker'])
sns.boxplot(x=df['tip'],y=df['day'])
sns.displot(df['total_bill'],kde=False)
sns.displot(df['tip'],kde=False)
sns.set_theme()
fig,ax= plt.subplots()
ax.scatter(df['tip'],df['total_bill'])
sns.lmplot(x='tip',y='total_bill',data=df)
sns.swarmplot(data=df,x='tip',y='total_bill')
sns.lmplot(data=df,x='tip',y='total_bill',col='time',hue='size',fit_reg=False)
