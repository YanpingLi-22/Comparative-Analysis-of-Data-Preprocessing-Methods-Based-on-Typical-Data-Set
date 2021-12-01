import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import boxcox
from scipy.stats import yeojohnson
df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\twoclass\adult_new.csv")
#print(df["education-num"])
df["hours-per-week"] = boxcox(df['hours-per-week'])[0]
#df["fnlwgt"] = 1/df["fnlwgt"]
#df["capital-loss"] = yeojohnson(df["capital-loss"])[0]
#df["capital-loss"] = np.log1p(df["capital-loss"])
sns.displot(df["hours-per-week"],kde=True)
skewness=str(df["hours-per-week"].skew())
kurtosis=str(df["hours-per-week"].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))
plt.savefig('age.png', dpi=300, bbox_inches = 'tight')
plt.show()
stats.probplot(df["hours-per-week"],dist="norm",plot=plt)
plt.show()