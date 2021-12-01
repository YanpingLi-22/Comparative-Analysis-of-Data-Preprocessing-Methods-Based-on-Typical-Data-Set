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
df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\muiltclass\cmc.csv")
#print(df["education-num"])
#df["NumOfBorn"] = boxcox(df['NumOfBorn'])[0]
#df["fnlwgt"] = 1/df["fnlwgt"]
df["NumOfBorn"] = yeojohnson(df["NumOfBorn"])[0]
df["NumOfBorn"] = yeojohnson(df["NumOfBorn"])[0]
#df["capital-loss"] = np.log1p(df["capital-loss"])
sns.displot(df["NumOfBorn"],kde=True)
skewness=str(df["NumOfBorn"].skew())
kurtosis=str(df["NumOfBorn"].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))
plt.savefig('age.png', dpi=300, bbox_inches = 'tight')
plt.show()
stats.probplot(df["NumOfBorn"],dist="norm",plot=plt)
plt.show()