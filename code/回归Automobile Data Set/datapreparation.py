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
df = pd.read_csv(r"C:\Users\AAA\Desktop\new_project\huigui\new-imports-85.csv")

#df["Height"] = boxcox(df['Height'])[0]
#df["Shellweight"] = 1/df["Shellweight"]
df["length"] = yeojohnson(df["length"])[0]
#df["Shellweight"] = np.log1p(df["Shellweight"])
sns.displot(df["length"],kde=True)
skewness=str(df["length"].skew())
kurtosis=str(df["length"].kurt())
plt.legend([skewness,kurtosis],title=("skewness and kurtosis"))
plt.savefig('age.png', dpi=300, bbox_inches = 'tight')
plt.show()
stats.probplot(df["length"],dist="norm",plot=plt)
plt.show()

conf_mat = np.zeros(4)
conf_mat[1]=12
conf_mat.reshape(2, 2).astype(np.int16)
print(conf_mat[0, 0] / conf_mat[0, :])
