# solubility_rf.py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读数据
df = pd.read_csv('AqSolDB.csv')          # 字段：SMILES, logS
print('样本数:', len(df))

# 2. 把 SMILES → MACCS 指纹（167 维）
def smi_to_maccs(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return list(MACCSkeys.GenMACCSKeys(mol).ToList())
X = df['SMILES'].apply(smi_to_maccs).tolist()
X = pd.DataFrame(X).dropna()             # 去掉解析失败的
y = df['Solubility'].loc[X.index]

# 3. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# 4. 随机森林回归
model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5. 预测 + 评估
pred = model.predict(X_test)
rmse = mean_squared_error(y_test, pred) ** 0.5
print(f'测试集 RMSE: {rmse:.3f}')

# 6. 画图
plt.figure(figsize=(5,5))
sns.scatterplot(x=y_test, y=pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('实测 logS')
plt.ylabel('预测 logS')
plt.title(f'RandomForest + MACCS  (RMSE={rmse:.2f})')
plt.tight_layout()
plt.show()