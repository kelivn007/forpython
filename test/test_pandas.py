import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 随机生成1000个数据
data = pd.Series(np.random.randn(1000), index=np.arange(1000))

print(data)

# pandas 数据可以直接观看其可视化形式
data.plot()

plt.show()