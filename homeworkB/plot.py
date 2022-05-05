import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data/RB10000.csv', delimiter=',')

plt.plot(df, alpha=0.4, label='data')
plt.plot(df.rolling(100).mean(), label='100 moving average')
plt.title('Reward vs episodes')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend(loc='lower right')
plt.show()
print('bajs')