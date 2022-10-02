import matplotlib.pyplot as plt
import pandas as pd, os
import matplotlib.ticker as ticker

df = pd.read_csv('/home/admin/Downloads/plother.txt', sep=" ", header=None)
df.columns = ["y1", "y2", "y3","y4", "y5", "y6", "y7", "y8", "y9", "y10"]

x = range(5606)
# xi = list(range(len([:1000])))
print(df)

fig, ax = plt.subplots(figsize=(10,8), dpi=400)

# ax = plt.figure()

ax.plot(x, df['y1'], label='Standing', linewidth=3)
ax.plot(x, df['y2'], label='Head Tilt', linewidth=3)
ax.plot(x, df['y3'], label='Hand Position (XYZ)', linewidth=3)
ax.plot(x, df['y4'], label='Ball Contact', linewidth=5)
ax.plot(x, df['y5'], label='Consecutive Contact', linewidth=5)
ax.plot(x, df['y6'], label='Ball Height', linewidth=2)
ax.plot(x, df['y7'], label='Foot Tilt', linewidth=3)
ax.plot(x, df['y8'], label='Ball Dropped', linewidth=3)
ax.plot(x, df['y9'], label='Alive', linewidth=3)
# ax.plot(x, df['y7'], label='Terminated')
# tick_spacing = 1
# ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plt.xlabel('Time step')
plt.ylabel('Reward')
# plt.title('Per Episode Reward', fontsize=15)
plt.grid(linewidth=0.4)
plt.subplots_adjust(right=0.97, left=0.085, bottom=0.12)
plt.legend(fontsize=10.1, loc=9, ncol=5, bbox_to_anchor=(.5,.9), framealpha=0.8)
plt.show()