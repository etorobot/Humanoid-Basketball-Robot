import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PATH='/home/admin/Downloads/len2/'
# PATH='/home/admin/Downloads/chart_len/'
fileNames = os.listdir(PATH)
fileNames = [file for file in fileNames if '.csv' in file]
plt.figure(figsize=(11,5), dpi=300)
# x,y =[],[]
### Loop over all files
for file in fileNames:
    # print (PATH+file)
    # x = np.loadtxt(PATH + file)
    # if 'p' not in file:
    #     if '33%_NO_DR' in file:
    #         da = pd.read_csv(PATH + file, encoding='UTF8')
    #         df = pd.DataFrame(da)
    #         X = (df.iloc[:,0]); y=np.arange(len(X))
    #         plt.bar(X, y, width=40, label=file[:-5], alpha=0.6, color='C3' )
    #     elif '10%_NO_DR' in file:
    #         da = pd.read_csv(PATH + file, encoding='UTF8')
    #         df = pd.DataFrame(da)
    #         X = (df.iloc[:,0]); y=np.arange(len(X))
    #         plt.bar(X, y, width=40, label=file[:-5], alpha=0.6, color='C2' )
    #     elif '10%_DR' in file:
    #         da = pd.read_csv(PATH + file, encoding='UTF8')
    #         df = pd.DataFrame(da)
    #         X = (df.iloc[:,0]); y=np.arange(len(X))
    #         plt.bar(X, y, width=40, label=file[:-5], alpha=0.6, color='C1' )
    #     else:
    da = pd.read_csv(PATH + file, encoding='UTF8');df = pd.DataFrame(da)
    X = (df.iloc[:,0].median())
    xpos = [1199, 925, 1200, 765, 771, 1152, 672, 970]
    print ("means", file, X)
    team = [
        "Control(0%)",
        'Random(25%)',
        'Random(40%)',
        'Random(50%)',
        # 'Randomized(75%)',
        'Random(100%)']
# withDR = [1152.7878, 970.77, 1200.8, 1199.047]
withDR = [
    1281,
    890, #25%
    691, #40%
    784,
    869]
# noDR=[672.3, 765.16, 771.13, 925.034]
noDR=[
    638.5, #CTRL
    609, #25
    610, #40
    571, #50
    557 #100
    ]
xaxis = np.arange(len(noDR))
plt.bar(xaxis-0.14, withDR, width=0.2, label='With DR')
plt.bar(xaxis+0.14, noDR, width=0.2, label='Without DR')
plt.xticks(xaxis, team, fontsize = 16)
y = [1,2,3,4,5,6, 7, 8]
            # y=np.arange(len(X))
    # plt.bar(X+1, X, width=01.4, label=file[:-5], alpha=0.6 )

# plt.ylabel('No. episodes', fontsize=15)
plt.ylabel('Episode Length', labelpad=5,  fontsize=15)
# plt.ylabel('Episode Length', fontsize=20)
# plt.title('Balance Task (Minimum Floor Friction)' ) #, fontsize = 17 )
plt.legend(fontsize=26.5, loc=1, ncol=2, framealpha=0.6)
# plt.grid(linewidth=0.4)
# plt.subplots_adjust(right=0.97, left=0.085, bottom=0.08, top=0.98)
# plt.xticks(range(1, 10_000))
plt.show()