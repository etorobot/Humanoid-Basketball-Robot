import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# PATH='/home/admin/Downloads/costs/'
PATH='/home/admin/Downloads/monREW/'
# PATH='/home/admin/Downloads/rew/f7C/'
# PATH='/home/admin/Downloads/chart_len/'
fileNames = os.listdir(PATH)

### Filter file name list for files ending with .csv
fileNames = [file for file in fileNames if '.csv' in file]
plt.figure(figsize=(8,3), dpi=450)
i=0
### Loop over all files
for file in fileNames:
    ### Read .csv file and append to list
    df = pd.read_csv(PATH + file, index_col = 1)
    means = pd.DataFrame(np.array(df['Value']).T)  
    stds = pd.DataFrame(np.zeros(df['Value'].shape)) 
    meanst = np.array(means.iloc[i].values[:], dtype=np.float64)
    sdt = np.array(stds.iloc[i].values[:], dtype=np.float64)    
    ### Create line for every file
    # for i in range(len(fileNames)):
    if i >100:
        None

    # elif i %2 == 0:
    #     plt.plot(df['Value'], label='_nolegend_', linewidth=4)
    # elif i %3 == 0:
    #     plt.plot(df['Value'], label='_nolegend_', linewidth=4)
    # if  '1' in file:
    #     plt.plot(df['Value'], label=file[:-5], linewidth=4)
        
    elif 'stand' in file:
        plt.plot(df['Value'], label='_nolegend_', linewidth=3, color="C4")
    # elif 'spk' in file:
    #     plt.plot(df['Value'], label='_nolegend_', linewidth=4, color="C1")
    # elif 'drb' in file:
    #     plt.plot(df['Value'], label='_nolegend_', linewidth=4, color="C5")
    # elif 'srv' in file:
    #     plt.plot(df['Value'], label='_nolegend_', linewidth=4, color="C3")
    # elif 'catch' in file:
    #     plt.plot(df['Value'], label='_nolegend_', linewidth=4, color="C2")
    # elif 'hold' in file:
    #     plt.plot(df['Value'], label='_nolegend_', linewidth=4, color="C6")   
    # elif '7' in file:
    #     plt.plot(df['Value'], label='_nolegend_', linewidth=4, color="C0")                
    # if  '2' in file:
    #     plt.plot(color="r")
        
    i += 1
    # plt.plot(df[:], label=file[:-4])
    # for i in fileNames:
    # plt.fill_betweenx(df['Value'], 0, 10e5, color='C2', alpha=.02)
    # plt.fill_betweenx(df['Value'],meanst-sdt, meanst+sdt, color='C2', alpha=.02)
    # plt.fill_betweenx(df['Value'], 10e5, 15e5, color='C8', alpha=.02)
    # plt.fill_betweenx(df['Value'], 15e5, 20e5, color='C9', alpha=.02)
    # plt.fill_betweenx(df['Value'], 20e5, 30e5, color='C6', alpha=.02)
    # plt.fill_betweenx(df['Value'], 15e5, 20e5, color='yellow', alpha=.1)
# plt.rcParams.update({'font.size': 72})
plt.xlabel('Time steps')
plt.ylabel('Mean Reward', labelpad=1)
# plt.ylabel('Episode Length', fontsize=20)
# plt.title('Holding Task (Initial Domain Randomization, No Curriculum)' ) #, fontsize = 17 )
plt.legend(fontsize=12.5, loc=2, ncol=4, framealpha=0.8)
plt.grid(linewidth=0.4)
plt.subplots_adjust(right=0.97, left=0.08, bottom=0.08, top=0.98)
# plt.xticks(range(1, 10_000))
plt.show()