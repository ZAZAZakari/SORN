'''
======================================================================================
DESCRIPTION: PLOTTING THE MEAN AND STANDARD DEVIATION OF THE FITNESS OVER GENERATION
			 OF THE TWO CASES: LEARNING+EVOLUTION, EVOLUTION ALONE
DATE: 1 JULY 2014
======================================================================================
'''

# =========================== IMPORTING ESSENTIAL LIBRARIES ========================== #
import numpy as np
import matplotlib.pyplot as plt

# ============================ INITIALIZING CONSTANTS ================================ #
path = "fitness/"			# [PATH] THE PATH SAVING ALL THE FILES OF THE FITNESS
dataSize = 20				# [INTEGER] THE NUMBER OF FITNESS FILE YOU HAVE
interval = 30				# [INTEGER] THE SAMPLING RATE
generations = 1799			# [INTEGER] THE NUMBER OF GENERATIONS OF THE FITNESS

meanFig1 = []				# [GRAPH] THE MEAN OF THE FITNESS OVER EACH GENERATION  
meanFig2 = []				# [GRAPH] THE MEAN OF THE FITNESS OVER EACH GENERATION
sdFig1 = []					# [GRAPH] THE STANDARD DEVIATION OF THE FITNESS OVER EACH GENERATION 
sdFig2 = []					# [GRAPH] THE STANDARD DEVIATION OF THE FITNESS OVER EACH GENERATION
x = []						# [X-AXIS] OF THE GRAPH
	
mean1 = np.zeros(generations)	# [1D ARRAY] THE MEAN OF THE FITNESS OVER EACH GENERATION
mean2 = np.zeros(generations)	# [1D ARRAY] THE MEAN OF THE FITNESS OVER EACH GENERATION
standardDeviation1 = np.zeros(generations)	# [1D ARRAY] THE STANDARD DEVIATION OF THE FITNESS OVER EACH GENERATION
standardDeviation2 = np.zeros(generations)	# [1D ARRAY] THE STANDARD DEVIATION OF THE FITNESS OVER EACH GENERATION
dataSet1 = np.zeros(shape=(dataSize, generations))	# [2D ARRAY] THE FITNESS OVER EACH GENERATION IN SEVERAL RUNS
dataSet2 = np.zeros(shape=(dataSize, generations))	# [2D ARRAY] THE FITNESS OVER EACH GENERATION IN SEVERAL RUNS

'''
======================================================================================
======================================================================================
======================================================================================
							MAIN PROGRAM STARTS HERE
======================================================================================
======================================================================================
======================================================================================
'''
# ============= LOADING DATASETS FROM SAVED FILES IN DIRECTORY:path ============ #
for i in range(0, dataSize):
	a = np.loadtxt(path + "fitness_evolution" + str(i) + ".txt")
	b = np.loadtxt(path + "fitness_learning_evolution" + str(i) + ".txt")
	
	for j in range(0, generations):
		dataSet1[i][j] = a[j]
		dataSet2[i][j] = b[j]

# =========== CALCULATING THE MEAN OF THE FITNESS OVER GENERATIONS OF THE TWO CASES =========== #
mean1 = np.mean(dataSet1, axis=0)
mean2 = np.mean(dataSet2, axis=0)

# ==== CALCULATING THE STANDARD DEVIATION OF THE FITNESS OVER GENERATIONS OF THE TWO CASES ==== #
standardDeviation1 = np.std(dataSet1, axis=0)
standardDeviation2 = np.std(dataSet2, axis=0)

# =========== SAMPLING THE MEAN AND SD WITH THE GIVEN SSAMPLING RATE =========== #
for i in range(0, generations):
	if i % interval == 0:
		x.append(i)
		meanFig1.append(mean1[i])
		meanFig2.append(mean2[i])
		sdFig1.append(standardDeviation1[i])
		sdFig2.append(standardDeviation2[i])

# ========== PLOTTING THE TWO ERROR BAR IN THE SAME FIGURE WITH DIFFERENT COLOR AND LABEL =========== #
plt.errorbar(x, meanFig2, sdFig2, linestyle='-', marker='^', label='Learning + Evolution')
plt.errorbar(x, meanFig1, sdFig1, linestyle='-', marker='^', label='Evolution Alone')
plt.legend(loc="lower right")
plt.show()

'''
======================================================================================
======================================================================================
======================================================================================
									END
======================================================================================
======================================================================================
======================================================================================
'''