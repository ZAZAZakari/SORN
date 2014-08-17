'''
======================================================================================
DESCRIPTION: IMPLEMENTION OF ROYAL ROAD FITNESS WITH EVOLUTION ONLY
DATE: 23 JUNE 2014
======================================================================================
'''
# =========================== IMPORTING ESSENTIAL LIBRARIES ========================== #
import aux01
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================ INITIALIZING CONSTANTS ================================ #
thresholdOn = 5			# [INTEGER] THRESHOLD OF DEFINING ON OF FIRING NEURONS
N = 21					# [INTEGER] PROBLEM DIMENSIONALITY

pMut = 0.3				# [FLOATING NUMBER] PROBABILITY OF MUTATION 
etaMut = 0.5			# [FLOATING NUMBER] NOISE STRENGTH OF MUTATION 
NN_E = 120				# [INTEGER] NUMBER OF EXCITORY NEURONS IN THE MODEL 
NN_I = 24				# [INTEGER] NUMBER OF INHIBITORY NERUONS IN THE MODEL = 0.2*NN_E
NUM_I = 120				# [INTEGER] NUMBER OF THE EXTERNAL INPUT NEURONS  
						# THE INPUT NEURONS ARE THE FIRST NUM_I EXCITATORY NEURONS
						
TE_MAX = 1.0			# [FLOATING NUMBER] MAXIMUM OF EXCITATORY THRESHOLD
TI_MAX = 0.5			# [FLOATING NUMBER] MAXIMUM OF INHIBITORY THRESHOLD

sig2 = 0.001			# [FLOATING NUMBER] SD OF RANDOM NOISE ON EXCITATORY NEURONS
muIP = 0.2				# [FLOATING NUMBER] THE RATE OF NEURONS FIRING
etaIP = 0.01			# [FLOATING NUMBER] STRENGTH OF EXCITATORY THRESHOLD PLASTICITY
etaSTDP = 0.004			# [FLOATING NUMBER] STRENGTH OF STDP
etaINHIB = 0.001		# [FLOATING NUMBER] STRENGTH OF INHIBITORY STDP
strength = 0.8			# [FLOATING NUMBER] STRNEGHT OF EXTERNAL INPUTS

pEE = 0.1				# [PROBABILITY] OF INITIALIZING E --> E CONNECTIONS
pEI = 1.0				# [PROBABILITY] OF INITIALIZING I --> E CONNECTIONS
paddEE = 0.1			# [PROBABILITY] OF ADDING A NEW E --> E CONNECTION

SAVE_STEPS = 2000		# [INTEGER] NUMBER OF TIME STEPS TO BE SAVED

duration = 1			# [INTEGER] NUMBER OF TIME STEPS OF EACH ACTIVATION
stateIteration = 50		# [INTEGER] NUMBER OF STATES IN EACH ITERATION
iteration = 1800		# [INTEGER] NUMBER OF ITERATION 

fix = 1					# [BOOLEAN] TO SWITCH ON AND OFF PLASTICITY

scoreFig = [] 										# [GRAPH] THE PLOT OF SCORES OVER TIME
STEPS = duration * stateIteration * iteration 		# [INTEGER] TOTAL NUMBER OF STEPS IN THIS RUN

VISUAL = 1 											# [BOOLEAN] TO SWITCH ON AND OFF VISUALISATION
MASK = np.random.binomial(1,0.5,N)					# [MASK] THE BIT MASK APPLIED FOR THE STRING PATTERN

# =========================== INITIALIZING ARRAYS ================================ #
te = np.zeros(NN_E)								# [1D ARRAY] THRESHOLD OF ALL EXCITATORY NEURONS
ti = np.zeros(NN_I)								# [1D ARRAY] THRESHOLD OF ALL INHIBITORY NEURONS
inp = np.zeros(NUM_I)							# [2D ARRAY] ACTIVITY STATES OF EXTERNAL NEURONS
x = np.zeros(shape=(2, NN_E))					# [2D ARRAY] ACTIVITY STATES OF EXCITATORY NEURONS
y = np.zeros(shape=(2, NN_I))					# [2D ARRAY] ACTIVITY STATES OF INHIBITORY NEURONS

wee = np.zeros(shape=(NN_E, NN_E))				# [2D ARRAY] WEIGTHS OF ALL E ---> E CONNECTIONS
wei = np.zeros(shape=(NN_E, NN_I))				# [2D ARRAY] WEIGHTS OF ALL E ---> I CONNECTIONS
wie = np.zeros(shape=(NN_I, NN_E))				# [2D ARRAY] WEIGHTS OF ALL I ---> E CONNECTIONS

img = np.zeros(shape=(NN_E+NN_I, SAVE_STEPS))	# [2D ARRAY] SPIKING OF THE NEURONS OVER TIME
 
stringPattern = np.zeros(N)						# [1D ARRAY] THE STRING PATTERN IN THE GA
score = np.zeros(3)								# [1D ARRAY] THE SCORE IN THE GA

tempx = np.zeros(shape=(2, NN_E))				# [1D ARRAY] THE SAVING PARAMETERS FOR x
tempy = np.zeros(shape=(2, NN_I))				# [1D ARRAY] THE SAVING PARAMETERS FOR y
savedWEE = np.zeros(shape=(3, NN_E, NN_E))		# [3D ARRAY] THE SAVING PARAMETERS FOR wee
savedWEI = np.zeros(shape=(3, NN_E, NN_I))		# [3D ARRAY] THE SAVING PARAMETERS FOR wei
savedWIE = np.zeros(shape=(3, NN_I, NN_E))		# [3D ARRAY] THE SAVING PARAMETERS FOR wie
savedTE = np.zeros(shape=(3, NN_E))				# [3D ARRAY] THE SAVING PARAMETERS FOR te
savedTI = np.zeros(shape=(3, NN_I))				# [3D ARRAY] THE SAVING PARAMETERS FOR ti
'''
======================================================================================
FUNCTION: INITIALIZING ALL THE VARIABLES 
======================================================================================
'''
def init():
	# ================= GLOBALIZE VARIABLES ====================== #
	global te, ti, wee, wei, wie, tempx, tempy
	global savedWEE, savedWEI, savedWIE, savedTE, savedTI
	global score, scoreFig, stringPattern, MASK
	
	# ===== FOR ALL THE EXCITATORY NEURONS, RANDOMLY ASSIGNED THRESHOLD FOR THEM ==== # 
	for i in range(0, NN_E):
		te[i] = aux01.randd() * TE_MAX 

	# ===== FOR ALL THE INHIBITORY NEURONS, RANDOMLY ASSIGNED THRESHOLD FOR THEM ==== #
	for i in range(0, NN_I):
		ti[i] = aux01.randd() * TI_MAX 

	# ===== FOR ALL E ---> I PAIRS, RANDOMLY ASSIGNED WEIGHT BETWEEN THEM WITH PROB 1.0 ===== #
	for i in range(0, NN_I):	
		sum = 0 
		# ======== SUMMING THE WEIGHTS ======== #
		for j in range(0, NN_E):
			wie[i][j] = aux01.randd() 
			sum = sum + wie[i][j]
		
		# ===== PERFORMING NORMALIZATION ====== #
		for j in range(0, NN_E):
			if (sum <> 0):
				wie[i][j] = wie[i][j] / sum 

	# ===== FOR ALL E ---> E PAIRS, RANDOMLY ASSIGNED WEIGHT BETWEEN THEM WITH PROB pEE ===== #
	for i in range(0, NN_E):	
		sum = 0 
		# ======== SUMMING THE WEIGHTS ======== #
		for j in range(0, NN_E): 
			if ((i <> j) and (aux01.randd() <= pEE)):	# >>NO SELF-CONN
				wee[i][j] = aux01.randd() 
				sum = sum + wee[i][j] 
			else:
				wee[i][j] = -99.0 	# NOT CONNECTED

		# ===== PERFORMING NORMALIZATION ====== #
		for j in range(0, NN_E):
			if ((sum <> 0.0) and (wee[i][j] >= 0.0)):
				wee[i][j] = wee[i][j] / sum 
				
	# ===== FOR ALL I ---> E PAIRS, RANDOMLY ASSIGNED WEIGHT BETWEEN THEM WITH PROB pEI ===== #
	for i in range(0, NN_E):	
		sum = 0 
		# ======== SUMMING THE WEIGHTS ======== #
		for j in range(0, NN_I):
			if (aux01.randd() <= pEI):
				wei[i][j] = aux01.randd() 
				sum = sum + wei[i][j] 
			else:
				wei[i][j] = -99.0 	# NOT CONNECTED
		
		# ===== PERFORMING NORMALIZATION ====== #
		for j in range(0, NN_I):
			if ((sum <> 0) and (wei[i][j] >= 0)):
				wei[i][j] = wei[i][j] / sum 

	# ========== INITIALIZING THE STATE OF EXCITATORY NEURONS TO ZERO ========== #
	for i in range(0, NN_E):
		x[0][i] = 0
		tempx[0][i] = 0
	
	# ========== INITIALIZING THE STATE OF INHIBITORY NEURONS TO ZERO ========== #
	for i in range(0, NN_I):
		y[0][i] = 0
		tempy[0][i] = 0
		
	# ========== INITIALIZING THE STATE OF SAVING VARIABLES OF THE WEIGHTS ========== #
	savedWEE = np.zeros(shape=(3, NN_E, NN_E))		
	savedWEI = np.zeros(shape=(3, NN_E, NN_I))		
	savedWIE = np.zeros(shape=(3, NN_I, NN_E))		
	
	# ========== INITIALIZING THE STATE OF SAVING VARIABLES OF THE THRESHOLDS ======= #
	savedTE = np.zeros(shape=(3, NN_E))				
	savedTI = np.zeros(shape=(3, NN_I))		
	
	# ========== INITIALIZING THE VARIABLES USED IN EVOLUTIONARY STEPS =========== #
	stringPattern = np.zeros(N)						
	score = np.zeros(3)								
	MASK = np.random.binomial(1,0.5,N)	
	scoreFig = []
'''
======================================================================================
FUNCTION: IMPLMENTING THE STEPS OF THE SORN MODEL 
======================================================================================
'''
def step(t):
	# ============ GLOBALIZING VARIABLES ============== #
	global wee, wei, wie, te, ti, x, y

	# ================ DISPLAY STEPS COUNTING EVERY 100 STEPS ================== #
	if (t % 100 == 0):
		print "Running step " + str(t) + " / " + str(STEPS) 
		
	# =============== INDEX OF ACTIVITY STATES FOR THE LAST AND CURRENT STEP ============== #
	t0 = (t-1) % 2 
	t1 = t % 2 
	
	# ================================================================================== #
	# ============ UPDATING ACTIVITIES OF EXCITATORY NEURONS USING EQ 3.4 ============== #
	# ================================================================================== #
	for i in range(0, NN_E):	
		sum1 = 0.0 
		sum2 = 0.0 

		# =============== CALCULATING ALL THE E-E CONNECTIONS TO THAT NEURON ============= #
		for j in range(0, NN_E):
			if (wee[i][j] >= 0.0):
				sum1 += wee[i][j] * x[t0][j] 
		
		# =============== CALCULATING ALL THE I-E CONNECTIONS TO THAT NEURON ============= #
		for j in range(0, NN_I):
			if (wei[i][j] >= 0.0):
				sum2 += wei[i][j] * y[t0][j] 
				
		# =============== CALCULATING RESULTANT SUMMATION ================ #
		sum = sum1 - sum2 + math.sqrt(sig2) * aux01.gasdev() 
		
		# =============== CACULATING SUMMATION BY THE INPUT NEURONS ============ #
		if (i < NUM_I):
			sum = sum + inp[i] 
		
		# =============== OBTAIN ACTIVITY BY COMPARING SUMMATION WITH ITS THRESHOLD ============ #
		if (sum > te[i]):
			x[t1][i] = 1 
		else:
			x[t1][i] = 0 

	# ================================================================================== #
	# ============= UPDATING ACTIVITIES OF INHIBITORY NEURONS USING EQ 3.5 ============= #
	# ================================================================================== #
	for i in range(0, NN_I):	# INHIBITORY UPDATE (EQ.2)
		sum1 = 0 
			
		# ============== CALCULATING ALL THE E-I CONNECTIONS TO THAT NEURON ================ #
		for j in range(0, NN_E):
			sum1 += wie[i][j] * x[t0][j] 
			
		sum1 += math.sqrt(sig2) * aux01.gasdev() 
		
		# =============== OBTAIN ACTIVITY BY COMPARING SUMMATION WITH ITS THRESHOLD ============ #
		if (sum1 > ti[i]):
			y[t1][i] = 1 
		else:
			y[t1][i] = 0 

	if fix == 0:
		for i in range(0, NN_E):	
			sum = 0 	
			# ================================================================================== #
			# ==================== UPDATING E-E WEIGHTS BY STDP (EQ 3.7) ======================= #
			# ================================================================================== #
			for j in range(0, NN_E):
				if ((wee[i][j] > 0.0) and (i <> j)):
					tt = etaSTDP * (float(x[t1][i]) * float(x[t0][j]) - float(x[t0][i]) * float(x[t1][j])) 
					if (wee[i][j] + tt <= 0):
						wee[i][j] = -99 
					else:
						wee[i][j] = wee[i][j] + tt 
						sum = sum + wee[i][j] 
			
			# ================================================================================== #
			# ========================== WEIGHT NORMALISATION (EQ 3.1) ========================= #
			# ================================================================================== #
			for j in range(0, NN_E):	
				if ((sum <> 0) and (wee[i][j] > 0.0)):
					wee[i][j] = wee[i][j] / sum 
			
			sum = 0 	
			# ================================================================================== #
			# ================== UPDATING I-E WEIGHTS BY iSTDP (EQ 3.9) ======================== #
			# ================================================================================== #
			for j in range(0, NN_I):
				if (wei[i][j] > 0):
					tt = -1.0 * etaINHIB * y[t0][j] * (1-float(x[t1][i])) * (1.0+1.0/float(muIP)) 
					if (wei[i][j] + tt <= 0):
						wei[i][j] = -99 
					else:
						wei[i][j] = wei[i][j] + tt 
						sum = sum + wei[i][j] 
						
			# ================================================================================== #
			# ========================= WEIGHT NORMALISATION (EQ 3.2) ========================== #
			# ================================================================================== #
			for j in range(0, NN_I):
				if ((sum <> 0) and (wei[i][j] > 0)):
					wei[i][j] = wei[i][j] / sum 

			# ================================================================================== #
			# ================== UPDATING THRESHOLD BY IP (EQ 3.10) ============================ #
			# ================================================================================== #
			te[i] += etaIP * (float(x[t0][i]) - muIP)  
			
			if (te[i] < 0):
				te[i] = 0 
		
		# ================================================================================== #
		# =========== ADD NEW E-E CONNECTION WITH paddEE(STRUCTURAL PLASTICITY) ============ #
		# ================================================================================== #
		if (aux01.randd() < paddEE):
			ind = 0 
			sum = 0 
			
			while (ind == 0):
				# ============== RANDOM TWO EXCITATORY NEURONS WHICH IS NOT YET CONNECTED =========== #
				r1 = aux01.randl(NN_E) 
				r2 = aux01.randl(NN_E) 
			
				if ((wee[r1][r2] == -99) and (r1 <> r2)):
					# ============= ADD NEW CONNECTION WITH WEIGHT: 0.001 =============== #
					wee[r1][r2] = 0.001 
					ind = 1 
					
					# ================================================================================== #
					# ========================= WEIGHT NORMALISATION (EQ 3.1) ========================== #
					# ================================================================================== #
					for j in range(0, NN_E):	# (RE) NORMALISATION
						if (wee[r1][j] > 0):
							sum = sum + wee[r1][j] 
					for j in range(0, NN_E):
						if ((sum <> 0) and (wee[r1][j] >= 0)):
							wee[r1][j] = wee[r1][j] / sum 
'''
======================================================================================
FUNCTION: VISUALISATION OF THE SPIKING OF THE NERUONS
======================================================================================
'''
def printImg(t):
	t1 = t % 2 
	for i in range(0, NN_E):
		if (x[t1][i] <> 0):
			img[i][t] = 0 
		else:
			img[i][t] = 255 
	for i in range(0, NN_I):
		if (y[t1][i] <> 0):
			img[i+NN_E][t] = 0 
		else:
			img[i+NN_E][t] = 255 
'''
======================================================================================
FUNCTION: CLEARING THE STRING PATTERN
======================================================================================
'''
def clearStringPattern():
	for i in range(0, N):
		stringPattern[i] = 0 
'''
======================================================================================
FUNCTION: UPDATING THE STRING PATTERN
======================================================================================
'''
def updateStringPattern(t):
	t1 = t % 2 
	for i in range(0, N):
		if (x[t1][i+120-N] == 1):
			stringPattern[i] = stringPattern[i] + 1 
'''
======================================================================================
FUNCTION: ENCODING STRING PATTERN
======================================================================================
'''
def calculateStringScore():
	for i in range(0, N):
		if (stringPattern[i] >= thresholdOn):
			stringPattern[i] = 1 
		else:
			stringPattern[i] = 0 
'''
======================================================================================
FUNCTION: CALCULATING SCORE
======================================================================================
'''
def scoreCal():
	count = 0
	stringPatternMASK = np.array(stringPattern, "b") ^ MASK

	for i in range(0, N/3):
		sum = 0
		for j in range(3*i, 3*i+3):
			sum = sum + stringPatternMASK[j]
		
		if sum == 3:
			count = count + 1
	return count
'''
======================================================================================
FUNCTION: SAVING PARAMETERS TO VARIABLES
======================================================================================
'''
def savePara(pair):
	global savedWEE, savedWEI, savedWIE, savedTE, savedTI
	savedWEE[pair] = np.copy(wee)
	savedWEI[pair] = np.copy(wei)
	savedWIE[pair] = np.copy(wie)
	savedTE[pair] = np.copy(te)
	savedTI[pair] = np.copy(ti)
'''
======================================================================================
FUNCTION: LOADING PARAMETERS FROM VARIABLES
======================================================================================
'''	
def loadPara(pair):
	global wee, wei, wie, te, ti
	wee = np.copy(savedWEE[pair])
	wei = np.copy(savedWEI[pair])
	wie = np.copy(savedWIE[pair])
	te = np.copy(savedTE[pair])
	ti = np.copy(savedTI[pair])
'''
======================================================================================
FUNCTION: CLEARING THE NEURONS ACRIVITIES
======================================================================================
'''		
def clearActivity():
	global x, y

	for i in range(0, 2):
		for j in range(0, NN_E):
			x[i][j] = 0
	
	for i in range(0, 2):
		for j in range(0, NN_I):
			y[i][j] = 0
'''
======================================================================================
FUNCTION: PERFORMING MUTATIONG ON INDIVIDUAL
======================================================================================
'''
def performMutation():
	global wee, wei, wie, te, ti

	# ============================================================ #
	# ============ MUTATING WEIGHT OF E-E CONNECTIONS ============ #
	# ============================================================ #
	for i in range(0, NN_E):
		for j in range(0, NN_E):
			if (wee[i][j] <> -99 and np.random.random() < pMut):
				wee[i][j] = wee[i][j] + np.random.normal(0, etaMut) 
				if wee[i][j] < 0.0:
					wee[i][j] = 0.0 
				if wee[i][j] > 1.0:
					wee[i][j] = 1.0 
	
	# ========== RENORMALIZATION OF THE E-E CONNECTIONS ========== #
	for i in range(0, NN_E):
		sum = 0 
		for j in range(0, NN_E):
			if (wee[i][j] <> -99):
				sum = sum + wee[i][j] 
				
		for j in range(0, NN_E):
			if (wee[i][j] <> -99 and sum <> 0):
				wee[i][j] = wee[i][j] / sum
				
	# ============================================================ #
	# ============ MUTATING WEIGHT OF I-E CONNECTIONS ============ #
	# ============================================================ #
	
	for i in range(0, NN_E):
		for j in range(0, NN_I):
			if (wei[i][j] <> -99 and np.random.random() < pMut):
				wei[i][j] = wei[i][j] + np.random.normal(0, etaMut) 
				if wei[i][j] < 0.0:
					wei[i][j] = 0
				if wei[i][j] > 1.0:
					wei[i][j] = 1.0
	
	# ========== RENORMALIZATION OF THE I-E CONNECTIONS ========== #
	for i in range(0, NN_E):
		sum = 0 
		for j in range(0, NN_I):
			if (wei[i][j] <> -99):
				sum = sum + wei[i][j] 
		for j in range(0, NN_I):
			if (wei[i][j] <> -99 and sum <> 0):
				wei[i][j] = wei[i][j] / sum
				
	# ============================================================ #
	# ============ MUTATING WEIGHT OF E-I CONNECTIONS ============ #
	# ============================================================ #
	
	for i in range(0, NN_I):
		for j in range(0, NN_E):
			if (wie[i][j] <> -99 and np.random.random() < pMut):
				wie[i][j] = wie[i][j] + np.random.normal(0, etaMut) 
				if wie[i][j] < 0:
					wie[i][j] = 0 
				if wie[i][j] > 1.0:
					wie[i][j] = 1.0 
	
	# ========== RENORMALIZATION OF THE E-I CONNECTIONS ========== #
	for i in range(0, NN_I):
		sum = 0 
		for j in range(0, NN_E):
			if (wie[i][j] <> -99):
				sum = sum + wie[i][j] 
		for j in range(0, NN_E):
			if (wie[i][j] <> -99 and sum <> 0):
				wie[i][j] = wie[i][j] / sum
				
	# ============================================================ #
	# ============== MUTATING EXCITATORY THRESHOLDS ============== #
	# ============================================================ #	
				
	for i in range(0, NN_E):
		if (np.random.random() < pMut):
			te[i] = te[i] + np.random.normal(0, etaMut) 
			if te[i] < 0:
				te[i] = 0 
			if te[i] > TE_MAX:
				te[i] = TE_MAX 
				
	# ============================================================ #
	# ============== MUTATING INHIBITORY THRESHOLDS ============== #
	# ============================================================ #	
	for i in range(0, NN_I):
		if (np.random.random() < pMut):
			ti[i] = ti[i] + np.random.normal(0, etaMut) 
			
			if ti[i] < 0:
				ti[i] = 0 
			if ti[i] > TI_MAX:
				ti[i] = TI_MAX 
				
'''
======================================================================================
======================================================================================
======================================================================================
							MAIN PROGRAM STARTS HERE
======================================================================================
======================================================================================
======================================================================================
'''
# ================== GENERATING SEED ==================== #
aux01.seed(500)

# ================== INITIALIZING SORN MODEL =============== #
init()

# ================== INITIALIZING PRIMITIVES =============== #
pair = 0	
st = 0

# ================== SAVE INTIALIZED PARAMETERS ============= #
savePara(2)

'''
===================================================
============== ALGORITHM STARTS HERE ==============
===================================================
'''

for time in range(1, STEPS):
	# ============== ASSIGN ACTIVITIES FOR INPUT NEURONS =============== #
	for i in range(0, 100):
		inp[i] = 0
		
	inp[st % 10] = strength
		
	# ============== SAVE PARAMETERS AND PERFORM MUTATION ============== #
	if st == 0 and pair == 0:
		savePara(0)
			
	if st == 0 and pair == 1:
		performMutation()
		savePara(1)

	# ============== PERFORM ONE STEP OF THE SORN MODEL ================ #
	step(time)

	# ============== INCREASE STATE COUNT BY ONE AND UPDATING STRING PATTERN ========== #
	st = st + 1
	updateStringPattern(time)

	# ============== WHEN ONE TRIAL IS FINISHED ============= #
	if st == stateIteration:
		st = 0
		
		# ============= CALCULATING FITNESS OF THAT GENERATION ============ #
		calculateStringScore()
		score[pair] = scoreCal()
		scoreFig.append(score[pair])

		# ============ RESET STRING PATTERN AND ACTIVITIES ============= #
		clearStringPattern()
		clearActivity()

		pair = pair + 1

		if pair == 1:
			# ====== IF FIRST TRIAL, LOAD PARAMETERS FROM THE MOTHER PARAMETERS ======= #
			loadPara(2)
			
		if pair == 2:
			pair = 0

			# ====== IF SECOND TRIAL, COMPARING FITNESS TO THE FIRST TRIAL ====== #
			if score[0] >= score[1] and score[0] >= score[2]:
				score[2] = score[0]
				loadPara(0)
				savePara(2)
			elif score[1] >= score[0] and score[1] >= score[2]:
				score[2] = score[1]
				loadPara(1)
				savePara(2)
			else:
				loadPara(2)	

	# ================ VISUALISE THE RESULT ON THE GRAPH =================== #
	if (VISUAL and time >= STEPS - SAVE_STEPS):    # VISUAL THE RESULT ON THE DIAGRAM
		printImg(time - STEPS + SAVE_STEPS) 

# ========== SAVE THE FITNESS ALONG EACH GENERATIONS TO A TEXT FILE =========== #
np.savetxt('fitness_evolution.txt', scoreFig)


# ========== SHOW VISUALISATION ======= #
plt.figure(1)
plt.imshow(img, aspect="auto")
plt.gray()

plt.figure(2)
plt.plot(scoreFig)
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
