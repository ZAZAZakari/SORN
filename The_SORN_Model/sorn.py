'''
======================================================================================
DESCRIPTION: IMPLEMENTION OF THE SORN MODEL
DATE: 5 JAN 2014
======================================================================================
'''
# =========================== IMPORTING ESSENTIAL LIBRARIES ========================== #
import aux01
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================ INITIALIZING CONSTANTS ================================ #
NN_E = 120				# [INTEGER] NUMBER OF EXCITATORY NEURONS IN THE MODEL
NN_I = 24				# [INTEGER] NUMBER OF INHIBITORY NERUONS IN THE MODEL = 0.2*NN_E
NUM_I = 100				# [INTEGER] NUMBER OF THE EXTERNAL INPUT NEURONS  
						# THE INPUT NEURONS ARE THE FIRST NUM_I EXCITATORY NEURONS
						
TE_MAX = 1.0			# [FLOATING NUMBER] MAXIMUM OF EXCITATORY THRESHOLD
TI_MAX = 0.5			# [FLOATING NUMBER] MAXIMUM OF INHIBITORY THRESHOLD

sig2 = 0.001			# [FLOATING NUMBER] SD OF RANDOM NOISE ON EXCITATORY NEURONS
muIP = 0.1				# [FLOATING NUMBER] THE RATE OF NEURONS FIRING
etaIP = 0.01			# [FLOATING NUMBER] THE STRENGTH OF EXCITATORY THRESHOLD PLASTICITY
etaSTDP = 0.004			# [FLOATING NUMBER] THE STRENGTH OF STDP
etaINHIB = 0.001		# [FLOATING NUMBER] THE STRENGTH OF INHIBITORY STDP

						
pEE = 0.1				# [PROBABILITY] OF INITIAL E --> E CONNECTIONS
pEI = 1.0				# [PROBABILITY] OF INITIAL I --> E CONNECTIONS
paddEE = 0.3			# [PROBABILITY] OF ADDING A NEW E --> E CONNECTION

SAVE_STEPS = 2000		# [INTEGER] NUMBER OF TIME STEPS TO BE SAVED
STEPS = 5000			# [INTEGER] TOTAL NUMBER OF STEPS IN THIS RUN

fix = 0					# [BOOLEAN] TO SWITCH ON AND OFF PLASTICITY
VISUAL = 1				# [BOOLEAN] TO SWITCH ON AND OFF VISUALISATION

# =========================== INITIALIZING ARRAYS ================================ #
te = np.zeros(NN_E)					# [1D ARRAY] THRESHOLD OF ALL EXCITATORY NEURONS
ti = np.zeros(NN_I)					# [1D ARRAY] THRESHOLD OF ALL INHIBITORY NEURONS
inp = np.zeros(NUM_I)				# [1D ARRAY] ACTIVITY STATES OF EXTERNAL NEURONS
x = np.zeros(shape=(2, NN_E))		# [2D ARRAY] ACTIVITY STATES OF EXCITATORY NEURONS
y = np.zeros(shape=(2, NN_I))		# [2D ARRAY] ACTIVITY STATES OF INHIBITORY NEURONS

wee = np.zeros(shape=(NN_E, NN_E))				# [2D ARRAY] WEIGHTS OF ALL E ---> E CONNECTIONS
wei = np.zeros(shape=(NN_E, NN_I))				# [2D ARRAY] WEIGHTS OF ALL E ---> I CONNECTIONS
wie = np.zeros(shape=(NN_I, NN_E))				# [2D ARRAY] WEIGHTS OF ALL I ---> E CONNECTIONS

img = np.zeros(shape=(NN_E+NN_I, SAVE_STEPS))	# [2D ARRAY] SPIKING OF THE NEURONS OVER TIME
'''
======================================================================================
FUNCTION: INITIALIZING ALL THE VARIABLES 
======================================================================================
'''
def init():
	# ================= GLOBALIZE VARIABLES ====================== #
	global te, ti, wee, wei, wie, x, y
	
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
	
	# ========== INITIALIZING THE STATE OF INHIBITORY NEURONS TO ZERO ========== #
	for i in range(0, NN_I):
		y[0][i] = 0 
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
	for i in range(0, NN_I):	
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
					for j in range(0, NN_E):	
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

for time in range(1, STEPS):
	# ============== ASSIGN ACTIVITIES FOR INPUT NEURONS =============== #
	for i in range(0, NUM_I):
		inp[i] = 1
	
	# ============== PERFORM ONE STEP OF THE SORN MODEL ================ #
	step(time)
	
	# ================ VISUALISE THE RESULT ON THE GRAPH =================== #
	if (VISUAL and time >= STEPS - SAVE_STEPS):
		printImg(time - STEPS + SAVE_STEPS) 

# ========== SHOW VISUALISATION ======= #
plt.figure(1)
plt.imshow(img, aspect="auto")
plt.gray()
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