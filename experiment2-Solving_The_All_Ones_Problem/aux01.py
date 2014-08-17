import math;

IA = 16807
IM = 2147483647
AM = (1.0 / IM)
IQ = 127773
IR = 2836
NTAB = 32
NDIV = (1 + (IM - 1) / NTAB )
EPS = 1.2 * (10 ** -7)
RNMX = (1.0 - EPS)

idum = -4533; 	# SEED OF THE GAUSSIAN RANDOM NUMBER GENERATOR
iy = 0;
iv = [0]*NTAB;
iset = 0;
gset = 0;

def ran1():
	global idum;
	global iy;
	global iv;

	if (idum <= 0 or iy == 0):
		if (-idum < 1):
			idum = 1;
		else:
			idum = (-1) * idum;

		j1 = NTAB + 7;
		while (j1 >= 0):
			k = idum / IQ;
			idum = IA * (idum - k * IQ) - IR * k;
		
			if (idum < 0):
				idum = idum + IM;
			if (j1 < NTAB):
				iv[j1] = idum;
			
			j1 = j1 - 1;
		
		iy = iv[0];


	k = idum / IQ;
	idum = IA * (idum - k * IQ) - IR * k;

	
	if (idum < 0):
		idum = idum + IM;

	j = iy / NDIV;
	iy = iv[j];
	iv[j] = idum;
	
	temp = AM * iy;

	if (temp > RNMX):
		return RNMX;
	else:
		return temp;

def gasdev():
	global idum;
	global iset;
	global gset;

	if (iset == 0):
		while True:
			v1 = 2.0 * ran1() - 1.0;
			v2 = 2.0 * ran1() - 1.0;

			rsq = v1 * v1 + v2 * v2;			

			if not(rsq >= 1.0 or rsq == 0.0):
				break

		fac = math.sqrt(-2.0 * math.log(rsq) / rsq);
		gset = v1 * fac;
		iset = 1;
		return (v2 * fac);
	else:
		iset = 0;
		return (gset);

		

AA = 471
B = 1586
CC = 6988
DD = 9689
M = 16383
RIMAX = 2147483648.0

ra = [0.0] * (M+1);
nd = 0;

def seed(sd):
	if (sd < 0):
		print "SEED ERROR";
	
	ra[0] = math.fmod(16807.0 *sd, 2147483647.0);
	
	for i in range(1, M+1):
		ra[i] = math.fmod(16807.0 * ra[i-1], 2147483647.0);

def RandomInteger():
	global nd;

	nd = nd + 1;
		
	ra[nd & M] = int(ra[(nd-AA) & M]) ^ int(ra[(nd-B) & M]) ^ int(ra[(nd-CC) & M]) ^ int(ra[(nd-DD) & M]);		
	return ra[nd & M];

def randl(num):
	return (RandomInteger() % num);

def randd():
	return (float(RandomInteger()) / RIMAX);
