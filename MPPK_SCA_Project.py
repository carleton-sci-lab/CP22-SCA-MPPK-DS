from ast import Index
from numpy import random
import numpy as np
#Finite Field Index
p=257
#Euler's totient of p
tp = p-1
#number of noise varibles
m=2
#degree of base polynomials
n=2
#degree of multipler polynomials
lam=1
#upper limits
ell=[1,1]
# key generation algorithm
def K(n,lam,ell,p):
   # m = number of noise variables
   # n = degree of base polynomial
   # lam = degree of univariate polynomials
   # ell = upper limits, in base polynomial
   # p = finite field index
   tp = p - 1 # Euler's totient of p
   ## Base Polynomial
   # randomly generate coefficients for beta()
   c={random.randint(low=0,high=tp-1,size=(n+1,(ell(1)+1)*(ell(2)+1)))}
   # c = zeros(n+1, (ell(1)+1)*(ell(2)+1) );
   ## Univariate ploynomials and
   # randomly generate coefficients of f()
   f={random.randint(0, high=tp-1,size=(1,lam+1))}
   # randomly generate coefficients of h()
   h={random.randint(0,high=tp-1, size=(1,lam+1))}
   ## Product polynomials and
   # init \phi and \psi to zeros
   phi = np.zeros(n+lam+1, (ell(1)+1)*(ell(2)+1))
   psi = np.zeros(n+lam+1, (ell(1)+1)*(ell(2)+1))
   i=0
   for i in n:
      j=0
      for j in lam:
         
        phi[i+j+1] = phi[i+j+1] + np.multiply(c[i+1],f(j+1))
        psi[i+j+1] = psi[i+j+1] + np.multiply(c[i+1],h(j+1))
   
   phi = phi % tp
   psi = psi % tp
   ## Polynomials and
   # randomly generate coefficients
   Ephi={random.randint(low=0, high=(tp-1), size=(1,n+lam-1))}
   # randomly generate coefficients 
   Epsi={random.randint(0 ,high=tp-1, size=(1,n+lam-1))}
   ## , and
   R0 = (random.randint(1,(tp-2)/2,1))*2
   Rn = (random.randint(1,(tp-2)/2,1))*2
   ## Noise functions and
   N0 = R0 * c[1]% tp
   Nn = Rn * c[n+1]% tp
   ## and
   Phi = phi[2:n+lam]
   Psi = psi[2:n+lam]
   ## and
   P = np.zeros(n+lam-1,(ell(1)+1)*(ell(2)+1))
   Q = np.zeros(n+lam-1,(ell(1)+1)*(ell(2)+1))
   i=1
   for i in(n+lam-1):
      
    
      P[i]= R0 * np.multiply((Phi[i,1]-Ephi[i]) , Phi[i])
      Q[i] = Rn * np.multiply((Psi[i,1]-Epsi[i]),Psi[i])
 
   P = P%tp
   Q = Q%tp
   #Private-key and public-key pair
   s = { f, h, R0, Rn ,Ephi, Epsi }
   v = { P, Q, N0, Nn }
   return s,v

##Signing Algorithm
def S(s,mu,lam,p):
   #t = digital signature
   # mu = message
   # s = private key
   f = s(1).copy
   h = s(2).copy
   R0 = s(3).copy
   Rn = s(4).copy
   Ephi = s(5).copy 
   Epsi = s(6).copy
   # m = number of noise variables
   # n = degree of base polynomial
   # lam = degree of univariate polynomials
   # ell = upper limits, in base polynomial
   # p = finite field index
   tp = p - 1 # Euler's totient of p
   #Random base
   g = random.randint(2, tp-1,1)
   #Evaluate on
   fm = np.polyval(np.flip(f),mu)
   a = R0*fm % tp
   A = (g**a)%p
   #Evaluate on
   hm = np.polyval(np.flip(h),mu)
   b = Rn*hm % tp
   B = (g**b)%p
   c = Rn * ( hm*f(1) - fm*h(1) )% tp
   C = g**c%p
   d = R0 * ( hm*f(lam+1) - fm*h(lam+1) )%tp
   D = g**d%p
   #Evaluate on
   flip=np.flip(Ephi)
   flip.append(0)
   Ephim = np.polyval(flip,mu)
   Ephim = Ephim % tp
   #Evaluate on
   flip2=np.flip(Ephi)
   flip2.append(0)
   Epsim = np.polyval(flip2,mu)
   Epsim = Epsim % tp
   e = R0*Rn*(hm*Ephim - fm*Epsim) % tp
   E = (g**e)%p
   # digital signature
   t = {A, B, C, D, E }
   return mu,t

#Signature Verifying Algorithm
def V(v,mu,t,m,n,lam,ell,p):
   # v = public key
   P = v(1).copy 
   Q = v(2).copy
   N0 = v(3).copy
   Nn = v(4).copy
   # mu = message
   # t = digital signature
   A = t(1).copy
   B = t(2).copy
   C = t(3).copyj
   D = t(4).copy
   E = t(5).copy
   # m = number of noise variables
   # n = degree of base polynomial
   # lam = degree of univariate polynomials
   # ell = upper limits, in base polynomial
   # p = finite field index
   tp = p - 1 # Euler's totient of p
   #Noise variables
   r={random.randint(1, high=tp-1,size=(1,m))}
   # r = [2 2]
   #Evaluate , , and
   barP = 0; barQ = 0 
   i=1
   for i in(n+lam-1):
      j_1=0
      for j_1 in ell(1):
         j_2=0
         for j_2 in ell(2):
            barP = (barP + P(i,(j_1*2)+j_2+1)*((r[1]**j_1)%tp)*((r[2]**j_2)%tp)*((mu**i)%tp))%tp
            barQ = (barQ + Q(i,(j_1*2)+j_2+1)*((r[1]**j_1)%tp)*((r[2]**j_2)%tp)*((mu**i)%tp))%tp

   barP = barP % tp
   barQ = barQ % tp
   barN0 = 0
   barNn = 0
   j_1=0
   for j_1 in ell(1):
      j_2=0
      for j_2 in ell(2):
         barN0 = (barN0 + N0((j_1*2)+j_2+1)*((r(1)**j_1)% tp)*((r(2),j_2)%tp))%tp
         barNn = (barNn + Nn((j_1*2)+j_2+1)*((r(1)**j_1) % tp)*((r(2),j_2)%tp))%tp

   barN0 = barN0%tp
   barNn = (barNn*(mu**(n+lam)) %tp)%tp 
   # Verification
   left = (A**barQ) %p
   right = ((B**barP) % p * (C**barN0) % p *(D**barNn) % p*E ) % p
   if left==right:
      verdict = 'VALID'
   else:
      verdict = 'INVALID'

   return mu, verdict
   
def MPRKDS(m,n,lam,ell,p):
   #Random message
   mu = random.randint(0,p-1,1)
   # disp(mu)
   [s,v] = K(m,n,lam,ell,p)
   # disp(s)
   # disp(v)
   [mu,t] = S(s,mu,m,n,lam,ell,p)
   print(t) # [A B C D E]
   # A = t(1); B = t(2); C = t(3); D = t(4); E = t(5);
   # if (A*B*C*D*E)==0 | ((A-1)*(B-1)*(C-1)*(D-1)*(E-1))==0
   # fprintf('A, B, C, D and E must not be equal to 0 or 1');
   # return;
   # else
   [mu,verdict] = V(v,mu,t,m,n,lam,ell,p)
   print(verdict)
   # end
   return verdict