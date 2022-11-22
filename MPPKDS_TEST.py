# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 19:17:01 2022

@author: Owner
"""

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
#upper limit
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
   #OLDLINE: c=np.random.randint(low=0,high=tp-1,size=(n+1,(ell[0]+1)*(ell[1]+1)))
   c = np.random.randint(0, tp - 1, (n + 1, (ell[0]+1)*(ell[1]+1)))
   # c = zeros(n+1, (ell(1)+1)*(ell(2)+1) );
   ## Univariate ploynomials and
   # randomly generate coefficients of f()
   f=np.random.randint(0, high=tp-1, size=(lam+1))
   # randomly generate coefficients of h()
   h=np.random.randint(0, high=tp-1, size=(lam+1))
   ## Product polynomials and
   # init \phi and \psi to zeros
   phi = np.zeros([n+lam+1, (ell[0]+1)*(ell[1]+1)])
   psi = np.zeros([n+lam+1, (ell[0]+1)*(ell[1]+1)])
   #OLD LOOP
   #i=0
   #for i in range(n):
   #   j=0
   #   for j in range(lam):
   #     phi[i+j+1] = phi[i+j+1] + np.multiply(c[i+1],f[j][1])
   #     psi[i+j+1] = psi[i+j+1] + np.multiply(c[i+1],h[j][1])
   for i in range(0, n + 1):
      for j in range(0, lam + 1):
        r1 = [item * f[j] for item in c[i]]
        #print("R1:")
        #print(r1);
        r2 = [item * h[j] for item in c[i]]
        phi[i+j] = phi[i+j] + r1
        psi[i+j] = psi[i+j] + r2
        #print("Phi and psi after iteration")
        print(phi)
        #print(psi)
   
   phi = phi % tp
   psi = psi % tp
   print("Test Below")
   print(phi)
   print(psi)
   ## Polynomials and
   # randomly generate coefficients
   Ephi=np.random.randint(low=0, high=(tp-1), size=(1,n+lam-1))
   # randomly generate coefficients 
   Epsi=np.random.randint(0 ,high=tp-1, size=(1,n+lam-1))
   ## , and
   R0 = (np.random.randint(1,(tp-2)/2,1))*2
   Rn = (np.random.randint(1,(tp-2)/2,1))*2
   ## Noise functions and
   
   N0 = R0 * c[0]% tp
   Nn = Rn * c[n] % tp
   ## and
   Phi_total = phi[2:n+lam]
   Psi_total = psi[2:n+lam]
   ## and
   
   P = np.zeros((n+lam-1,(ell[0]+1)*(ell[1]+1)))
   Q = np.zeros((n+lam-1,(ell[0]+1)*(ell[1]+1)))
   i=0
  
   for i in range(n+lam-1):
      #print(Phi_total[i-1])
      el0=Phi_total[i-1][0]
      Phi_total[i-1]=np.insert(np.delete(Phi_total[i-1],[0]),0,(el0-Ephi[0][i]))
      #print(Phi_total)
      P[i]= R0 @ Phi_total
      #print(P[i])
      el1=Psi_total[i-1][0]
      Psi_total[i-1]=np.insert(np.delete(Psi_total[i-1],[0]),0,(el1-Epsi[0][i]))
      Q[i] = Rn @ Psi_total
 
   P = P%tp
   Q = Q%tp
   #Private-key and public-key pair
   s = [f, h, R0, Rn, Ephi, Epsi]
   v = [ P, Q, N0, Nn ]
   return s,v

##Signing Algorithm
def S(s,mu,lam,p):
   #t = digital signature
   t=np.empty(5,dtype=object) 
   # mu = message
   # s = private key
   f = np.copy(s[0])
   h = np.copy(s[1])
   R0 = np.copy(s[2])
   Rn = np.copy(s[3])
   Ephi = np.copy(s[4]) 
   Epsi = np.copy(s[5])
   
   tp = p - 1 # Euler's totient of p
   #Random base
   g = np.random.randint(2, tp-1,1)
   #Evaluate f(x) on mu
   fm = np.polyval(np.flip(f),mu)
   #A
   a = R0*fm % tp
   t[0] = (g**a)%p
   #Evaluate h(x) on mu
   hm = np.polyval(np.flip(h),mu)
   #B
   b = Rn*hm % tp
   t[1] = (g**b)%p
   #C
   c = Rn * (hm*f[0] - fm*h[0] )% tp
   t[2] = g**c%p
   #D
   d = R0 * (hm*f[lam-1] - fm*h[lam-1])%tp
   t[3] = g**d%p
   #Evaluate Ephi
   flip=np.flip(Ephi)
   np.insert(flip,flip.size,0)
   Ephim = np.polyval(flip,mu)%tp
   #Evaluate Epsi
   flip2=np.flip(Epsi)
   np.insert(flip2,flip2.size,0)
   Epsim = np.polyval(flip2,mu)% tp
   #E
   e = R0*Rn*(hm*Ephim - fm*Epsim) % tp
   t[4] = (g**e)%p
   return mu,t

#Signature Verifying Algorithm
def V(v,mu,t,m,n,lam,ell,p):
   # v = public key
   P = np.copy(v[0])
   Q = np.copy(v[1])
   N0 = np.copy(v[2])
   Nn =np.copy(v[3])

   A = np.copy(t[0])
   B = np.copy(t[1])
   C = np.copy(t[2])
   D = np.copy(t[3])
   E =np.copy(t[4])
   
   tp = p - 1 # Euler's totient of p
   #Noise variables
   r=np.random.randint(1, high=tp-1,size=(1,m))
   r=r[0]

   #Evaluate
   barP = 0; barQ = 0 
   i=0
   for i in range(n+lam-1):
      j_1=0
      for j_1 in range(ell[0]):
         j_2=0
         for j_2 in range(ell[1]):
            barP = (barP + P[i][(j_1*2)+j_2+1]*((r[0]**j_1)%tp)*((r[1]**j_2)%tp)*((mu**i)%tp))%tp
            barQ = (barQ + Q[i][(j_1*2)+j_2+1]*((r[0]**j_1)%tp)*((r[1]**j_2)%tp)*((mu**i)%tp))%tp

   barP = barP % tp
   barQ = barQ % tp
   barN0 = 0
   barNn = 0
   for j_1 in range(ell[0]):
      for j_2 in range(ell[1]):
         barN0 = (barN0 + N0[(j_1*2)+j_2+1]*((r[0]**j_1)% tp)*((r[1]**j_2)%tp))%tp
         barNn = (barNn + Nn[(j_1*2)+j_2+1]*((r[0]**j_1) % tp)*((r[1]**j_2)%tp))%tp

   barN0 = barN0%tp
   barNn = (barNn*(mu**(n+lam)) %tp)%tp 
   # Verification
   left = (A**barQ) %p
   right = (((B**barP) % p) * ((C**barN0) % p) * (((D**barNn) % p)*E )) % p
   if np.array_equal(left,right):
      verdict = 'VALID'
   else:
      verdict = 'INVALID'

   return mu, verdict
   
def MPPKDS(m,n,lam,ell,p):
   #np message
   mu = np.random.randint(0,p-1,1)
   # disp(mu)
   [s,v] = K(n,lam,ell,p)
   # disp(s)
   # disp(v)
   [mu,t] = S(s,mu,lam,p)
   #print(t[0][1])
   print("t value:",t) # [A B C D E]
   print(t[0])
   # A = t(1); B = t(2); C = t(3); D = t(4); E = t(5);
   # if (A*B*C*D*E)==0 | ((A-1)*(B-1)*(C-1)*(D-1)*(E-1))==0
   # fprintf('A, B, C, D and E must not be equal to 0 or 1');
   # return;
   # else
   [mu,verdict] = V(v,mu,t,m,n,lam,ell,p)
   print("verdict is:",verdict)
   # end
   return verdict

def main():
    j=0
    i=1
    for i in range(100):
        verdict=MPPKDS(m,n,lam,ell,p)
        if verdict=='INVALID':
           print('iteration',i)
           j=j+1
        print("\n")
    print("\n valid passes:",j)
    print("\n")

if __name__ == '__main__':
    main()