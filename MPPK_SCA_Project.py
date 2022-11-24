from math import expm1
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
 
   #c=[[153, 89, 136, 12],[77, 28, 54, 113],[86, 134, 60, 180]]
   
   #f=[117,51]
   #h=[230,96]
   
   # randomly generate coefficients of f()
   c=np.random.randint(low=0,high=tp-1,size=(n+1,(ell[0]+1)*(ell[1]+1)))
   print("c value",c)
   f=np.random.randint(0, high=tp-1,size=(1,lam+1))
   f=list(f[0]) 
   print(f)
   h=np.random.randint(0,high=tp-1, size=(1,lam+1))
   h=list(h[0])
   print(h)
   ## Product polynomials and init \phi and \psi to zeros
   phi = np.zeros([n+lam+1, (ell[0]+1)*(ell[1]+1)])
   psi = np.zeros([n+lam+1, (ell[0]+1)*(ell[1]+1)])
        
   for i in range(n+1):      
      for j in range(lam+1):
        phi[i+j] = phi[i+j] +[item * f[j] for item in c[i]]       
        psi[i+j] = psi[i+j] +[item * h[j] for item in c[i]] 
   phi = np.mod(phi,tp)
   psi = np.mod(psi,tp)
  
   # randomly generate coefficients
   Ephi=np.random.randint(low=0, high=(tp-1), size=(1,n+lam-1))
   Ephi=Ephi[0]
   Epsi=np.random.randint(0 ,high=tp-1, size=(1,n+lam-1))
   Epsi=Epsi[0]
   R0 = (np.random.randint(1,(tp-2)/2,1))*2
   Rn = (np.random.randint(1,(tp-2)/2,1))*2
   print(Ephi)
   #Ephi=[127,248]
   #Epsi=[237,4]   
   #R0=52
   #Rn=144 
   
   ## Noise functions and
   N0=[]
   Nn=[]
  
   for k in c[0]:N0.append((R0*k)%tp)
   for m in c[n]:Nn.append((Rn*m)%tp)
   
   P = [[0 for col in range((ell[0]+1)*(ell[1]+1))] for row in range((n+lam-1))]
   Q = [[0 for col in range((ell[0]+1)*(ell[1]+1))] for row in range((n+lam-1))]
      
   for i in range((n+lam)-1):         
      phi[i+1][0]=phi[i+1][0]-Ephi[i]
      P[i]= R0 * phi[i+1]         
      psi[i+1][0]=psi[i+1][0]-Epsi[i]
      Q[i]= Rn * psi[i+1] 
   P = np.mod(P,tp)
   Q = np.mod(Q,tp)
   #Private-key and public-key pair
   s =  [f, h, R0,Rn ,Ephi, Epsi]
   v =  [ list(P),list (Q),list( N0),list(Nn )]
   return s,v

##Signing Algorithm
def S(s,mu,lam,p):
   #t = digital signature
   t=np.zeros(5) 
   print('t',t)
   # mu = message
   # s = private key
   f = np.copy(s[0])
   h = np.copy(s[1])
   R0 = int(np.copy(s[2]))
   Rn = int(np.copy(s[3]))
   Ephi = np.copy(s[4]) 
   Epsi = np.copy(s[5])
   # m = number of noise variables
   # n = degree of base polynomial
   # lam = degree of univariate polynomials
   # ell = upper limits, in base polynomial
   # p = finite field index
   tp = p - 1 # Euler's totient of p
   #Random base
   g = np.random.randint(2, tp-1,1)
   print("g",g)
   #Evaluate on
   fm = np.polyval(np.flip(f),mu)
   
   #A
   a = R0*fm % tp
   print("inside data ",a," ",g," ",fm )
   t[0] = pow(int(g[0]),int(a[0]),p)
   #Evaluate on
   hm = int(np.polyval(np.flip(h),mu))
   #B
   b= (Rn*hm)%tp
   t[1] = pow(int(g[0]),b,p)
   print(" t1 ",t[1])
   #C
   c = Rn * (hm*f[0] - fm*h[0] )% tp
   t[2] = pow(int(g[0]),int (c[0]),p)
   #D
   d = R0 * (hm*f[lam-1] - fm*h[lam-1])%tp
   t[3] = pow(int(g[0]),int(d[0]),p)
   #Evaluate on
   flip=np.flip(Ephi)
   flip=np.append(flip,0)
   Ephim = np.polyval(flip,mu)%tp
   #Evaluate on
   flip2=np.flip(Epsi)
   flip2=flip2[0]
   
   flip2=np.append(flip2,0)
   #np.insert(flip2,flip2.size,0)
   
   
   Epsim = np.polyval(flip2,mu)% tp
   print("Epsim",hm*Epsim)
   #E
   e = (R0*Rn*(hm*Ephim - fm*Epsim)) % tp
  
   t[4] = pow(int(g[0]),int(e[0]),p)
   return mu,t

#Signature Verifying Algorithm
def V(v,mu,t,m,n,lam,ell,pp):
   # v = public key
   P = np.copy(v[0])
   Q = np.copy(v[1])
   N0 = np.copy(v[2])
   Nn =np.copy(v[3])
   # mu = message
   # t = digital signature
   A = np.copy(t[0])
   B = np.copy(t[1])
   C = np.copy(t[2])
   D = np.copy(t[3])
   E =np.copy(t[4])
   # m = number of noise variables
   # n = degree of base polynomial
   # lam = degree of univariate polynomials
   # ell = upper limits, in base polynomial
   # p = finite field index
   tp = pp - 1 # Euler's totient of p
   #Noise variables
   r=np.random.randint(1, high=tp-1,size=(1,m))
   r=r[0]
   # r = [2 2]
   #Evaluate , , and
   barP = 0; barQ = 0 
   i=1
   for i in range(n+lam-1):      
      for j_1 in range(ell[0]):         
         for j_2 in range(ell[1]):
            barP = (barP + P[i][(j_1*2)+j_2+1]*(int(r[0])*j_1%tp)*(int(r[1])*j_2%tp)*(int(mu)*i%tp))%tp
            barQ = (barQ + Q[i][(j_1*2)+j_2+1]*(int(r[0])*j_1%tp)*(int(r[1])*j_2%tp)*(int(mu)*i%tp))%tp

   barP = barP % tp
   barQ = barQ % tp
   barN0 = 0; barNn = 0
   for j_1 in range(ell[0]):
      for j_2 in range(ell[1]):
         barN0 = (barN0 + N0[(j_1*2)+j_2+1]*(int(r[0])*j_1%tp)*(int(r[1])*j_2%tp))%tp
         barNn = (barNn + Nn[(j_1*2)+j_2+1]*(int(r[0])*j_1%tp)*(int(r[1])*j_2%tp))%tp

   barN0 = barN0%tp
   barNn = (barNn*(int(mu)*(n+lam)%tp))%tp 
   # Verification
   left = (int(A)*int(barQ)%pp)
   right = ((int(B)*int(barP)%pp) * (int(C)*int(barN0)% pp) *(int(D)*int(barNn)%pp)*E ) % pp
   if np.array_equal(left,right):
      verdict = 'VALID'
   else:
      verdict = 'INVALID'

   return mu, verdict
   
def MPPKDS(m,n,lam,ell,p):
   #np message
   mu = np.random.randint(0,p-1,1)
   print("mu ",mu)
   [s,v] = K(n,lam,ell,p)
   # disp(s)
   # disp(v)
   [mu,t] = S(s,mu,lam,p)   
   print("t value:",t) # [A B C D E]
  
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
    k=0
    i=1
    for i in range(100):
        verdict=MPPKDS(m,n,lam,ell,p)
        if verdict=='INVALID':
           print('iteration',i)
           j=j+1
        print("\n")
        k=k+1
    print("\n invalid passes:",k-j,"/",k)
    print("\n")

def test_K():

    s,v=K(n,lam,ell,p)
    print(s,"\n")
    print(v)
   

if __name__ == '__main__':
    main()
