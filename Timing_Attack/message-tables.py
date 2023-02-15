from math import expm1
import numpy as np
import numpy.core.numeric as NX
import sqlite3
import time


# Connect to the database
conn = sqlite3.connect('data_time.db')
cursor = conn.cursor()

# Create tables to store the data
cursor.execute("""CREATE TABLE IF NOT EXISTS fm_message_table (mu INTEGER, execution_time REAL)""")
cursor.execute("""CREATE TABLE IF NOT EXISTS hm_message_table (mu INTEGER, execution_time REAL)""")

# Finite Field Index

p = 257

# Euler's totient of p

tp = p-1

# number of noise varibles

m = 2

# degree of base polynomials

n = 2

# degree of multipler polynomials

lam = 1

# upper limit

ell = [1, 1]

# key generation algorithm

def K(n, lam, ell, p):

    tp = p - 1  # Euler's totient of p

    # c=[[16,201,31,65],[190,228,75,225],[30,136,83,246]]

    f=[11,207]

    h=[32,119]

    # Ephi=[175,227]

    # Epsi=[217,151]

    # R0=[52]

    # Rn=[176]

    # randomly generate coefficients of f()

    c = np.random.randint(low=0, high=tp-1, size=(n+1, (ell[0]+1)*(ell[1]+1)))

    # print("c value", list(c))

    #f = np.random.randint(0, high=tp-1, size=(1, lam+1))

    #f = list(f[0])

    #h = np.random.randint(0, high=tp-1, size=(1, lam+1))

    #h = list(h[0])

    # print("f=", f, "h=", h)

    # Product polynomials and init \phi and \psi to zeros

    phi = np.zeros([n+lam+1, (ell[0]+1)*(ell[1]+1)])

    psi = np.zeros([n+lam+1, (ell[0]+1)*(ell[1]+1)])

    for i in range(n+1):

        for j in range(lam+1):

            phi[i+j] = phi[i+j] + [item * f[j] for item in c[i]]

            psi[i+j] = psi[i+j] + [item * h[j] for item in c[i]]

    phi = np.mod(phi, tp)

    psi = np.mod(psi, tp)

    # randomly generate coefficients

    Ephi = np.random.randint(low=0, high=(tp-1), size=(1, n+lam-1))

    Ephi = Ephi[0]

    Epsi = np.random.randint(0, high=tp-1, size=(1, n+lam-1))

    Epsi = Epsi[0]

    R0 = (np.random.randint(1, (tp-2)/2, 1))*2

    Rn = (np.random.randint(1, (tp-2)/2, 1))*2

    # print("Ephi=", Ephi, "Epsi=", Epsi, "R0=", R0, "Rn=", Rn)

    N0 = []
    Nn = []

    for k in c[0]:
        N0.append((R0[0]*k) % tp)

    for m in c[n]:
        Nn.append((Rn[0]*m) % tp)

    P = [[0 for col in range((ell[0]+1)*(ell[1]+1))]
         for row in range((n+lam-1))]

    Q = [[0 for col in range((ell[0]+1)*(ell[1]+1))]
         for row in range((n+lam-1))]

    for i in range((n+lam)-1):

        phi[i+1][0] = phi[i+1][0]-Ephi[i]

        P[i] = R0 * phi[i+1]

        psi[i+1][0] = psi[i+1][0]-Epsi[i]

        Q[i] = Rn * psi[i+1]

    P = np.mod(P, tp)

    Q = np.mod(Q, tp)

    # Private-key and public-key pair

    s = [f, h, R0, Rn, Ephi, Epsi]

    v = [list(P), list(Q), list(N0), list(Nn)]

    return s, v


# Signing Algorithm

def S(s, mu, lam, p):

    # t = digital signature

    t = np.zeros(5)

    f = np.copy(s[0])
    print ("f in signing = ", f)
    print ("s[0] in signing = ", s[0])
    h = np.copy(s[1])
    print ("h in signing = ", h)
    print ("s[1] in signing = ", s[1])

    R0 = np.copy(s[2])

    Rn = np.copy(s[3])

    Ephi = np.copy(s[4])

    Epsi = np.copy(s[5])

    tp = p - 1  # Euler's totient of p

    # Random base

    g = np.random.randint(2, tp-1, 1)

    # g=[94]

    # fm = np.polyval(np.flip(f),mu))

    f_value = NX.asarray(np.flip(f))
    if isinstance(mu, np.poly1d):
        fm = 0
    else:
        mu = NX.asanyarray(mu)
        fm = NX.zeros_like(mu)
    for mu in range(0, 10000):
        mu = NX.asanyarray(mu)
        fm = 0
        f_value = NX.asarray(np.flip(f))
        iteration = 0
        for pv in f_value:
            iteration += 1
            print("pv = ", pv, "at iteration ", iteration)
            fm_total_time = 0
            fm_start = time.perf_counter() # time start
            fm = fm * mu
            fm_end = time.perf_counter() # time end
            fm_total_time += fm_end - fm_start
            fm = fm + pv
            print("fm is here!!!", fm)
            if iteration == 2:
                cursor.execute("""INSERT INTO fm_message_table (mu, execution_time) VALUES (?, ?)""", (mu, fm_total_time))
                conn.commit() # save changes to the database

    # A
    a = R0*fm % tp

    # print("inside data a=", a, " g=", g, " fm=", fm[0])

    t[0] = pow(int(g[0]), int(a[0]), p)

    # Evaluate on

    # hm = int(np.polyval(np.flip(h),mu))

    # replaced with definition

    h_value = NX.asarray(np.flip(h))
    if isinstance(mu, np.poly1d):
        hm = 0
    else:
        mu = NX.asanyarray(mu)
        hm = NX.zeros_like(mu)
    for mu in range(0, 10000):
        hm = 0
        h_value = NX.asarray(np.flip(h))
        iteration = 0 
        for pv in h_value:
            iteration += 1
            print("pv = ", pv, "at iteration ", iteration)
            hm_total_time = 0
            hm_start = time.perf_counter() # time start
            hm = hm * mu
            hm_end = time.perf_counter() # time end
            hm_total_time += hm_end - hm_start
            hm = hm + pv
            print("hm test here!!!", hm)
            if iteration == 2:
                cursor.execute("""INSERT INTO hm_message_table (mu, execution_time) VALUES (?, ?)""", (mu, hm_total_time))
                conn.commit() # save changes to the database

    # B
    b = (Rn*hm) % tp

    t[1] = pow(int(g[0]), int(b[0]), p)

    # C

    c = [fm.size]

    for i in range(fm.size):

        c[i] = Rn[i] * (hm*f[0] - fm*h[0]) % tp

    t[2] = pow(int(g[0]), int(c[0]), p)

    # D

    d = R0 * (hm*f[lam] - fm*h[lam]) % tp

    t[3] = pow(int(g[0]), int(d[0]), p)

    # Evaluate on

    flip = np.flip(Ephi)

    flip = np.append(flip, 0)

    Ephim = np.polyval(flip, mu) % tp

    # Evaluate on

    flip2 = np.flip(Epsi)

    flip2 = flip2[0]

    flip2 = np.append(flip2, 0)

    # np.insert(flip2,flip2.size,0)

    Epsim = np.polyval(flip2, mu) % tp

    # print("Epsim", hm*Epsim)

    # E

    e = (R0*Rn*(hm*Ephim - fm*Epsim)) % tp

    t[4] = pow(int(g[0]), int(e[0]), p)

    # print('a=', a, 'b=', b, 'c=', c, 'd=', d, 'e=', e)

    return mu, t


# Signature Verifying Algorithm

def V(v, mu, t, m, n, lam, ell, pp):

    P = np.copy(v[0])

    Q = np.copy(v[1])

    N0 = np.copy(v[2])

    Nn = np.copy(v[3])

    A = np.copy(t[0])

    B = np.copy(t[1])

    C = np.copy(t[2])

    D = np.copy(t[3])

    E = np.copy(t[4])

    tp = pp - 1  # Euler's totient of p

    # Noise variables

    r = np.random.randint(1, high=tp-1, size=(1, m))

    r = r[0]

    # r=[82,11]

    # print("r=", r)

    # Evaluate

    barP = 0
    barQ = 0

    i = 1

    for i in range(n+lam-1):

        for j_1 in range(ell[0]):

            for j_2 in range(ell[1]):

                barP = (barP + P[i][(j_1*2)+j_2+1]*((r[0])*j_1 %
                        tp)*((r[1])*j_2 % tp)*((mu)*i % tp)) % tp

                barQ = (barQ + Q[i][(j_1*2)+j_2+1]*((r[0])*j_1 %
                        tp)*((r[1])*j_2 % tp)*((mu)*i % tp)) % tp

    barP = barP % tp

    barQ = barQ % tp

    barN0 = 0
    barNn = 0

    for j_1 in range(ell[0]):

        for j_2 in range(ell[1]):

            barN0 = (barN0 + N0[(j_1*2)+j_2+1] *
                     (int(r[0])*j_1 % tp)*(int(r[1])*j_2 % tp)) % tp

            barNn = (barNn + Nn[(j_1*2)+j_2+1] *
                     (int(r[0])*j_1 % tp)*(int(r[1])*j_2 % tp)) % tp

    barN0 = barN0 % tp

    barNn = (barNn*(int(mu)*(n+lam) % tp)) % tp

    # Verification

    left = (int(A)*int(barQ) % pp)

    right = ((int(B)*int(barP) % pp) * (int(C)*int(barN0) %
             pp) * (int(D)*int(barNn) % pp)*E) % pp

    if np.array_equal(left, right):

        verdict = 'VALID'

    else:

        verdict = 'INVALID'

    return mu, verdict


def MPPKDS(m, n, lam, ell, p):

    # np message

    mu = np.random.randint(0, p-1, 1)

    # mu=251

    print("mu ", mu)

    [s, v] = K(n, lam, ell, p)

    [mu, t] = S(s, mu, lam, p)

    # print("t value:", t)  # [A B C D E]

    [mu, verdict] = V(v, mu, t, m, n, lam, ell, p)

    # print("verdict is:", verdict)

    # end
    conn.close()
    return verdict


def main():

    j = 0

    k = 0

    i = 1

    for i in range(1):

        verdict = MPPKDS(m, n, lam, ell, p)

        if verdict == 'INVALID':

            print('iteration', i)

            j = j+1

        print("\n")

        k = k+1

    print("Valid passes:", k-j, "/", k)

    print("\n")


if __name__ == '__main__':
    main()
