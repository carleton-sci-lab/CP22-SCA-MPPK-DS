import random

# helper functions
# clean zero list
def zeroList(x, y):
    # outer list
    l1 = []
    for i in range(x):
        # inner list
        l2 = []
        for j in range(y):
            l2.append(0)
        l1.append(l2)
    return l1


# random list generator
def randList(x, y, tp):
    l = []
    for i in range(x):
        l1 = []
        for j in range(y):
            t = 0
            for m in range(4):
                t = (t * 100000) + random.randint(0, (2**16) - 1)
            l1.append(t % tp)
        l.append(l1)
    return l


# polynomial mutiplier
def polyMulti(list1, list2, list3, tp):
    for j in range(len(list2)):
        for m in range(len(list1)):
            trans = list3.pop(j + m)
            for i in range(len(list2[j])):
                trans[i] = (trans[i] + (list2[j][i] * list1[m])) % tp
            list3.insert(j + m, trans)


b = (2**64) + 1
print(b)
# Finite Field Index
p = (2**64) + 1
# Euler's totient of p
tp = p - 1
# number of noise varibles
m = 3
# degree of base polynomials
n = 2
# degree of multipler polynomials
lam = 3
# upper limit
ell = [1, 1]


# key generation algorithm
def K(n, m, lam, ell, p):
    # m = number of noise variables
    # n = degree of base polynomial
    # lam = degree of univariate polynomials
    # ell = upper limits, in base polynomial
    # p = finite field index
    tp = p - 1  # Euler's totient of p

    # randomly generate coefficients of f()
    c = randList(n + 1, (ell[0] + 1) * (ell[1] + 1), tp)
    f = randList(1, lam, tp).pop()
    h = randList(1, lam, tp).pop()

    print("f=", f, "\nh=", h, " \nc=")
    for item in c:
        print(item)
    ## Product polynomials and init \phi and \psi to zeros

    phi = zeroList((n + lam), ((ell[0] + 1) * (ell[1] + 1)))
    psi = zeroList((n + lam), ((ell[0] + 1) * (ell[1] + 1)))
    polyMulti(f, c, phi, tp)
    polyMulti(h, c, psi, tp)
    # randomly generate coefficients
    Ephi = randList(1, n + lam + 1, tp)[0]
    Epsi = randList(1, n + lam + 1, tp)[0]

    R0 = 0
    Rn = 0
    for j in range(4):
        R0 = R0 * 100000 + random.randint(0, (2**16) - 1)
    R0 = (R0 * 2) % tp
    for j in range(4):
        Rn = Rn * 100000 + random.randint(0, (2**16) - 1)
    Rn = (Rn * 2) % tp
    print("Ephi=", Ephi, "\nEpsi=", Epsi, "\nR0=", R0, "\nRn=", Rn)
    ## Noise functions and
    N0 = []
    Nn = []
    for k in c[0]:
        N0.append((R0 * k) % tp)
    for m in c[n]:
        Nn.append((Rn * m) % tp)
    P = []
    Q = []
    for i in range(1, len(phi) - 1):
        P.append(phi[i])
        Q.append(psi[i])
    for i in range(0, len(P)):
        for j in range(0, len(P[i])):
            if j == 0:
                P[i][j] = ((P[i][j]) - Ephi[i]) % tp
                Q[i][j] = (Q[i][j] - Epsi[i]) % tp
            P[i][j] = (P[i][j] * R0) % tp
            Q[i][j] = (Q[i][j] * Rn) % tp
    # Private-key and public-key pair
    print("P= ", P, "\nQ= ", Q, "\nN0= ", N0, "\nNn= ", Nn)
    s = [f, h, R0, Rn, Ephi, Epsi]
    v = [P, Q, N0, Nn]
    return s, v


##Signing Algorithm
def S(s, mu, lam, p):
    # t = digital signature
    t = []
    # mu = message
    # s = private key
    f = s[0]
    h = s[1]
    R0 = s[2]
    Rn = s[3]
    Ephi = s[4]
    Epsi = s[5]
    # m = number of noise variables
    # n = degree of base polynomial
    # lam = degree of univariate polynomials
    # ell = upper limits, in base polynomial
    # p = finite field index
    tp = p - 1  # Euler's totient of p
    # Random base
    g = 0
    for i in range(4):
        g = g * 100000 + random.randint(0, (2**16) - 1)
    g = (g * 2) % tp
    # g=[94]
    # Evaluate on
    f = f[::-1]  # flip
    fm = 0
    # A
    for i in range(len(f)):
        fm = fm + (f[i] * (mu**i))
    fm = fm % tp
    a = (R0 * fm) % tp
    print("g=", g)
    t.append(pow(g, a, p))
    # Evaluate on
    h = h[::-1]  # flip
    hm = 0
    for i in range(len(h)):
        hm = hm + int(h[i] * (mu**i))
    hm = hm % tp
    # B
    b = (Rn * hm) % tp
    t.append(pow(g, b, p))

    # C
    c = 0
    c = (Rn * (hm * f[0] - fm * h[0])) % tp
    t.append(pow(g, c, p))
    # D
    d = R0 * (hm * f[lam - 1] - fm * h[lam - 1]) % tp
    t.append(pow(g, d, p))
    # Evaluate on
    flip = Ephi[::-1]  # flip
    flip.append(0)
    Ephim = 0
    for i in range(len(flip)):
        Ephim = Ephim + int(flip[i] * (mu**i))
    Ephim = Ephim % tp
    # Evaluate on
    flip2 = Epsi[::-1]
    flip2.append(0)
    # np.insert(flip2,flip2.size,0)
    Epsim = 0
    for i in range(len(flip2)):
        Epsim = Epsim + int(flip2[i] * (mu**i))
    Epsim = Epsim % tp
    # E
    e = (R0 * Rn * (hm * Ephim - fm * Epsim)) % tp

    t.append(pow(int(g), int(e), p))
    print("a=", a, "\nb=", b, "\nc=", c, "\nd=", d, "\ne=", e)
    return mu, t


# Signature Verifying Algorithm
def V(v, mu, t, m, n, lam, ell, pp):
    # v = public key
    P = v[0]
    Q = v[1]
    N0 = v[2]
    Nn = v[3]
    # mu = message
    # t = digital signature
    A = t[0]
    B = t[1]
    C = t[2]
    D = t[3]
    E = t[4]
    # m = number of noise variables
    # n = degree of base polynomial
    # lam = degree of univariate polynomials
    # ell = upper limits, in base polynomial
    # p = finite field index
    tp = pp - 1  # Euler's totient of p
    # Noise variables
    # r=np.random.randint(1, high=tp-1,size=(1,m))
    r = []
    for i in range(m):
        r2 = 0
        for j in range(4):
            r2 = r2 * 100000 + random.randint(0, (2**16) - 1)
        r.append(r2)
    # r=r[0]
    # r=[82,11]
    # Evaluate , , and
    barP = 0
    barQ = 0
    i = 1
    for i in range(n + lam - 2):
        for j_1 in range(ell[0]):
            for j_2 in range(ell[1]):
                barP = (
                    barP
                    + P[i][(j_1 * 2) + j_2]
                    * ((r[0]) * j_1 % tp)
                    * ((r[1]) * j_2 % tp)
                    * ((mu) * i % tp)
                ) % tp
                barQ = (
                    barQ
                    + Q[i][(j_1 * 2) + j_2]
                    * ((r[0]) * j_1 % tp)
                    * ((r[1]) * j_2 % tp)
                    * ((mu) * i % tp)
                ) % tp

    barP = barP % tp
    barQ = barQ % tp
    barN0 = 0
    barNn = 0
    for j_1 in range(ell[0]):
        for j_2 in range(ell[1]):
            barN0 = (
                barN0
                + N0[(j_1 * 2) + j_2 + 1]
                * (int(r[0]) * j_1 % tp)
                * (int(r[1]) * j_2 % tp)
            ) % tp
            barNn = (
                barNn
                + Nn[(j_1 * 2) + j_2 + 1]
                * (int(r[0]) * j_1 % tp)
                * (int(r[1]) * j_2 % tp)
            ) % tp

    barN0 = barN0 % tp
    barNn = (barNn * (int(mu) * (n + lam) % tp)) % tp
    # Verification
    left = int(A) * int(barQ) % pp
    right = (
        (int(B) * int(barP) % pp)
        * (int(C) * int(barN0) % pp)
        * (int(D) * int(barNn) % pp)
        * E
    ) % pp

    if right == left:
        verdict = "VALID"
    else:
        verdict = "INVALID"

    return mu, verdict


def MPPKDS(m, n, lam, ell, p):
    # np message
    # mu = np.random.randint(0,p-1,1)
    total = 0
    for j in range(4):
        total = 100000 * total + random.randint(0, (2**16) - 1)
    mu = total
    print("mu ", mu)
    # disp(mu)
    [s, v] = K(n, m, lam, ell, p)
    # disp(s)
    # disp(v)
    [mu, t] = S(s, mu, lam, p)
    print("t value:", t)  # [A B C D E]

    # A = t(1); B = t(2); C = t(3); D = t(4); E = t(5);
    # if (A*B*C*D*E)==0 | ((A-1)*(B-1)*(C-1)*(D-1)*(E-1))==0
    # fprintf('A, B, C, D and E must not be equal to 0 or 1');
    # return;
    # else
    [mu, verdict] = V(v, mu, t, m, n, lam, ell, p)
    print("\nVerdict is:", verdict)
    # end

    return verdict


def main():
    j = 0
    k = 0
    i = 1
    avgTime = 0
    for i in range(1):
        verdict = MPPKDS(m, n, lam, ell, p)
        if verdict == "INVALID":
            print("iteration", i)
            j = j + 1
        print("\n")
        k = k + 1
    print("Valid passes:", k - j, "/", k)
    print("\n")


if __name__ == "__main__":
    main()
