import numpy as np

#to normalise all N equations
NFactor = 1.0/2.0

#isotropic term, domain 1
def Ny1_A(x, beta):
    norm = NFactor * (beta - 1) / (3 * beta)
    return norm * (5 - (x**2 * (9 - 4 * x)))

#isotropic term, domain 2
def Ny1_B(x, beta):
    norm = NFactor * (beta - 1) / (3 * beta)
    x1 = x * (1 - beta) / (1 + beta)

    A = 8 * beta**3 + 24 * beta
    B = 36 * beta**2 + 36 * beta
    C = beta**3 + 3 * beta**2 + 3 * beta + 1
    return norm * (x**3 * A - x**2 * B) / C

#anisotropic term, domain 1
def Ny2_A(x, beta):
    norm = NFactor * (beta - 1) / (3 * beta)
    return norm * (6 / beta) * (x**3 * ((1 / 3) - beta) + x**2 * (beta - 0.5) + 1 / 6)

#anisotropic term, domain 2
def Ny2_B(x, beta):
    norm = NFactor * (beta - 1) / (3 * beta)
    A = -16 * beta**5 - 32 * beta**4 - 16 * beta**3
    B = 6 * beta**6 + 18 * beta**5 + 18 * beta**4 + 6 * beta**3
    C = 3 * beta**5 + 15 * beta**4 + 30 * beta**3 + 30 * beta**2 + 15 * beta + 3
    return norm * (6 / beta) * (x**3 * A + x**2 * B) / C

#isotropic term, complete
def Isotropic(x, beta):
    cutoff = (1+beta) / (1-beta)
    if x <= cutoff:
        return Ny1_B(x, beta)
    return Ny1_A(x,beta)
    #return np.append( Ny1_B(x[x <= cutoff], beta), Ny1_A(x[x > cutoff], beta))

#anisotropic term, complete
def Anisotropic(x, beta):
    cutoff = (1+beta) / (1-beta)
    if x <= cutoff:
        return Ny2_B(x, beta)
    return Ny2_A(x,beta)
    
    #return np.append( Ny2_B(x[x <= cutoff], beta), Ny2_A(x[x > cutoff], beta))

def MaxAngle(x, beta):
    if x <= (1+beta): 
        return np.pi/2
    
    a1 = beta**2 * (1 - x**2) + 2*beta*x*(x-1) - (x-1)**2
    gamma = 1/np.sqrt(1-beta**2)
    a2 = gamma*x*beta*(1-beta)
    return -np.arcsin(np.sqrt(a1) / a2)

#return lab energies based on rest frame energies
def MaxLabEnergy(y, beta):
    #doesn't depend on beta, but pass in for symmetry
    return y

def MinLabEnergy(y, beta):
    return y * (1+beta) / (1-beta)

#return rest frames energies based on lab frame energies
def MinRestEnergy(l, beta):
    return l

def MaxRestEnergy(l, beta):
    cutoff = (1+beta) / (1-beta)
    if l < cutoff:
        return l*(1/cutoff)
    else:
        return 1

vMinRestEnergy = np.vectorize(MinRestEnergy)
vMaxRestEnergy = np.vectorize(MaxRestEnergy)
vMaxAngle = np.vectorize(MaxAngle)

def p_lambda_ang_iso(l, ang, beta):

    #convert to array early so don't have to vectorize
    l = np.array(l)
    ang = np.array(ang)

    A = beta**2 - 1
    B = 2*l*(1-beta) / np.fabs(A)
    C = l**2 * (-beta**2 + 2*beta - 1 - ( np.sin(ang)**2 * beta**2 * (1-beta) / (1+beta)) )
    C = C/ np.fabs(A)
    A = np.sqrt(np.fabs(A))

    #double norm = 2 * (beta - 1) * lambda * (1 - beta) / sqrt(1-beta*beta);
    pf = 1/(A)
    ymin = vMinRestEnergy(l, beta)
    ymax = vMaxRestEnergy(l, beta)
    maxAngle = vMaxAngle(l, beta)
    
    #do some boundary checking
    tmp1 = lambda y :-y**2 + B*y + C
    tmp2 = 4*C + B**2

    #set tolerance level to cope with rounding errors
    tol = 1E-5
    
    a1 = lambda y : np.where( tmp1(y) > tol, np.sqrt(tmp1(y)), 0) 
    tmp3 = lambda y : np.where( tmp2 > tol, (B-2*y) / np.sqrt(tmp2), 0) 

    a2 = lambda y : np.where( np.fabs(tmp3(y)) < 1, np.arcsin( np.fabs(tmp3(y)) ), np.pi/2)
    asinFactor = lambda y : np.where( (B - 2*y) < 0, -1.0, 1.0)

    f1 = lambda y : (4*y + 6*B - 12) * a1(y)
    f2 = lambda y : (4*C + 3*B**2 - 6*B ) * asinFactor(y) * a2(y)
    
    F1 = f1(ymax) - f1(ymin)
    F2 = f2(ymax) - f2(ymin)

    #force zero if out of range
    F1 = np.where(np.fabs(F1) < tol, 0, F1)
    F2 = np.where(np.fabs(F2) < tol, 0, F2)

    ans = pf * np.cos(ang) * (F1 + F2)
    return (l * (1-beta)**2 / (4* np.pi * np.sqrt(1-beta**2)) ) * ans #

def p_lambda_ang_aniso_x(l, ang, beta, pol):
    if np.fabs(pol) < 1E-6:
        return np.zeros_like(l)

    #ymin = vMinRestEnergy(l, beta)
    #ymax = vMaxRestEnergy(l, beta)
    #
    #func = lambda y : y*(y-1)
    #F1 = func(ymax)
    #F2 = func(ymin)
    #
    #gamma = 1/np.sqrt(1-beta**2)
    #Q = -1*(1-beta)**2 * gamma * l / (4*np.pi)
    #ans = pol * (8*np.pi * Q / beta) * np.cos(ang) * (F1 - F2)

    return np.zeros_like(l)

def p_lambda_ang_aniso_y(l, ang, beta, pol):
    A = beta**2 - 1
    B = 2*l*(1-beta) / np.fabs(A)
    C = l**2 * (-beta**2 + 2*beta - 1 - ( np.sin(ang)**2 * beta**2 * (1-beta) / (1+beta)) )
    C = C/ np.fabs(A)
    A = np.sqrt(np.fabs(A))

    ymin = vMinRestEnergy(l, beta)
    ymax = vMaxRestEnergy(l, beta)

    #set tolerance level to cope with rounding errors
    tol = 1E-5
    
    #for sqrt() term
    tmp1 = lambda y :-y**2 + B*y + C
    gsqrt = lambda y : np.where( tmp1(y) > tol, np.sqrt(tmp1(y)), 0) 

    #for arcsin term
    tmp2 = 4*C + B**2
    tmp3 = lambda y : np.where( tmp2 > tol, (B-2*y) / np.sqrt(tmp2), 0) 
    gasin = lambda y : np.where( np.fabs(tmp3(y)) < 1, np.arcsin( np.fabs(tmp3(y)) ), np.pi/2)    

    #to get sign of asin term correct
    asinFactor = lambda y : np.where( (B - 2*y) < 0, -1.0, 1.0)

    fasin = lambda y : (1-B) * asinFactor(y) * gasin(y)
    fsqrt = lambda y : -2 * gsqrt(y)

    Fasin = fasin(ymax) - fasin(ymin)
    Fsqrt = fsqrt(ymax) - fsqrt(ymin)

    gamma = 1/np.sqrt(1-beta**2)
    Q = -1*(1-beta)**2 * gamma * l / (4*np.pi)
    
    #this includes the factor of 2 from moving from phi -> theta_L
    ans = (4 * Q * l  / ((1+beta) * gamma * A) ) * np.sin(ang) * np.cos(ang) * (Fasin + Fsqrt) #add together because

    #HACK ans here if needed
    out = -1.0 * ans * pol
    out = np.where(np.abs(pol) < 1e-6, 0.0, out)
    return out

def p_lambda_ang_aniso_z(l, ang, beta, pol):
    A = beta**2 - 1
    B = 2*l*(1-beta) / np.fabs(A)
    C = l**2 * (-beta**2 + 2*beta - 1 - ( np.sin(ang)**2 * beta**2 * (1-beta) / (1+beta)) )
    C = C/ np.fabs(A)
    A = np.sqrt(np.fabs(A))

    ymin = vMinRestEnergy(l, beta)
    ymax = vMaxRestEnergy(l, beta)

    #set tolerance level to cope with rounding errors
    tol = 1E-5
    
    #for sqrt() term
    tmp1 = lambda y :-y**2 + B*y + C
    gsqrt = lambda y : np.where( tmp1(y) > tol, np.sqrt(tmp1(y)), 0) 
    
    #for arcsin term
    tmp2 = 4*C + B**2
    tmp3 = lambda y : np.where( tmp2 > tol, (B- (2*y)) / np.sqrt(tmp2), 0) 
    gasin = lambda y : np.where( np.fabs(tmp3(y)) < 1, np.arcsin( np.fabs(tmp3(y)) ), np.pi/2)    

    #to get sign of asin term correct
    asinFactor = lambda y : np.where( (B - (2*y)) < 0, -1.0, 1.0)

    #F1
    fasin1 = lambda y : (4*C + 3*B**2 -2*B ) * asinFactor(y) * gasin(y)
    fsqrt1 = lambda y : (4*y + 6*B -4) * gsqrt(y)

    #F2 (note same as in py)
    fasin2 = lambda y : (1-B) * asinFactor(y) * gasin(y)
    fsqrt2 = lambda y : -2 * gsqrt(y)

    #get max - min
    F1 = (-1/(4*A)) * ( (fasin1(ymax) + fsqrt1(ymax)) - (fasin1(ymin) + fsqrt1(ymin)) )
    F2 = (l * (1-beta) / A) * ( (fasin2(ymax) + fsqrt2(ymax)) - (fasin2(ymin) + fsqrt2(ymin)) )

    gamma = 1/np.sqrt(1-beta**2)
    Q = -1*(1-beta)**2 * gamma * l / (4*np.pi) 

    ans = Q * (4/beta) * np.cos(ang) * (F1-F2)

    #HACK factor of -1
    out = -1.0 * ans * pol
    out = np.where(np.abs(pol) < 1e-6, 0.0, out)
    return out

def p_lambda_ang_aniso(l, ang, beta, pol):
    #check pol has three components
    pol = np.array(pol)
    if not pol.size == 3:
        print(f"in p_lambda_ang_isol, pol:{pol} should be list type with size 3")
        return

    #check pol is normalised
    norm = pol[0]**2 + pol[1]**2 + pol[2]**2
    if np.fabs(norm - 1.0) > 1E-6:
        print(f"in p_lambda_ang_isol, pol:{pol}, mag^2:{norm} should be normalised, ")
        return

    #sum all 3 anisotropic parts together
    ans = p_lambda_ang_aniso_x(l, ang, beta, pol[0]) + p_lambda_ang_aniso_y(l, ang, beta, pol[1]) + p_lambda_ang_aniso_z(l, ang, beta, pol[2])
    return ans 

def p_lambda_ang(l, ang, beta, pol):
    #check pol has three components
    pol = np.array(pol)
    if not pol.size == 3:
        print(f"in p_lambda_ang, pol:{pol} should be list type with size 3")
        return

    #check pol is normalised
    norm = pol[0]**2 + pol[1]**2 + pol[2]**2
    if np.fabs(norm - 1.0) > 1E-6:
        print(f"in p_lambda_ang, pol:{pol}, mag^2:{norm} should be normalised, ")
        return

    ans = p_lambda_ang_iso(l, ang, beta) + p_lambda_ang_aniso_x(l, ang, beta, pol[0]) + p_lambda_ang_aniso_y(l, ang, beta, pol[1]) + p_lambda_ang_aniso_z(l, ang, beta, pol[2])
    return ans 

#vectorize all functions so can pass in single elements or numpy arrays
vIsotropic = np.vectorize(Isotropic)
vAnisotropic = np.vectorize(Anisotropic)
vMaxLabEnergy = np.vectorize(MaxLabEnergy)
vMinLabEnergy = np.vectorize(MinLabEnergy)
