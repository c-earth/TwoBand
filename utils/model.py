import numpy as np

# base functions
# --------------------------------------------------------------------------------------------------------------
def sxx(Bxx, uh, nh, ue, ne):
    return uh*nh/(1+(uh*Bxx)**2) + ue*ne/(1+(ue*Bxx)**2)

def sxy(Bxy, uh, nh, ue, ne):
    return uh*nh*uh*Bxy/(1+(uh*Bxy)**2) - ue*ne*ue*Bxy/(1+(ue*Bxy)**2)
# --------------------------------------------------------------------------------------------------------------

# model1a
# --------------------------------------------------------------------------------------------------------------
def model1a(BsSs, uh, nh, ue, ne, sxx0):
    l = int(len(BsSs) // 4)
    Bxy = BsSs[1*l:2*l]
    Sxx = BsSs[2*l:3*l]
    return np.concatenate((Sxx, sxy(Bxy, uh, (Sxx[0]-ue*ne)/uh, ue, ne)))

def model1a_verify(BsSs, uh, nh, ue, ne, sxx0):
    l = int(len(BsSs) // 4)
    Bxx = BsSs[0*l:1*l]
    Bxy = BsSs[1*l:2*l]
    Sxx = BsSs[2*l:3*l]
    return np.concatenate((sxx(Bxx, uh, (Sxx[0]-ue*ne)/uh, ue, ne), sxy(Bxy, uh, (Sxx[0]-ue*ne)/uh, ue, ne)))

def model1a_post_param(Sxx, uh, nh, ue, ne, sxx0, duh, dnh, due, dne, dsxx0):
    nh_sub = (Sxx[0]-ue*ne)/uh
    dnh_sub = nh_sub * np.sqrt((duh/uh)**2+((ue*dne)**2 + (due*ne)**2)/(Sxx[0]-ue*ne)**2)
    sxx0_sub = Sxx[0]
    dsxx0_sub = 0
    return uh, nh_sub, ue, ne, sxx0_sub, duh, dnh_sub, due, dne, dsxx0_sub
# --------------------------------------------------------------------------------------------------------------

# model1b
# --------------------------------------------------------------------------------------------------------------
def model1b(BsSs, uh, nh, ue, ne, sxx0):
    l = int(len(BsSs) // 4)
    Bxy = BsSs[1*l:2*l]
    Sxx = BsSs[2*l:3*l]
    return np.concatenate((Sxx, sxy(Bxy, uh, nh, ue, (Sxx[0]-uh*nh)/ue)))

def model1b_verify(BsSs, uh, nh, ue, ne, sxx0):
    l = int(len(BsSs) // 4)
    Bxx = BsSs[0*l:1*l]
    Bxy = BsSs[1*l:2*l]
    Sxx = BsSs[2*l:3*l]
    return np.concatenate((sxx(Bxx, uh, nh, ue, (Sxx[0]-uh*nh)/ue), sxy(Bxy, uh, nh, ue, (Sxx[0]-uh*nh)/ue)))

def model1b_post_param(Sxx, uh, nh, ue, ne, sxx0, duh, dnh, due, dne, dsxx0):
    ne_sub = (Sxx[0]-uh*nh)/ue
    dne_sub = ne_sub * np.sqrt((due/ue)**2+((uh*dnh)**2 + (duh*nh)**2)/(Sxx[0]-uh*nh)**2)
    sxx0_sub = Sxx[0]
    dsxx0_sub = 0
    return uh, nh, ue, ne_sub, sxx0_sub, duh, dnh, due, dne_sub, dsxx0_sub
# --------------------------------------------------------------------------------------------------------------

# model2
# --------------------------------------------------------------------------------------------------------------
def model2(BsSs, uh, nh, ue, ne, sxx0):
    l = int(len(BsSs) // 4)
    Bxx = BsSs[0*l:1*l]
    Bxy = BsSs[1*l:2*l]
    Sxx = BsSs[2*l:3*l]
    return np.concatenate((sxx(Bxx, uh, (Sxx[0]-ue*ne)/uh, ue, ne), sxy(Bxy, uh, nh, ue, ne)))

def model2_verify(BsSs, uh, nh, ue, ne, sxx0):
    return model2(BsSs, uh, nh, ue, ne, sxx0)

def model2_post_param(Sxx, uh, nh, ue, ne, sxx0, duh, dnh, due, dne, dsxx0):
    sxx0_sub = Sxx[0]
    dsxx0_sub = 0
    return uh, nh, ue, ne, sxx0_sub, duh, dnh, due, dne, dsxx0_sub
# --------------------------------------------------------------------------------------------------------------

# model3
# --------------------------------------------------------------------------------------------------------------
def model3(BsSs, uh, nh, ue, ne, sxx0):
    l = int(len(BsSs) // 4)
    Bxx = BsSs[0*l:1*l]
    Bxy = BsSs[1*l:2*l]
    return np.concatenate((sxx(Bxx, uh, (sxx0 - ue*ne)/uh, ue, ne), sxy(Bxy, uh, nh, ue, ne)))

def model3_verify(BsSs, uh, nh, ue, ne, sxx0):
    return model3(BsSs, uh, nh, ue, ne, sxx0)

def model3_post_param(Sxx, uh, nh, ue, ne, sxx0, duh, dnh, due, dne, dsxx0):
    return uh, nh, ue, ne, sxx0, duh, dnh, due, dne, dsxx0
# --------------------------------------------------------------------------------------------------------------

# model4
# --------------------------------------------------------------------------------------------------------------
def model4(BsSs, uh, nh, ue, ne, sxx0):
    l = int(len(BsSs) // 4)
    Bxx = BsSs[0*l:1*l]
    Bxy = BsSs[1*l:2*l]
    return np.concatenate((sxx(Bxx, uh, nh, ue, ne), sxy(Bxy, uh, nh, ue, ne)))

def model4_verify(BsSs, uh, nh, ue, ne, sxx0):
    return model4(BsSs, uh, nh, ue, ne, sxx0)

def model4_post_param(Sxx, uh, nh, ue, ne, sxx0, duh, dnh, due, dne, dsxx0):
    sxx0_sub = uh*nh + ue*ne
    dsxx0_sub = np.sqrt((uh*dnh)**2 + (duh*nh)**2 + (ue*dne)**2 + (due*ne)**2)
    return uh, nh, ue, ne, sxx0_sub, duh, dnh, due, dne, dsxx0_sub
# --------------------------------------------------------------------------------------------------------------