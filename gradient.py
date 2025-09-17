import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import multiprocessing as mp
import time
import measurements as MEAS


#{{{ realign_iiac(X,Y,x,y)
def realign_iiac(X,x):
    #plt.subplot(2,1,1)
    #plt.plot(np.abs(x))
    #plt.plot(np.abs(X))
    #plt.subplot(2,1,2)
    #plt.plot(np.unwrap(np.angle(x)))
    #plt.plot(np.unwrap(np.angle(X)))
    #plt.show()
    #M = len(X)
    #X = X if np.real(np.sum(X*x.conj())) < np.real(np.sum(X*x)) else X.conj()
    # Mag Adjust
    offset = np.median(np.diff(np.unwrap(np.angle(X/x))))
    X*=np.exp(-1j * np.arange(len(X)) * offset)
    phasex = np.angle(X.conj() @ x)
    X*= np.exp(1j * phasex)
    ambig = [ offset, phasex]
    return X,ambig
#}}}
#{{{ realign_iicc(X,Y,x,y)
def realign_iicc(X,Y,x,y):
    M = len(X)
    N = len(Y)
    # Mag Adjust
    scale = np.linalg.norm(X)/np.linalg.norm(Y) * np.linalg.norm(y)/np.linalg.norm(x)
    X/=np.sqrt(scale)
    Y*=np.sqrt(scale)
    offset = np.median(np.concatenate([np.diff(np.unwrap(np.angle(Y/y))),np.diff(np.unwrap(np.angle(X/x)))]))
    X*=np.exp(-1j * np.arange(len(X)) * offset)
    Y*=np.exp(-1j * np.arange(len(Y)) * offset)
    phasex = np.angle(X.conj() @ x)
    X*= np.exp(1j * phasex)
    phasey = np.angle(Y.conj() @ y)
    Y*= np.exp(1j * phasey)
    ambig = [np.sqrt(scale), offset, phasex, phasey]
    return X,Y,ambig
#}}}
#{{{ realign(X,Y,x,y)
def realign(X,Y,x,y):
    d = 64
    M = len(X)
    N = len(Y)
    # Mag Adjust
    scale = np.linalg.norm(X)/np.linalg.norm(Y)
    X/=np.sqrt(scale)
    Y*=np.sqrt(scale)
    # Check Swap
    total_len = d * max(M,N)
    xmiss = total_len-M
    ymiss = total_len-N

    freqXf = np.abs(np.fft.fft(np.pad(X.conj() * x,[M*d,M*d],'constant')))
    freqYf = np.abs(np.fft.fft(np.pad(Y.conj() * y,[N*d,N*d],'constant')))
    freqXb = np.abs(np.fft.fft(np.pad(X * x,[M*d,M*d],'constant')))
    freqYb = np.abs(np.fft.fft(np.pad(Y * y,[N*d,N*d],'constant')))

    offsety = np.mean(np.diff(np.unwrap(np.angle(Y/y))))
    offsetx = np.mean(np.diff(np.unwrap(np.angle(X/x))))
    X*=np.exp(-1j * np.arange(len(X)) * offsetx)
    Y*=np.exp(-1j * np.arange(len(Y)) * offsety)

    phasex = np.angle(X.conj() @ x)
    X*= np.exp(1j * phasex)
    phasey = np.angle(Y.conj() @ y)
    Y*= np.exp(1j * phasey)
    ambig = [np.sqrt(scale), -phasex, -phasey, offsetx]
    return X,Y,ambig
#}}}
#{{{ cost(X,Y, c,d, measurements)
def cost(X,Y, c,d, measurements):
    measurements2 = np.zeros_like(measurements)
    for i in range(measurements.shape[0]):
        measurements2[i,:] = np.abs(np.correlate(Y*d[i,:],(X*c[i,:]),mode='full').conj())**2
    return np.linalg.norm(measurements - measurements2)**2
#}}}
#{{{ cost_iicc(X,Y, c,d, measurements)
def cost_iicc(X,Y, c,d, measurements):
    measurements2 = np.zeros_like(measurements)
    for i in range(measurements.shape[0]):
        measurements2[i] = np.sum(np.abs(np.correlate(Y*d[i,:],(X*c[i,:]),mode='full').conj())**2)
    return np.linalg.norm(measurements - measurements2)**2
#}}}
#{{{ cost_iiac(X,Y, c,d, measurements)
def cost_iiac(X, c, measurements):
    measurements2 = np.zeros_like(measurements)
    for i in range(measurements.shape[0]):
        g_i = np.convolve(X*c[i,:], X*c[i,:],mode='full')[::-1]
        measurements2[i] = np.sum(np.abs(g_i)**2)
    return np.linalg.norm(measurements - measurements2)**2
#}}}
#{{{ wgrad_mats(X,Y,c,d,measurements)
def wgrad_mats(X,Y,c,d,measurements):
    G_X = np.zeros_like(X)
    G_Y = np.zeros_like(Y)
    M = len(X)
    N = len(Y)
    k = np.arange(-M+1,N)
    l = 1 - np.min(np.vstack([np.zeros(N+M-1), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k) * N - k]),axis=0)
    u = (u-1).astype(int)
    l = (l-1).astype(int)
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    gx = np.zeros([M,N],dtype=complex)
    gy = np.zeros([N,M],dtype=complex)
    for i in range(measurements.shape[0]):
        g_i = np.correlate(Y*d[i,:],(X*c[i,:]),mode='full').conj()
        H_i = g_i * g_i.conj()
        Lambda_i = np.outer(c[i,:], d[i,:].conj())
        for k_i in range(len(k)):
            Lambda_i[rows[k_i],cols[k_i]] *= g_i[k_i].conj() * (H_i[k_i]-measurements[i,k_i])
        gx += Lambda_i.conj()
        gy += Lambda_i.T
    return gx, gy
#}}}
#{{{ wgradient(X,Y,c,d,measurements)
def wgradient(X,Y,c,d,measurements):
    G_X = np.zeros_like(X)
    G_Y = np.zeros_like(Y)
    M = len(X)
    N = len(Y)
    k = np.arange(-M+1,N)
    l = 1 - np.min(np.vstack([np.zeros(N+M-1), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k) * N - k]),axis=0)
    u = (u-1).astype(int)
    l = (l-1).astype(int)
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    gx = np.zeros([M,N],dtype=complex)
    gy = np.zeros([N,M],dtype=complex)
    for i in range(measurements.shape[0]):
        g_i = np.correlate(Y*d[i,:],(X*c[i,:]),mode='full').conj()
        H_i = g_i * g_i.conj()
        Lambda_i = np.outer(c[i,:], d[i,:].conj())
        for k_i in range(len(k)):
            Lambda_i[rows[k_i],cols[k_i]] *= g_i[k_i].conj() * (H_i[k_i]-measurements[i,k_i])
        gx += Lambda_i.conj()
        gy += Lambda_i.T
    # In form x,y on xconj, yconj
    wgrad = np.hstack([gx @ Y, gy @ X, gx.conj() @ (Y.conj()), gy.conj() @ (X.conj())])
    sci_grad = np.hstack([(wgrad[:M+N] + wgrad[M+N:]),(wgrad[:M+N] - wgrad[M+N:])*-1j])
    return np.real(sci_grad)
#}}}
#{{{ wgradient_iicc(X,Y,c,d,measurements)
def wgradient_iicc(X,Y,c,d,measurements):
    G_X = np.zeros_like(X)
    G_Y = np.zeros_like(Y)
    M = len(X)
    N = len(Y)
    k = np.arange(-M+1,N)
    l = 1 - np.min(np.vstack([np.zeros(N+M-1), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k) * N - k]),axis=0)
    u = (u-1).astype(int)
    l = (l-1).astype(int)
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    gx = np.zeros([M,N],dtype=complex)
    gy = np.zeros([N,M],dtype=complex)
    for i in range(measurements.shape[0]):
        g_i = np.correlate(Y*d[i,:],(X*c[i,:]),mode='full').conj()
        H_i = np.sum(np.abs(g_i * g_i.conj()))
        Lambda_i = np.outer(c[i,:], d[i,:].conj())
        e_i = (H_i - measurements[i])
        for k_i in range(len(k)):
            Lambda_i[rows[k_i],cols[k_i]] *= g_i[k_i].conj() 
        gx += Lambda_i.conj() * e_i
        gy += Lambda_i.T      * e_i
    # In form x,y on xconj, yconj
    wgrad = np.hstack([gx @ Y, gy @ X, gx.conj() @ (Y.conj()), gy.conj() @ (X.conj())])
    sci_grad = np.hstack([(wgrad[:M+N] + wgrad[M+N:]),(wgrad[:M+N] - wgrad[M+N:])*-1j])
    return np.real(sci_grad)
#}}}
#{{{ wgradient_iiac(X,c,measurements)
def wgradient_iiac(X,c,measurements):
    M = len(X)
    k = np.arange(-M+1,M)
    l = 1 - np.min(np.vstack([np.zeros(M+M-1), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k) * M - k]),axis=0)
    u = (u-1).astype(int)
    l = (l-1).astype(int)
    rows = [M - 1 - np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    gx = np.zeros([M,M],dtype=complex)
    gxc = np.zeros([M,M],dtype=complex)
    for i in range(measurements.shape[0]):
        g_i = np.convolve(X*c[i,:], X*c[i,:],mode='full')
        H_i = np.sum(np.abs(g_i)**2)
        Lambda_i    = np.outer(c[i,:], c[i,:])
        e_i = (H_i - measurements[i])
        for k_i in range(len(k)):
            Lambda_i[rows[k_i],cols[k_i]] *= g_i[k_i].conj()
        gx += Lambda_i * e_i
    # In form x,y on xconj, yconj
    wgrad = np.hstack([gx.conj() @ X.conj(), gx @ X])
    sci_grad = np.hstack([(wgrad[:M] + wgrad[M:]),(wgrad[:M] - wgrad[M:])*-1j])
    return np.real(sci_grad)
#}}}
#{{{ whess(X,Y,c,d,measurements)
def whess(X,Y,c,d,measurements):
    G_X = np.zeros_like(X)
    G_Y = np.zeros_like(Y)
    M = len(X)
    N = len(Y)
    k = np.arange(-M+1,N)
    l = 1 - np.min(np.vstack([np.zeros(N+M-1), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k) * N - k]),axis=0)
    u = (u-1).astype(int)
    l = (l-1).astype(int)
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    fxx  = np.zeros([M,M],dtype=complex)
    fxxp = np.zeros([M,M],dtype=complex)
    fxy  = np.zeros([N,M],dtype=complex)
    fxyp = np.zeros([N,M],dtype=complex)

    fxpx = np.zeros([M,M],dtype=complex)
    fxpxp= np.zeros([M,M],dtype=complex)
    fxpy = np.zeros([N,M],dtype=complex)
    fxpyp= np.zeros([N,M],dtype=complex)

    fyx  = np.zeros([M,N],dtype=complex)
    fyxp = np.zeros([M,N],dtype=complex)
    fyy  = np.zeros([N,N],dtype=complex)
    fyyp = np.zeros([N,N],dtype=complex)

    fypx  = np.zeros([M,N],dtype=complex)
    fypxp = np.zeros([M,N],dtype=complex)
    fypy  = np.zeros([N,N],dtype=complex)
    fypyp = np.zeros([N,N],dtype=complex)

    for i in range(measurements.shape[0]):
        g_i = np.correlate(Y*d[i,:],(X*c[i,:]),mode='full').conj()
        H_i = g_i * g_i.conj()
        Lambda_i = np.outer(c[i,:], d[i,:].conj())
        for k_i in range(len(k)):
            Lambda_k = np.zeros_like(Lambda_i)
            Lambda_k[rows[k_i], cols[k_i]] = np.diag(Lambda_i,k = k[k_i])
            LYC = Lambda_k @ Y.conj()
            LX  = Lambda_k.T @ X
            H2h = 2 * H_i[k_i] - measurements[i,k_i]
            Hh = H_i[k_i] - measurements[i,k_i]
            gi = g_i[k_i]
            gip = gi.conj()
            gi2 = gi*gi
            gip2= gip*gip

            fxx  += H2h * np.outer(LYC, LYC.conj())
            fxxp += gi2 * np.outer(LYC.conj(), LYC.conj())
            fxy  += Hh * gi * Lambda_k.T.conj() + gi2 * np.outer(LX.conj(), LYC.conj())
            fxyp += H2h * np.outer(LX, LYC.conj())

            fxpx += gip2 * np.outer(LYC, LYC)
            fxpxp+= H2h * np.outer(LYC.conj(), LYC)
            fxpy += H2h * np.outer(LX.conj(), LYC)
            fxpyp+= Hh * gip * Lambda_k.T + gip2 * np.outer(LX, LYC)

            fyx  += Hh * gip * Lambda_k + gip2 * np.outer(LYC, LX)
            fyxp += H2h * np.outer(LYC.conj(), LX)
            fyy  += H2h * np.outer(LX.conj(), LX)
            fyyp += gip2 * np.outer(LX, LX)

            fypx += H2h * np.outer(LYC, LX.conj())
            fypxp+= Hh * gi * Lambda_k.conj() + gi2 * np.outer(LYC.conj() , LX.conj())
            fypy += gi2 * np.outer(LX.conj(), LX.conj())
            fypyp+= H2h * np.outer(LX, LX.conj())

    hess = np.vstack([
        np.hstack([fxx, fyx, fxpx, fypx]),
        np.hstack([fxy, fyy, fxpy, fypy]),
        np.hstack([fxxp, fyxp, fxpxp, fypxp]),
        np.hstack([fxyp, fyyp, fxpyp, fypyp])
        ])
    mn = M + N
    res = np.hstack([hess[:, :mn] + hess[:,mn:], -1j*(hess[:, :mn] - hess[:,mn:])])
    res = np.vstack([res[:mn,:] + res[mn:,:], 1j*(res[:mn,:] - res[mn:,:])])
    return_res = np.real(res)
    return return_res
#}}}
#{{{ whess_iicc_old(X,Y,c,d,measurements)
def whess_iicc_old(X,Y,c,d,measurements):
    G_X = np.zeros_like(X)
    G_Y = np.zeros_like(Y)
    M = len(X)
    N = len(Y)
    k = np.arange(-M+1,N)
    l = 1 - np.min(np.vstack([np.zeros(N+M-1), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k) * N - k]),axis=0)
    u = (u-1).astype(int)
    l = (l-1).astype(int)
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]

    hess = np.zeros([2*(M+N), 2*(M+N)],dtype=complex)
    zx = np.zeros_like(X)
    zy = np.zeros_like(Y)

    zxx = np.zeros([M,M])
    zxy = np.zeros([M,N])
    zyy = np.zeros([N,N])

    for i in range(measurements.shape[0]):
        g_i = np.correlate(Y*d[i,:],(X*c[i,:]),mode='full').conj()
        H_i = np.sum(np.abs(g_i * g_i.conj()))
        Lambda_i = np.outer(c[i,:], d[i,:].conj())
        gLc = np.copy(Lambda_i).conj()
        gcL = np.copy(Lambda_i)
        A_21 = np.zeros_like(hess)
        A_22 = np.zeros_like(hess)
        A_23 = np.zeros_like(hess)
        for k_i in range(len(k)):
            gLc[rows[k_i],cols[k_i]] *= g_i[k_i]
            gcL[rows[k_i],cols[k_i]] *= g_i[len(k) - k_i - 1].conj()
            Lk = np.zeros_like(Lambda_i)
            Lk[rows[k_i],cols[k_i]] = Lambda_i[rows[k_i], cols[k_i]]
            a21 = np.hstack([Lk.conj() @ Y, zy, zx, Lk.conj().T @ X.conj()])
            A_21 += np.outer(a21,a21.conj())
            a22 = np.hstack([zx, Lk.T @ X, Lk @ Y.conj(), zy])
            A_22 += np.outer(a22,a22.conj())
            r1 = np.hstack([zxx, g_i[k_i] * Lk.conj(),zxx,zxy])
            r2 = np.hstack([g_i[k_i].conj() * Lk.T, zyy, zxy.T, zyy])
            r3 = np.hstack([zxx, zxy, zxx, g_i[k_i].conj() * Lk])
            r4 = np.hstack([zxy.T, zyy, g_i[k_i] * Lk.T.conj(), zyy])
            A_23 += np.vstack([r1,r2,r3,r4])
        a_i_0 = gLc @ Y
        a_i_1 = gcL.T @ X
        a_i = np.hstack([a_i_0,a_i_1, a_i_0.conj(), a_i_1.conj()])
        hess += np.outer(a_i,a_i.conj())
        hess -= (measurements[i] - H_i) * (A_21 + A_22 + A_23)
    hess = hess
    mn = M + N
    res = np.hstack([hess[:, :mn] + hess[:,mn:], -1j*(hess[:, :mn] - hess[:,mn:])])
    res = np.vstack([res[:mn,:] + res[mn:,:], 1j*(res[:mn,:] - res[mn:,:])])
    return_res = np.real(res)
    return return_res
#}}}
#{{{ whess_iicc(X,Y,c,d,measurements)
def whess_iicc(X,Y,c,d,measurements):
    G_X = np.zeros_like(X)
    G_Y = np.zeros_like(Y)
    M = len(X)
    N = len(Y)
    k = np.arange(-M+1,N)
    l = 1 - np.min(np.vstack([np.zeros(N+M-1), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k) * N - k]),axis=0)
    u = (u-1).astype(int)
    l = (l-1).astype(int)
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    fxx  = np.zeros([M,M],dtype=complex)
    fxxp = np.zeros([M,M],dtype=complex)
    fxy  = np.zeros([N,M],dtype=complex)
    fxyp = np.zeros([N,M],dtype=complex)

    fxpx = np.zeros([M,M],dtype=complex)
    fxpxp= np.zeros([M,M],dtype=complex)
    fxpy = np.zeros([N,M],dtype=complex)
    fxpyp= np.zeros([N,M],dtype=complex)

    fyx  = np.zeros([M,N],dtype=complex)
    fyxp = np.zeros([M,N],dtype=complex)
    fyy  = np.zeros([N,N],dtype=complex)
    fyyp = np.zeros([N,N],dtype=complex)

    fypx  = np.zeros([M,N],dtype=complex)
    fypxp = np.zeros([M,N],dtype=complex)
    fypy  = np.zeros([N,N],dtype=complex)
    fypyp = np.zeros([N,N],dtype=complex)

    for i in range(measurements.shape[0]):
        g_i = np.correlate(Y*d[i,:],(X*c[i,:]),mode='full').conj()
        H_i = np.sum(g_i * g_i.conj())
        Lambda_i = np.outer(c[i,:], d[i,:].conj())
        for k_i in range(len(k)):
            Lambda_k = np.zeros_like(Lambda_i)
            Lambda_k[rows[k_i], cols[k_i]] = np.diag(Lambda_i,k = k[k_i])
            LYC = Lambda_k @ Y.conj()
            LX  = Lambda_k.T @ X
            H2h = 2 * H_i - measurements[i]
            Hh = H_i - measurements[i]
            gi = g_i[k_i]
            gip = gi.conj()
            gi2 = gi*gi
            gip2= gip*gip

            fxx  += H2h * np.outer(LYC, LYC.conj())
            fxxp += gi2 * np.outer(LYC.conj(), LYC.conj())
            fxy  += Hh * gi * Lambda_k.T.conj() + gi2 * np.outer(LX.conj(), LYC.conj())
            fxyp += H2h * np.outer(LX, LYC.conj())

            fxpx += gip2 * np.outer(LYC, LYC)
            fxpxp+= H2h * np.outer(LYC.conj(), LYC)
            fxpy += H2h * np.outer(LX.conj(), LYC)
            fxpyp+= Hh * gip * Lambda_k.T + gip2 * np.outer(LX, LYC)

            fyx  += Hh * gip * Lambda_k + gip2 * np.outer(LYC, LX)
            fyxp += H2h * np.outer(LYC.conj(), LX)
            fyy  += H2h * np.outer(LX.conj(), LX)
            fyyp += gip2 * np.outer(LX, LX)

            fypx += H2h * np.outer(LYC, LX.conj())
            fypxp+= Hh * gi * Lambda_k.conj() + gi2 * np.outer(LYC.conj() , LX.conj())
            fypy += gi2 * np.outer(LX.conj(), LX.conj())
            fypyp+= H2h * np.outer(LX, LX.conj())

    hess = np.vstack([
        np.hstack([fxx, fyx, fxpx, fypx]),
        np.hstack([fxy, fyy, fxpy, fypy]),
        np.hstack([fxxp, fyxp, fxpxp, fypxp]),
        np.hstack([fxyp, fyyp, fxpyp, fypyp])
        ])
    mn = M + N
    res = np.hstack([hess[:, :mn] + hess[:,mn:], -1j*(hess[:, :mn] - hess[:,mn:])])
    res = np.vstack([res[:mn,:] + res[mn:,:], 1j*(res[:mn,:] - res[mn:,:])])
    return_res = np.real(res)
    return return_res
#}}}
#{{{ whess_iicc_inv(X,Y,c,d,measurements)
def whess_iicc_inv(X,Y,c,d,measurements):
    G_X = np.zeros_like(X)
    G_Y = np.zeros_like(Y)
    M = len(X)
    N = len(Y)
    k = np.arange(-M+1,N)
    l = 1 - np.min(np.vstack([np.zeros(N+M-1), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k) * N - k]),axis=0)
    u = (u-1).astype(int)
    l = (l-1).astype(int)
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    fxx  = np.zeros([M,M],dtype=complex)
    fxxp = np.zeros([M,M],dtype=complex)
    fxy  = np.zeros([N,M],dtype=complex)
    fxyp = np.zeros([N,M],dtype=complex)

    fxpx = np.zeros([M,M],dtype=complex)
    fxpxp= np.zeros([M,M],dtype=complex)
    fxpy = np.zeros([N,M],dtype=complex)
    fxpyp= np.zeros([N,M],dtype=complex)

    fyx  = np.zeros([M,N],dtype=complex)
    fyxp = np.zeros([M,N],dtype=complex)
    fyy  = np.zeros([N,N],dtype=complex)
    fyyp = np.zeros([N,N],dtype=complex)

    fypx  = np.zeros([M,N],dtype=complex)
    fypxp = np.zeros([M,N],dtype=complex)
    fypy  = np.zeros([N,N],dtype=complex)
    fypyp = np.zeros([N,N],dtype=complex)

    for i in range(measurements.shape[0]):
        g_i = np.correlate(Y*d[i,:],(X*c[i,:]),mode='full').conj()
        H_i = np.sum(g_i * g_i.conj())
        Lambda_i = np.outer(c[i,:], d[i,:].conj())
        for k_i in range(len(k)):
            Lambda_k = np.zeros_like(Lambda_i)
            Lambda_k[rows[k_i], cols[k_i]] = np.diag(Lambda_i,k = k[k_i])
            LYC = Lambda_k @ Y.conj()
            LX  = Lambda_k.T @ X
            H2h = 2 * H_i - measurements[i]
            Hh = H_i - measurements[i]
            gi = g_i[k_i]
            gip = gi.conj()
            gi2 = gi*gi
            gip2= gip*gip

            fxx  += H2h * np.outer(LYC, LYC.conj())
            fxxp += gi2 * np.outer(LYC.conj(), LYC.conj())
            fxy  += Hh * gi * Lambda_k.T.conj() + gi2 * np.outer(LX.conj(), LYC.conj())
            fxyp += H2h * np.outer(LX, LYC.conj())

            fxpx += gip2 * np.outer(LYC, LYC)
            fxpxp+= H2h * np.outer(LYC.conj(), LYC)
            fxpy += H2h * np.outer(LX.conj(), LYC)
            fxpyp+= Hh * gip * Lambda_k.T + gip2 * np.outer(LX, LYC)

            fyx  += Hh * gip * Lambda_k + gip2 * np.outer(LYC, LX)
            fyxp += H2h * np.outer(LYC.conj(), LX)
            fyy  += H2h * np.outer(LX.conj(), LX)
            fyyp += gip2 * np.outer(LX, LX)

            fypx += H2h * np.outer(LYC, LX.conj())
            fypxp+= Hh * gi * Lambda_k.conj() + gi2 * np.outer(LYC.conj() , LX.conj())
            fypy += gi2 * np.outer(LX.conj(), LX.conj())
            fypyp+= H2h * np.outer(LX, LX.conj())

    hess = np.vstack([
        np.hstack([fxx, fyx, fxpx, fypx]),
        np.hstack([fxy, fyy, fxpy, fypy]),
        np.hstack([fxxp, fyxp, fxpxp, fypxp]),
        np.hstack([fxyp, fyyp, fxpyp, fypyp])
        ])
    mn = M + N
    hess = np.linalg.inv(hess)
    res = np.hstack([hess[:, :mn] + hess[:,mn:], -1j*(hess[:, :mn] - hess[:,mn:])])
    res = np.vstack([res[:mn,:] + res[mn:,:], 1j*(res[:mn,:] - res[mn:,:])])
    return_res = np.real(res)
    return return_res
#}}}
#{{{ scipy_cost(H,[c,d,measurements])
def scipy_cost(H,*args):
    c = args[0][0]
    d = args[0][1]
    mn = c.shape[1] + d.shape[1]
    z = H[:mn]
    zconj = H[mn:]
    Z = z + 1j * zconj
    measurements = args[0][2]
    x = Z[:c.shape[1]] 
    y = Z[c.shape[1]:] 
    C = cost(x,y,c,d,measurements)
    return C
#}}}
#{{{ scipy_cost_iicc(H,[c,d,measurements])
def scipy_cost_iicc(H,*args):
    c = args[0][0]
    d = args[0][1]
    mn = c.shape[1] + d.shape[1]
    z = H[:mn]
    zconj = H[mn:]
    Z = z + 1j * zconj
    measurements = args[0][2]
    x = Z[:c.shape[1]] 
    y = Z[c.shape[1]:] 
    C = cost_iicc(x,y,c,d,measurements)
    return C
#}}}
#{{{ scipy_cost_iiac(H,[c,measurements])
def scipy_cost_iiac(H,*args):
    c = args[0][0]
    m = c.shape[1]
    x = H[:m]
    xconj = H[m:]
    X = x + 1j * xconj
    measurements = args[0][1]
    C = cost_iiac(X,c,measurements)
    return C
#}}}
#{{{ scipy_grad(H,[c,d,measurements])
def scipy_grad(H,*args):
    c = args[0][0]
    d = args[0][1]
    mn = c.shape[1] + d.shape[1]
    z = H[:mn]
    zconj = H[mn:]
    Z = z + 1j * zconj
    measurements = args[0][2]
    x = Z[:c.shape[1]] 
    y = Z[c.shape[1]:] 
    G = wgradient(x,y,c,d,measurements).flatten()
    return G
#}}}
#{{{ scipy_grad_iicc(H,[c,d,measurements])
def scipy_grad_iicc(H,*args):
    c = args[0][0]
    d = args[0][1]
    mn = c.shape[1] + d.shape[1]
    z = H[:mn]
    zconj = H[mn:]
    Z = z + 1j * zconj
    measurements = args[0][2]
    x = Z[:c.shape[1]] 
    y = Z[c.shape[1]:] 
    G = wgradient_iicc(x,y,c,d,measurements).flatten()
    return G
#}}}
#{{{ scipy_grad_iiac(H,[c,d,measurements])
def scipy_grad_iiac(H,*args):
    c = args[0][0]
    m = c.shape[1]
    x = H[:m]
    xconj = H[m:]
    X = x + 1j * xconj
    measurements = args[0][1]
    G = wgradient_iiac(X,c,measurements).flatten()
    return G
#}}}
#{{{ scipy_whess(H,[c,d,measurements])
def scipy_hess(H,*args):
    c = args[0][0]
    d = args[0][1]
    mn = c.shape[1] + d.shape[1]
    z = H[:mn]
    zconj = H[mn:]
    Z = z + 1j * zconj
    measurements = args[0][2]
    x = Z[:c.shape[1]] 
    y = Z[c.shape[1]:] 
    G = whess(x,y,c,d,measurements)
    return G
#}}}
#{{{ scipy_whess_iicc(H,[c,d,measurements])
def scipy_whess_iicc(H,*args):
    c = args[0][0]
    d = args[0][1]
    mn = c.shape[1] + d.shape[1]
    z = H[:mn]
    zconj = H[mn:]
    Z = z + 1j * zconj
    measurements = args[0][2]
    x = Z[:c.shape[1]] 
    y = Z[c.shape[1]:] 
    G = whess_iicc(x,y,c,d,measurements)
    return G
#}}}
#{{{ test_grad()
def test_grad():
    M = 12
    N = 8
    K = 20
    x = np.random.randn(M) + 1j * np.random.randn(M)
    y = np.random.randn(N) + 1j * np.random.randn(N)
    x/=np.linalg.norm(x)
    y/=np.linalg.norm(y)

    c = np.random.randn(K,M) + 1j * np.random.randn(K,M)
    d = np.random.randn(K,N) + 1j * np.random.randn(K,N)
    measurements = np.zeros([K, N + M -1],dtype=float)
    for i in range(K):
        measurements[i,:] = np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2

    Noise = np.random.uniform(0,1,measurements.shape) * 0

    measurements += Noise

    Xnoise = (np.random.randn(M) + np.random.randn(M) * 1j)  * 1
    Ynoise = (np.random.randn(N) + np.random.randn(N) * 1j)  * 1
    X = x + Xnoise
    Y = y + Ynoise
    #X = Xnoise
    #Y = Ynoise

    X0 = np.copy(X)
    Y0 = np.copy(Y)

    H = np.hstack([np.real(X), np.real(Y), np.imag(X), np.imag(Y)]).flatten()

    result = minimize(scipy_cost, H, [c,d,measurements], jac=scipy_grad,\
            hess = scipy_hess,\
            method='Newton-CG', \
            #method='L-BFGS-B', \
            options = dict(disp = True))
    Z = result.x[:M+N] + 1j*result.x[M+N:]
    X = Z[:M]
    Y = Z[M:]

    #X/=np.linalg.norm(X) / np.linalg.norm(x)
    #Y/=np.linalg.norm(Y) / np.linalg.norm(y)
    X,Y,_ = realign(X,Y,x,y)
    print(np.linalg.norm(X - x), np.linalg.norm(Y - y))

    plt.subplot(2,2,1)
    plt.plot(np.abs(x))
    plt.plot(np.abs(X))
    #plt.plot(np.abs(X0))
    plt.subplot(2,2,2)
    plt.plot(np.abs(y))
    plt.plot(np.abs(Y))
    #plt.plot(np.abs(Y0))
    plt.subplot(2,2,3)
    plt.plot(np.unwrap(np.angle(x)))
    plt.plot(np.unwrap(np.angle(X)))
    #plt.plot(np.unwrap(np.angle(X0)))
    plt.subplot(2,2,4)
    plt.plot(np.unwrap(np.angle(y)))
    plt.plot(np.unwrap(np.angle(Y)))
    #plt.plot(np.unwrap(np.angle(Y0)))
    plt.show()

    measurements2 = np.zeros([K, N + M -1],dtype=float)
    for i in range(K):
        measurements2[i,:] = np.abs(np.correlate(Y*d[i,:],(X*c[i,:]),mode='full').conj())**2

    plt.imshow((measurements - measurements2)/measurements)
    plt.show()
#}}}
#{{{ test_hess    
def test_hess():
    M = 36
    K = 18

    x,y,c,d,measurements = MEAS.generate_blind_FROG(M,M,K,.2)

    Xnoise = (np.random.randn(M) + np.random.randn(M) * 1j)  * 1
    Ynoise = (np.random.randn(M) + np.random.randn(M) * 1j)  * 1
    X = x + Xnoise
    Y = y + Ynoise
    H = np.hstack([np.real(X), np.real(Y), np.imag(X), np.imag(Y)]).flatten()
    result = minimize(scipy_cost, H, [c,d,measurements], jac= scipy_grad,\
                        hess = scipy_hess,
                      method='L-BFGS-B',\
                      options = dict(ftol = 1e-6, maxiter = 1000,disp=True))
    Z = result.x[:M+M] + 1j * result.x[M+M:]
    X = Z[:M]
    Y = Z[M:]

    X,Y, ambig   = realign(X,Y,x,y)
    meas2 = MEAS.measurement(X,Y,c,d)

    plt.subplot(2,2,1)
    plt.plot(np.abs(x))
    plt.plot(np.abs(X))
    plt.subplot(2,2,2)
    plt.plot(np.unwrap(np.angle(x)))
    plt.plot(np.unwrap(np.angle(X)))
    plt.subplot(2,2,3)
    plt.plot(np.abs(y))
    plt.plot(np.abs(Y))
    plt.subplot(2,2,4)
    plt.plot(np.unwrap(np.angle(y)))
    plt.plot(np.unwrap(np.angle(Y)))
    plt.show()

    plt.subplot(1,2,1)
    plt.imshow(measurements)
    plt.subplot(1,2,2)
    plt.imshow(meas2)
    plt.show()
#}}}
#{{{ test_iicc_grad()
def test_iicc_grad():
    M = 20
    N = 20
    K = np.ceil(np.sqrt(M*N) * np.log2(M*N)*3).astype(int)
    print(K)
    x = np.random.randn(M) + 1j * np.random.randn(M)
    y = np.random.randn(N) + 1j * np.random.randn(N)
    x/=np.linalg.norm(x)
    y/=np.linalg.norm(y)

    c = np.random.randn(K,M) + 1j * np.random.randn(K,M)
    d = np.random.randn(K,N) + 1j * np.random.randn(K,N)
    measurements = np.zeros([K],dtype=float)
    for i in range(K):
        measurements[i] = np.sum(np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2)

    Xnoise = (np.random.randn(M) + np.random.randn(M) * 1j) 
    Ynoise = (np.random.randn(N) + np.random.randn(N) * 1j)  
    Xnoise/=np.linalg.norm(Xnoise)
    Ynoise/=np.linalg.norm(Ynoise)
    #X = x + Xnoise
    #Y = y + Ynoise
    X = np.ones_like(x)
    Y = np.ones_like(y)
    X/=np.linalg.norm(X)
    Y/=np.linalg.norm(Y)
    #X = Xnoise
    #Y = Ynoise
    X0 = np.copy(X)
    Y0 = np.copy(Y)
    H = np.hstack([np.real(X), np.real(Y), np.imag(X), np.imag(Y)]).flatten()

    result = minimize(scipy_cost_iicc, H, [c,d,measurements], jac=scipy_grad_iicc,\
            method='L-BFGS-B', \
            options = dict(disp = True))
    Z = result.x[:M+N] + 1j*result.x[M+N:]
    X = Z[:M]
    Y = Z[M:]

    #X/=np.linalg.norm(X) / np.linalg.norm(x)
    #Y/=np.linalg.norm(Y) / np.linalg.norm(y)
    X,Y,_ = realign_iicc(X,Y,x,y)
    print(np.linalg.norm(X - x), np.linalg.norm(Y - y))

    plt.subplot(2,2,1)
    plt.plot(np.abs(x))
    plt.plot(np.abs(X))
    plt.plot(np.abs(X0))
    plt.subplot(2,2,2)
    plt.plot(np.abs(y))
    plt.plot(np.abs(Y))
    plt.plot(np.abs(Y0))
    plt.subplot(2,2,3)
    plt.plot(np.unwrap(np.angle(x)))
    plt.plot(np.unwrap(np.angle(X)))
    plt.plot(np.unwrap(np.angle(X0)))
    plt.subplot(2,2,4)
    plt.plot(np.unwrap(np.angle(y)))
    plt.plot(np.unwrap(np.angle(Y)))
    plt.plot(np.unwrap(np.angle(Y0)))
    plt.show()

    measurements2 = np.zeros([K],dtype=float)
    for i in range(K):
        measurements2[i] = np.sum(np.abs(np.correlate(Y*d[i,:],(X*c[i,:]),mode='full').conj())**2)

    print(np.linalg.norm(measurements - measurements2)**2)

    plt.plot(measurements)
    plt.plot(measurements2)
    plt.show()
#}}}
#{{{ test_iicc_hess()

def test_iicc_hess():
    M = 6
    N = 7
    #K = np.ceil(np.sqrt(M*N) * np.log2(M*N)*1).astype(int)
    K = 400
    x = np.random.randn(M) + 1j * np.random.randn(M)
    y = np.random.randn(N) + 1j * np.random.randn(N)
    x/=np.linalg.norm(x)
    y/=np.linalg.norm(y)

    c = np.random.randn(K,M) + 1j * np.random.randn(K,M)
    d = np.random.randn(K,N) + 1j * np.random.randn(K,N)
    measurements = np.zeros([K],dtype=float)
    for i in range(K):
        measurements[i] = np.sum(np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2)

    Xnoise = (np.random.randn(M) + np.random.randn(M) * 1j) 
    Ynoise = (np.random.randn(N) + np.random.randn(N) * 1j)  
    Xnoise/=np.linalg.norm(Xnoise)
    Ynoise/=np.linalg.norm(Ynoise)
    X = x + Xnoise * .25
    Y = y + Ynoise * .25
    #X = Xnoise
    #Y = Ynoise
    X0 = np.copy(X)
    Y0 = np.copy(Y)
    H = np.hstack([np.real(X), np.real(Y), np.imag(X), np.imag(Y)]).flatten()

    result = minimize(scipy_cost_iicc, H, [c,d,measurements], jac=scipy_grad_iicc,\
            hess = scipy_whess_iicc,\
            method='L-BFGS-B', \
            options = dict(disp = True))
    Z = result.x[:M+N] + 1j*result.x[M+N:]
    X = Z[:M]
    Y = Z[M:]


    Hess = whess_iicc(X,Y,c,d,measurements)
    hess = np.linalg.pinv(result.hess_inv.todense())
    plt.subplot(1,2,1)
    plt.imshow(np.abs(hess))
    plt.subplot(1,2,2)
    plt.imshow(np.abs(Hess))
    plt.show()

    #X/=np.linalg.norm(X) / np.linalg.norm(x)
    #Y/=np.linalg.norm(Y) / np.linalg.norm(y)
    X,Y,_ = realign_iicc(X,Y,x,y)
    print(np.linalg.norm(X - x), np.linalg.norm(Y - y))

    plt.subplot(2,2,1)
    plt.plot(np.abs(x))
    plt.plot(np.abs(X))
    plt.subplot(2,2,2)
    plt.plot(np.abs(y))
    plt.plot(np.abs(Y))
    plt.subplot(2,2,3)
    plt.plot(np.unwrap(np.angle(x)))
    plt.plot(np.unwrap(np.angle(X)))
    plt.subplot(2,2,4)
    plt.plot(np.unwrap(np.angle(y)))
    plt.plot(np.unwrap(np.angle(Y)))
    plt.show()

    measurements2 = np.zeros([K],dtype=float)
    for i in range(K):
        measurements2[i] = np.sum(np.abs(np.correlate(Y*d[i,:],
                                 (X*c[i,:]),mode='full').conj())**2)

    print(np.linalg.norm(measurements - measurements2)**2)

    plt.plot(measurements)
    plt.plot(measurements2)
    plt.show()
#}}}
#{{{ test_iiac_grad()
def test_iiac_grad():
    M = 16
    I = 128

    #x,c,measurements = MEAS.generate_complex_gaussian_IIAC(M,I)
    x,c,measurements = MEAS.generate_complex_uniform_IIAC(M,I)
    #x = np.random.randn(M) + 1j * np.random.randn(M)
    #x/=np.linalg.norm(x)

    #c = np.random.randn(I,M) + 1j * np.random.randn(I,M)
    ##c = np.abs(np.real(np.random.randn(I,M) + 1j * np.random.randn(I,M))).astype(complex)
    ##c = np.exp(1j * np.angle(np.random.randn(I,M) + 1j * np.random.randn(I,M)))
    #plt.subplot(2,1,1)
    #plt.imshow(np.abs(c))
    #plt.title("Abs")
    #plt.subplot(2,1,2)
    #plt.imshow(np.angle(c))
    #plt.title("Phase")
    #plt.show()

    #M = 16
    #I = (2*M+1)**2
    #x = np.random.randn(M) + 1j * np.random.randn(M)
    #x/=np.linalg.norm(x)
    #ii = int(np.sqrt(I))
    #center_wavelength = 1.034
    #freq = np.linspace(.286, .292,M)
    #restepped_dscan = np.linspace(-5, 5,ii) * 1e3
    #d = 1
    #littrow = .54385
    #c = .2998 
    #Wfreq = 2 * np.pi * freq 
    #center_frequency = c * 2 * np.pi / center_wavelength
    #prefac = np.sqrt(1 - (2*np.pi * c / (center_frequency * d) - np.sin(littrow))**2) 
    #second_prefac = np.sqrt(1-( 2 * np.pi * c / (Wfreq * d) - np.sin(littrow))**2)
    #phase = 2 * restepped_dscan[:,None] / c * Wfreq[None,:] * second_prefac[None,:] * prefac
    #plt.imshow(phase)
    #plt.show()
    #c = np.exp(1j*phase).T
    #center = 32
    #chirp = .4
    #c2 = np.cos( np.pi * np.linspace(-center,center,ii)[:,None] * np.linspace(0,chirp,M)[None,:]).T
    #c2 = (np.arctan(c2 * 20.5) + np.pi/2)/np.pi
    #plt.imshow(c2)
    #plt.show()

    #c = (c[:,:,None] * c2[:,None,:]).reshape(M,-1).T
    #plt.imshow(np.angle(c))
    #plt.show()
    #plt.imshow(np.abs(c))
    #plt.show()
    
    measurements2 = np.zeros([I],dtype=float)
    for i in range(I):
        measurements2[i] = np.sum(np.abs(np.correlate(x[::-1].conj()*c[i,::-1].conj(),(x*c[i,:]),mode='full').conj())**2)

    plt.plot(measurements)
    plt.plot(measurements2)
    plt.show()
    noise = np.random.randn(I)**2 
    noise/=np.linalg.norm(noise)/np.linalg.norm(measurements)
    measurements += noise * 1e-2
    H = np.hstack([np.real(x),np.imag(x)]).flatten()

    X = np.ones_like(x)
    X/=np.linalg.norm(X)
    X0 = np.copy(X)
    H = np.hstack([np.real(X),np.imag(X)]).flatten()

    result = minimize(scipy_cost_iiac, H, [c,measurements], jac=scipy_grad_iiac,\
            method='L-BFGS-B', \
            options = dict(disp = True,maxfun = 500,ftol=1e-12,gtol=1e-12))
    print(result)
    X = result.x[:M] + 1j*result.x[M:]

    X,_ = realign_iiac(X,x)

    print(np.linalg.norm(X - x))

    plt.subplot(2,1,1)
    plt.plot(np.abs(x))
    plt.plot(np.abs(X))
    plt.subplot(2,1,2)
    plt.plot(np.unwrap(np.angle(x/X)))
    plt.show()

    ##X/=np.linalg.norm(X) / np.linalg.norm(x)
    ##Y/=np.linalg.norm(Y) / np.linalg.norm(y)
    #X,Y,_ = realign_iicc(X,Y,x,y)
    #print(np.linalg.norm(X - x), np.linalg.norm(Y - y))

#}}}


if __name__ == '__main__':
    #test_grad()
    #test_hess()
    #test_iicc_grad()
    #test_iicc_hess()
    test_iiac_grad()

