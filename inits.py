import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import tucker,CPPower, CP,non_negative_tucker
from tensorly.decomposition import tensor_train_matrix, power_iteration, parafac, CP
from tensorly.decomposition import symmetric_parafac_power_iteration as SPPI
from gradient import realign_iicc, realign_iiac
from measurements import measurement as gen_meas
from measurements import measurement_iicc as gen_meas_iicc
from measurements import measurement_iiac as gen_meas_iiac

'''
Inits are all setup to take measurements in the order of:
measurements (the intensity of modulated cross-correlation of x and y, K by M+N-1)
c (complex modulations on x, shaped K by M)
d (complex modulations on y, shaped K by N)

RAAR algorithms have two parameters which I'll call alpha and beta.
Alpha is the mixing between mag reflection and current position,
beta is the mix between this mix and the mag projection.
'''

#{{{ spectral_init
def spectral_init(measurements,c,d,max_its = 1000, threshold = 1e-1):
    M = c.shape[1]
    N = d.shape[1]
    K = c.shape[0]
    k = np.arange(-M+1,N)
    l = 1 - np.min(np.vstack([np.zeros(N+M-1), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k) * N - k]),axis=0)
    u = (u-1).astype(int)
    l = (l-1).astype(int)
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    stack = c[:,:,None] * d[:,None,:].conj()
    output = np.zeros([M,N],dtype=complex)
    delta = []
    #{{{ diags
    for i in range(len(k)):
        forward = stack[:,rows[i], cols[i]]
        #print(measurements)
        result = measurements[:,i].flatten()
        spec_piece = np.sum(forward[:,:,None] * forward[:,None,:].conj() * result[:,None,None],axis=0)
        u,s,v = np.linalg.svd(spec_piece)
        vec = u[:,0] * np.sqrt(s[0]/K) / 5
        output[rows[i],cols[i]] = vec #* np.exp(-1j * np.angle(vec[0]))
    #}}}
    old = np.copy(output)
    #{{{ mag
    for i in range(max_its):
        mag_fix = np.abs((output[:-1,1:] * output[1:,:-1]) / (output[:-1,:-1] * output[1:,1:]) )**(1/8)
        fixer = np.ones_like(output)
        fixer[:-1,:-1] *= mag_fix
        fixer[1:,1:]   *= mag_fix
        fixer[:-1, 1:] /= mag_fix
        fixer[1: ,:-1] /= mag_fix
        output *= fixer
        D = np.linalg.norm(old - output)/np.linalg.norm(old) 
        delta.append(D)
        old = np.copy(output)
        if D < threshold:
            break
    #}}}
    #{{{ phase
    for i in range(output.shape[0] + output.shape[1]):
        phase_fix = np.exp(1j * np.angle((output[:-1,1:] * output[1:,:-1]) / (output[:-1,:-1] * output[1:,1:]) ) )
        phases = np.ones_like(output)
        for j in range(len(k)):
            m = np.diag(phase_fix, k=k[j]-1)
            m = np.hstack([m, np.ones(len(rows[j]) - len(m))])
            p = np.diag(phase_fix, k=k[j]+1)
            p = np.hstack([p, np.ones(len(rows[j]) - len(p))])
            if k[j] < 0:
                phases[rows[j],cols[j]] *= p
            elif k[j] > 1:
                phases[rows[j],cols[j]] *= m
        for j in range(len(k)):
            l = len(cols[j])
            m1 = np.diag(phases,k=k[j]-1)[:l]
            m2 = np.diag(phases,k=k[j]-2)[:l]
            p1 = np.diag(phases,k=k[j]+1)[:l]
            p2 = np.diag(phases,k=k[j]+2)[:l]
            p2 = np.hstack([p2, np.ones(l - len(p2))])
            if k[j] < 0:
                phases[rows[j],cols[j]] *=p1**2/p2
            if k[j] > 1:
                phases[rows[j],cols[j]] *=m1**2/m2
        output *= np.exp(-1j * np.angle(phases))
        D = np.linalg.norm(old - output)/np.linalg.norm(old) 
        delta.append(D)
        if D < 1e-4:
            break
        old = np.copy(output)
    #}}}
    u,s,v = np.linalg.svd(output)
    X = u[:,0].conj() * np.sqrt(s[0]) #* np.sqrt(M/N)
    Y = v[0,:] * np.sqrt(s[0])
    return X,Y
#}}}
#{{{ alt_min_init
def alt_min_init(measurements,c,d, max_its=1000, threshold = 1e-1):
    root_measurements = np.sqrt(measurements)
    M = c.shape[1]
    N = d.shape[1]
    X = np.zeros(M,dtype=complex)
    Y = np.zeros(N,dtype=complex)
    k = np.arange(-M+1,N)
    l = 1 - np.min(np.vstack([np.zeros(N+M-1), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k) * N - k]),axis=0)
    u = (u-1).astype(int)
    l = (l-1).astype(int)
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    stack = c[:,:,None] * d[:,None,:].conj()
    cd = [stack[:,rows[i],cols[i]] for i in range(len(k))]
    cdi = [np.linalg.pinv(cd[i]) for i in range(len(k))]
    O = np.outer(X,Y.conj())
    previous = np.copy(O)
    error = []
    for it in range(max_its):
        meas_project = np.zeros_like(O)
        for kk in range(len(k)):
            forward = cd[kk] @ np.diag(O,k=k[kk])
            phase = np.exp(1j*np.angle(forward))
            phased_measurements = phase * root_measurements[:,kk]
            back = cdi[kk] @ phased_measurements
            #back = cd[kk].T.conj() @ phased_measurements
            meas_project[rows[kk],cols[kk]] = back
        u,s,v = np.linalg.svd(meas_project)
        rank_project = np.outer(u[:,0], v[0,:]) * s[0]
        O = rank_project
        error.append(np.linalg.norm(O-previous))
        if np.linalg.norm(O - previous) < np.linalg.norm(O) * threshold:
            break
        previous = O
    #plt.plot(error)
    #plt.show()
    u,s,v = np.linalg.svd(O)
    X = u[:,0] * np.sqrt(s[0])
    Y = v[0,:].conj() * np.sqrt(s[0])
    return X,Y
#}}}
#{{{ raar_min_init
def raar_init(measurements,c,d, max_its=1000, threshold = 1e-1, alpha=.9, beta = .5):
    root_measurements = np.sqrt(measurements)
    M = c.shape[1]
    N = d.shape[1]
    X = np.zeros(M,dtype=complex)
    Y = np.zeros(N,dtype=complex)
    k = np.arange(-M+1,N)
    l = 1 - np.min(np.vstack([np.zeros(N+M-1), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k) * N - k]),axis=0)
    u = (u-1).astype(int)
    l = (l-1).astype(int)
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    stack = c[:,:,None] * d[:,None,:].conj()
    cd = [stack[:,rows[i],cols[i]] for i in range(len(k))]
    cdi = [np.linalg.pinv(cd[i]) for i in range(len(k))]
    O = np.outer(X,Y.conj())
    previous = np.copy(O)
    error = []
    for it in range(max_its):
        p2 = np.zeros_like(O)
        for kk in range(len(k)):
            forward = cd[kk] @ np.diag(O,k=k[kk])
            phase = np.exp(1j*np.angle(forward))
            phased_measurements = phase * root_measurements[:,kk]
            back = cdi[kk] @ phased_measurements
            p2[rows[kk],cols[kk]] = back
        r1 = 2*p2 - O
        u,s,v = np.linalg.svd(r1)
        p1 = np.outer(u[:,0], v[0,:]) * s[0]
        r2 = 2 * p1 - p2
        part1 = (r2 * alpha + O * (1-alpha))*beta
        part2 = p2 * (1 - beta)
        O = part1 + part2
        error.append(np.linalg.norm(O-previous))
        if np.linalg.norm(O - previous) < np.linalg.norm(O) * threshold:
            break
        previous = O
    #plt.semilogy(error)
    #plt.show()
    u,s,v = np.linalg.svd(O)
    X = u[:,0] * np.sqrt(s[0])
    Y = v[0,:].conj() * np.sqrt(s[0])
    return X,Y
#}}}
#{{{ tensor_mag_proj
def tensor_mag_proj(t,meas,gamma_set, gammainv_set,indexing_set):
    T = np.copy(t)
    error = 0
    for i in range(len(gamma_set)):
        NN = indexing_set[0][i]
        NPK = indexing_set[1][i]
        MM = indexing_set[2][i]
        MPK = indexing_set[3][i]
        single_lag = t[NN,NPK,MM,MPK].flatten()
        error_single_lag = meas[:,i] - gamma_set[i] @ single_lag
        error += np.linalg.norm(error_single_lag)**2
        back_project = gammainv_set[i] @ error_single_lag
        T[NN,NPK,MM,MPK] = single_lag + back_project
    return T, np.sqrt(error) / np.linalg.norm(meas)
#}}}
#{{{ tensor_mag_proj_iicc
def tensor_mag_proj_iicc(t,meas,gamma,gamma_inv,indexing_set):
    T = np.copy(t)
    error = 0
    forward_vector = []
    MN1 = len(indexing_set[0])
    for i in range(MN1):
        NN = indexing_set[0][i]
        NPK = indexing_set[1][i]
        MM = indexing_set[2][i]
        MPK = indexing_set[3][i]
        single_lag = t[NN,NPK,MM,MPK].flatten()
        forward_vector.append(single_lag)
    lens = [len(i) for i in forward_vector]
    lens = np.hstack([np.array([0]), np.cumsum(lens)])
    forward_vector = np.concatenate(forward_vector)
    error_single_lag = meas - gamma @ forward_vector
    error = np.linalg.norm(error_single_lag)**2
    if gamma_inv is None:
        #back_project = gamma.conj().T @ error_single_lag / np.linalg.norm(gamma)**2
        scale = np.linalg.norm(gamma @ ( gamma.conj().T @ meas)) / np.linalg.norm(meas)
        back_project = gamma.conj().T @ error_single_lag / scale
    else:
        back_project = gamma_inv @ error_single_lag  # Norm division is new
    for i in range(MN1):
        NN = indexing_set[0][i]
        NPK = indexing_set[1][i]
        MM = indexing_set[2][i]
        MPK = indexing_set[3][i]
        sub_back = back_project[lens[i]:lens[i+1]]
        T[NN,NPK,MM,MPK] += sub_back
    return T, np.sqrt(error) / np.linalg.norm(meas)
#}}}
#{{{ tensor_rank_proj
def tensor_rank_proj(t):
    factors = tucker(t, rank=[1,1,1,1])
    factors.factors[2] = factors.factors[0].conj()
    factors.factors[3] = factors.factors[1].conj()
    T = tl.tucker_to_tensor(factors)
    return T

def tensor_rank_proj_tt(t):
    factors = tensor_train(t, rank=1)
    factors.factors[2] = factors.factors[0].conj()
    factors.factors[3] = factors.factors[1].conj()
    T = tl.tt_to_tensor(factors)
    return T

def tensor_rank_proj_power_it(t):
    val,vec,deflated = power_iteration(t)
    T = val * vec[0][:,None,None,None] * vec[1][None,:,None,None] * vec[2][None,None,:,None] * vec[3][None,None,None,:]
    return T

def tensor_rank_proj_parafac(t):
    w,factors = parafac(t,rank=1)
    factors[2] = factors[0].conj()
    factors[3] = factors[1].conj()
    T = tl.cp_to_tensor([w,factors])
    return T
#}}}
#{{{ tensor_rank_proj_iiac
def tensor_rank_proj_iiac(t):
    factors = tucker(t, rank=[1,1,1,1])
    factors.factors[1] = factors.factors[0]
    factors.factors[2] = factors.factors[0].conj()
    factors.factors[3] = factors.factors[0].conj()
    T = tl.tucker_to_tensor(factors)
    return T

def tensor_rank_proj_iiac_CP(t):
    cp = CP(rank = 1,init='svd')
    factors = cp.fit_transform(t)
    #factors.factors[1] = factors.factors[0]
    #factors.factors[2] = factors.factors[0].conj()
    #factors.factors[3] = factors.factors[0].conj()
    T = tl.cp_to_tensor(factors)
    return T

def tensor_rank_proj_iiac_parafac(t):
    w,factors = parafac(t,rank=1)
    factors[1] = factors[0]
    factors[2] = factors[0].conj()
    factors[3] = factors[0].conj()
    T = tl.cp_to_tensor([w,factors])
    return T
#}}}
#{{{ gen_indexing_set
def gen_indexing_set(k,rows):
    NN = []
    MM = []
    NPK = []
    MPK = []
    for i in range(len(k)):
        n = np.copy(rows[i])
        m = np.copy(rows[i])
        NN.append(np.repeat(n,len(m)))
        MM.append(np.tile(m,len(n)))
        NPK.append(NN[-1] + k[i])
        MPK.append(MM[-1] + k[i])
    return [NN,NPK,MM,MPK]
#}}}
#{{{ gen_indexing_set_iiac
def gen_indexing_set_iiac(k,rows,cols):
    RN,RM,FN,FM = [],[],[],[]
    for i in range(len(k)):
        n = np.copy(rows[i])
        m = np.copy(cols[i])
        RN.append(np.repeat(n,len(m)))
        RM.append(np.repeat(m,len(n)))
        FN.append(np.tile(n,len(m)))
        FM.append(np.tile(m,len(n)))
    return [RN,RM,FN,FM]
#}}}
#{{{ raar_tensor_init
def raar_tensor_init(measurements,c,d,max_its=1000, threshold = 1e-1, alpha=.9, beta = .5,min_its = 100):
    M = c.shape[1]
    N = d.shape[1]
    K = c.shape[0]
    k = np.arange(-M+1,N)
    l = -np.min(np.vstack([np.zeros_like(k), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k)*N - k]), axis=0)  - 1
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    outer_meas = [c[:,rows[i]] * d[:,cols[i]].conj() for i in range(len(k))]
    gamma_set = []
    gammainv_set = []
    for kk in range(len(k)):
        val = (outer_meas[kk][:,:,None] * outer_meas[kk][:,None,:].conj()) 
        gamma_set.append(val.reshape(K,-1))
        gammainv_set.append(np.linalg.pinv(gamma_set[-1]))
    t = np.zeros([M,N,M,N],dtype=complex)
    t = np.random.randn(*t.shape) + 1j * np.random.randn(*t.shape)
    old = np.copy(t)
    error=[]
    indexing_set = gen_indexing_set(k, rows)
    for q in range(max_its):
        p2,e = tensor_mag_proj(t,measurements,gamma_set, gammainv_set,indexing_set)
        r1 = 2 * p2 - t
        p1 = tensor_rank_proj(r1)
        r2 = 2 * p1 - p2
        part1 = (r2 * alpha + t * (1-alpha)) * beta
        part2 = p2 * ( 1 - beta)
        t = part1 + part2
        if e < threshold : 
            break
        #print(e)
        error.append(e)
        old = np.copy(t)
    factors = tucker(t, rank=[1,1,1,1])
    X = factors.factors[0].flatten()
    Y = factors.factors[3].flatten()
    m2 = gen_meas(X,Y,c,d)
    scale = np.sqrt(np.sqrt(np.mean(m2/measurements)))
    X/=scale
    Y/=scale
    return X,Y,error
#}}}
#{{{ alt_min_tensor_init
def alt_min_tensor_init(measurements,c,d,max_its=1000, threshold = 1e-1, alpha=.9, beta = .5,min_its = 100):
    M = c.shape[1]
    N = d.shape[1]
    K = c.shape[0]
    k = np.arange(-M+1,N)
    l = -np.min(np.vstack([np.zeros_like(k), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k)*N - k]), axis=0)  - 1
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    outer_meas = [c[:,rows[i]] * d[:,cols[i]].conj() for i in range(len(k))]
    gamma_set = []
    gammainv_set = []
    for kk in range(len(k)):
        val = (outer_meas[kk][:,:,None] * outer_meas[kk][:,None,:].conj())
        gamma_set.append(val.reshape(K,-1))
        gammainv_set.append(np.linalg.pinv(gamma_set[-1]))
    t = np.zeros([M,N,M,N],dtype=complex)
    old = np.copy(t)
    error=[]
    indexing_set = gen_indexing_set(k, rows)
    for q in range(max_its):
        p2,e = tensor_mag_proj(t,measurements,gamma_set, gammainv_set,indexing_set)
        t = tensor_rank_proj(p2)
        if e < threshold : 
            break
        #print(e)
        error.append(e)
        old = np.copy(t)
    #out = CP(1)
    #lam,vecs = out.fit_transform(t)
    #X = vecs[0].flatten() 
    #Y = vecs[3].flatten() 
    factors = tucker(t, rank=[1,1,1,1])
    X = factors.factors[0].flatten()
    Y = factors.factors[3].flatten()
    m2 = gen_meas(X,Y,c,d)
    scale = np.sqrt(np.sqrt(np.mean(m2/measurements)))
    X/=scale
    Y/=scale
    return X,Y,error
#}}}
#{{{ test_inits
def test_inits():
    M = 20
    N = 20
    K = 10
    x = np.random.randn(M) + 1j * np.random.randn(M)
    y = np.random.randn(N) + 1j * np.random.randn(N)
    x/=np.linalg.norm(x)
    y/=np.linalg.norm(y)
    c = np.random.randn(K,M) + 1j * np.random.randn(K,M)
    d = np.random.randn(K,N) + 1j * np.random.randn(K,N)
    measurements = np.zeros([K, N + M -1],dtype=float)
    for i in range(K):
        measurements[i,:] = np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2
    noise = np.random.uniform(0,1, measurements.shape)
    noise/=np.linalg.norm(noise)/np.linalg.norm(measurements)
    measurements_no_noise = np.copy(measurements)
    #measurements += noise * 1e-2
    print(measurements.shape)

    #x_spectral, y_spectral = spectral_init(measurements,c,d)
    #x_spectral, y_spectral, ambig_spectral = realign(x_spectral,y_spectral,x,y)
    #measurements_spectral = np.zeros_like(measurements)
    #for i in range(K):
    #    measurements_spectral[i,:] = np.abs(np.correlate(y_spectral*d[i,:],(x_spectral*c[i,:]),mode='full').conj())**2

    #x_alt_min, y_alt_min = alt_min_init(measurements,c,d)
    #x_alt_min, y_alt_min, ambig_alt_min = realign(x_alt_min,y_alt_min,x,y)
    #measurements_alt_min = np.zeros_like(measurements)
    #for i in range(K):
    #    measurements_alt_min[i,:] = np.abs(np.correlate(y_alt_min*d[i,:],(x_alt_min*c[i,:]),mode='full').conj())**2

    x_alt_min_tensor, y_alt_min_tensor,e = raar_tensor_init(measurements,c,d,max_its=10000,threshold=1e-2,\
            alpha = .9, beta = .5)
    x_alt_min_tensor, y_alt_min_tensor, ambig_alt_min_tensor = realign(x_alt_min_tensor,y_alt_min_tensor,x,y)
    measurements_alt_min_tensor = np.zeros_like(measurements)
    plt.semilogy(e)
    plt.show()
    for i in range(K):
        measurements_alt_min_tensor[i,:] = np.abs(np.correlate(y_alt_min_tensor*d[i,:],(x_alt_min_tensor*c[i,:]),mode='full').conj())**2

    print('results\n')
    print(np.linalg.norm(x_alt_min_tensor - x)/np.linalg.norm(x))
    print(np.linalg.norm(y_alt_min_tensor - y)/np.linalg.norm(y))
    print(np.linalg.norm(measurements_alt_min_tensor - measurements)/np.linalg.norm(measurements))
    plt.subplot(2,2,1)
    plt.plot(np.abs(x))
    plt.plot(np.abs(x_alt_min_tensor))
    plt.subplot(2,2,2)
    plt.plot(np.abs(y))
    plt.plot(np.abs(y_alt_min_tensor))
    plt.subplot(2,2,3)
    plt.plot(np.unwrap(np.angle(x)))
    plt.plot(np.unwrap(np.angle(x_alt_min_tensor)))
    plt.subplot(2,2,4)
    plt.plot(np.unwrap(np.angle(y)))
    plt.plot(np.unwrap(np.angle(y_alt_min_tensor)))
    plt.show()

    plt.subplot(1,2,1)
    plt.imshow(np.abs(measurements),aspect = measurements.shape[1]/measurements.shape[0])
    plt.subplot(1,2,2)
    plt.imshow(np.abs(measurements_alt_min_tensor),aspect = measurements.shape[1]/measurements.shape[0])
    plt.show()
#}}}
#{{{ alt_min_tensor_init_iicc
def alt_min_tensor_init_iicc(measurements,c,d,max_its=1000, threshold = 1e-1, alpha=.9, beta = .5,min_its = 100,pinv=False):
    M = c.shape[1]
    N = d.shape[1]
    K = c.shape[0]
    k = np.arange(-M+1,N)
    l = -np.min(np.vstack([np.zeros_like(k), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k)*N - k]), axis=0)  - 1
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    outer_meas = [c[:,rows[i]] * d[:,cols[i]].conj() for i in range(len(k))]
    gamma_set = []
    for kk in range(len(k)):
        val = (outer_meas[kk][:,:,None] * outer_meas[kk][:,None,:].conj())
        gamma_set.append(val.reshape(K,-1))
    gamma = np.hstack(gamma_set)
    if pinv:
        gamma_inv = np.linalg.pinv(gamma)
    else:
        gamma_inv = None
    #t = np.zeros([M,N,M,N],dtype=complex)
    #t = np.random.randn(M,N,M,N) + 1j * np.random.randn(M,N,M,N)
    xx = np.random.randn(M) * 1j * np.random.randn(M)
    yy = np.random.randn(N) * 1j * np.random.randn(N)
    m2 = gen_meas_iicc(xx,yy,c,d)
    scale = np.sqrt(np.sqrt(np.mean(m2/measurements)))
    xx/=scale
    yy/=scale
    t = xx[:,None,None,None] * yy.conj()[None,:,None,None] * xx.conj()[None,None,:,None] * yy[None,None,None,:]
    old = np.copy(t)
    error=[]
    indexing_set = gen_indexing_set(k, rows)
    for q in range(max_its):
        p2,e = tensor_mag_proj_iicc(t,measurements,gamma, gamma_inv ,indexing_set)
        t = tensor_rank_proj(p2)
        if e < threshold : 
            break
        error.append(e)
        #print(e)
        old = np.copy(t)
    factors = tucker(t, rank=[1,1,1,1])
    X = factors.factors[0].flatten()
    Y = factors.factors[3].flatten()
    m2 = gen_meas_iicc(X,Y,c,d)
    scale = np.sqrt(np.sqrt(np.mean(m2/measurements)))
    X/=scale
    Y/=scale
    return X,Y,error
#}}}
#{{{ raar_min_tensor_init_iicc
def raar_min_tensor_init_iicc(measurements,c,d,max_its=1000, threshold = 1e-1, alpha=.9, beta = .5,min_its = 100,pinv=False):
    M = c.shape[1]
    N = d.shape[1]
    K = c.shape[0]
    k = np.arange(-M+1,N)
    l = -np.min(np.vstack([np.zeros_like(k), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k)*N - k]), axis=0)  - 1
    rows = [np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    outer_meas = [c[:,rows[i]] * d[:,cols[i]].conj() for i in range(len(k))]
    gamma_set = []
    for kk in range(len(k)):
        val = (outer_meas[kk][:,:,None] * outer_meas[kk][:,None,:].conj())
        gamma_set.append(val.reshape(K,-1))
    gamma = np.hstack(gamma_set)
    if pinv:
        gamma_inv = np.linalg.pinv(gamma)
    else:
        gamma_inv = None
    #t = np.zeros([M,N,M,N],dtype=complex)
    xx = np.random.randn(M) * 1j * np.random.randn(M)
    yy = np.random.randn(N) * 1j * np.random.randn(N)
    m2 = gen_meas_iicc(xx,yy,c,d)
    scale = np.sqrt(np.sqrt(np.mean(m2/measurements)))
    xx/=scale
    yy/=scale
    t = xx[:,None,None,None] * yy.conj()[None,:,None,None] * xx.conj()[None,None,:,None] * yy[None,None,None,:]
    old = np.copy(t)
    error=[]
    indexing_set = gen_indexing_set(k, rows)
    for q in range(max_its):
        p2,e = tensor_mag_proj_iicc(t,measurements,gamma, gamma_inv,indexing_set)
        r1 = 2 * p2 - t
        p1 = tensor_rank_proj(r1)
        r2 = 2 * p1 - p2
        part1 = (r2 * alpha + t * (1-alpha)) * beta
        part2 = p2 * ( 1 - beta)
        t = part1 + part2
        if e < threshold : 
            break
        error.append(e)
        old = np.copy(t)
    factors = tucker(t, rank=[1,1,1,1])
    X = factors.factors[0].flatten()
    Y = factors.factors[3].flatten()
    m2 = gen_meas_iicc(X,Y,c,d)
    scale = np.sqrt(np.sqrt(np.mean(m2/measurements)))
    X/=scale
    Y/=scale
    return X,Y,error
#}}}
#{{{ alt_min_tensor_init_iiac
def alt_min_tensor_init_iiac(measurements,c,max_its=1000, threshold = 1e-1, alpha=.9, beta = .5,min_its = 100,pinv=False):
    M = c.shape[1]
    K = c.shape[0]
    k = np.arange(-M+1,M)
    l = -np.min(np.vstack([np.zeros_like(k), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k)*M - k]), axis=0)  - 1
    rows = [M - 1 - np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    outer_meas = [c[:,rows[i]] * c[:,cols[i]] for i in range(len(k))]
    gamma_set = []
    for kk in range(len(k)):
        val = (outer_meas[kk][:,:,None] * outer_meas[kk][:,None,:].conj())
        gamma_set.append(val.reshape(K,-1))
    gamma = np.hstack(gamma_set)
    if pinv:
        gamma_inv = np.linalg.pinv(gamma)
    else:
        gamma_inv = None
    xx = np.random.randn(M) * 1j * np.random.randn(M)
    m2 = gen_meas_iiac(xx,c)
    scale = np.sqrt(np.sqrt(np.mean(m2/measurements)))
    xx/=scale
    t = xx[:,None,None,None] * xx[None,:,None,None] * xx.conj()[None,None,:,None] * xx.conj()[None,None,None,:]
    old = np.copy(t)
    error=[]
    indexing_set = gen_indexing_set_iiac(k, rows,cols)
    for q in range(max_its):
        p2,e = tensor_mag_proj_iicc(t,measurements,gamma, gamma_inv ,indexing_set)
        t = tensor_rank_proj_iiac(p2)
        if e < threshold : 
            break
        #print(e)
        error.append(e)
        old = np.copy(t)
    factors = tucker(t, rank=[1,1,1,1])
    X = factors.factors[0].flatten()
    m2 = gen_meas_iiac(X,c)
    scale = np.sqrt(np.sqrt(np.mean(m2/measurements)))
    X/=scale
    return X,error
#}}}
#{{{ raar_min_tensor_init_iiac
def raar_min_tensor_init_iiac(measurements,c,max_its=1000, threshold = 1e-1, alpha=.9, beta = .5,min_its = 100,pinv=False):
    M = c.shape[1]
    K = c.shape[0]
    k = np.arange(-M+1,M)
    l = -np.min(np.vstack([np.zeros_like(k), k]),axis=0)
    u = np.min(np.vstack([np.ones_like(k) * M, np.ones_like(k)*M - k]), axis=0)  - 1
    rows = [M - 1 - np.arange(l[i],u[i]+1) for i in range(len(k))]
    cols = [np.arange(l[i],u[i]+1) + k[i] for i in range(len(k))]
    outer_meas = [c[:,rows[i]] * c[:,cols[i]] for i in range(len(k))]
    gamma_set = []
    for kk in range(len(k)):
        val = (outer_meas[kk][:,:,None] * outer_meas[kk][:,None,:].conj())
        gamma_set.append(val.reshape(K,-1))
    gamma = np.hstack(gamma_set)
    if pinv:
        gamma_inv = np.linalg.pinv(gamma)
    else:
        gamma_inv = None
    xx = np.random.randn(M) * 1j * np.random.randn(M)
    m2 = gen_meas_iiac(xx,c)
    scale = np.sqrt(np.sqrt(np.mean(m2/measurements)))
    xx/=scale
    t = xx[:,None,None,None] * xx[None,:,None,None] * xx.conj()[None,None,:,None] * xx.conj()[None,None,None,:]
    old = np.copy(t)
    error=[]
    indexing_set = gen_indexing_set_iiac(k, rows,cols)
    for q in range(max_its):
        p2,e = tensor_mag_proj_iicc(t,measurements,gamma, gamma_inv,indexing_set)
        r1 = 2 * p2 - t
        p1 = tensor_rank_proj_iiac(r1)
        r2 = 2 * p1 - p2
        part1 = (r2 * alpha + t * (1-alpha)) * beta
        part2 = p2 * ( 1 - beta)
        t = part1 + part2
        if e < threshold:
            break
        print(e)
        error.append(e)
        old = np.copy(t)
    factors = tucker(t, rank=[1,1,1,1])
    X = factors.factors[0].flatten()
    m = gen_meas_iiac(X,c)
    scale = np.sqrt(np.sqrt(np.mean(m/measurements)))
    X/=scale
    return X,error
#}}}
#{{{ test_inits_iicc
def test_inits_iicc():
    M = 10
    N = 10
    K = 200
    x = np.random.randn(M) + 1j * np.random.randn(M)
    y = np.random.randn(N) + 1j * np.random.randn(N)
    x/=np.linalg.norm(x)
    y/=np.linalg.norm(y)
    c = np.random.randn(K,M) + 1j * np.random.randn(K,M)
    d = np.random.randn(K,N) + 1j * np.random.randn(K,N)
    measurements = np.zeros([K],dtype=float)
    for i in range(K):
        measurements[i] = np.sum(np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2)
    X,Y,error = alt_min_tensor_init_iicc(measurements,c,d,max_its=500, threshold = 1e-2, alpha=.9, beta = .5,min_its = 100,pinv=True)
    plt.plot(error)
    plt.show()
    X/=np.linalg.norm(X)/np.linalg.norm(x)
    Y/=np.linalg.norm(Y)/np.linalg.norm(y)
    X,Y,ambig = realign_iicc(X,Y,x,y)
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

#}}}
#{{{ test_inits_iiac
def test_inits_iiac():
    M = 10
    I = 40
    x = np.random.randn(M) + 1j * np.random.randn(M)
    x/=np.linalg.norm(x)

    c = np.random.randn(I,M) + 1j * np.random.randn(I,M)

    measurements = np.zeros([I],dtype=float)
    for i in range(I):
        measurements[i] = np.sum(np.abs(np.correlate(x[::-1].conj()*c[i,::-1].conj(),(x*c[i,:]),mode='full').conj())**2)
    X,error = raar_min_tensor_init_iiac(measurements,c,max_its=10000, threshold = 1e-6, alpha=.9, beta = .5,min_its = 100,pinv=True)
    #X,error = alt_min_tensor_init_iiac(measurements,c,max_its=1000, threshold = 1e-6, alpha=.9, beta = .5,min_its = 100,pinv=False)

    X1,_ = realign_iiac(X,x)

    X = X.conj()
    X2,_ = realign_iiac(X,x)

    plt.plot(error)
    plt.show()
    plt.subplot(2,1,1)
    plt.plot(np.abs(x))
    plt.plot(np.abs(X1))
    plt.plot(np.abs(X2))
    plt.subplot(2,1,2)
    plt.plot(np.unwrap(np.angle(X1/x)))
    plt.plot(np.unwrap(np.angle(X2/x)))
    plt.show()
#}}}

if __name__=='__main__':
    test_inits_iiac()
