import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.signal import remez

def generate_low_bandwidth_complex_gaussian(M,N,K,bw):
    x = np.random.randn(M) + 1j * np.random.randn(M)
    y = np.random.randn(N) + 1j * np.random.randn(N)
    x/=np.linalg.norm(x)/np.linalg.norm(y)
    filter_size = int(1./np.max(np.array([1./M,1./N, bw]))) + 1
    

    c = np.random.randn(K,M) + 1j * np.random.randn(K,M)
    d = np.random.randn(K,N) + 1j * np.random.randn(K,N)
    measurements = np.zeros([K, N + M -1],dtype=float)
    for i in range(K):
        measurements[i,:] = np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2
    return x,y,c,d,measurements


def generate_complex_gaussian(M,N,K):
    x = np.random.randn(M) + 1j * np.random.randn(M)
    y = np.random.randn(N) + 1j * np.random.randn(N)
    #x/=np.linalg.norm(x)/np.linalg.norm(y)

    c = np.random.randn(K,M) + 1j * np.random.randn(K,M)
    d = np.random.randn(K,N) + 1j * np.random.randn(K,N)
    measurements = np.zeros([K, N + M -1],dtype=float)
    for i in range(K):
        measurements[i,:] = np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2
    return x,y,c,d,measurements

def generate_complex_uniform_IIAC(M,I):
    x = np.random.randn(M) + 1j * np.random.randn(M)
    x/=np.linalg.norm(x) # DJRO TODO For convenient sizes
    c = np.exp(1j * np.random.uniform(0,2*np.pi, [I,M]))
    measurements = measurement_iiac(x,c)
    return x,c,measurements

def generate_complex_gaussian_IIAC(M,I):
    x = np.random.randn(M) + 1j * np.random.randn(M)
    x/=np.linalg.norm(x) # DJRO TODO For convenient sizes
    c = (np.random.randn(I,M) + 1j * np.random.randn(I,M)) / np.sqrt(2)
    measurements = measurement_iiac(x,c)
    return x,c,measurements

def generate_complex_gaussian_IIAC_half_bandwidth(M,I):
    x = np.random.randn(M) + 1j * np.random.randn(M)
    c = np.random.randn(I,M) + 1j * np.random.randn(I,M)
    zero = np.abs(np.fft.fftfreq(len(x))) >=.25
    X = np.fft.fft(x)
    X[zero]=0
    x = np.fft.ifft(X)
    x/=np.linalg.norm(x) # DJRO TODO For convenient sizes
    # ONLY Unknown vector is half-bandwidth
    #C = np.fft.fft(c,axis=1)
    #C[:,zero]=0
    #c = np.fft.ifft(C,axis=1)
    measurements = measurement_iiac(x,c)
    return x,c,measurements

def generate_complex_gaussian_IICC(M,N,K):
    x = np.random.randn(M) + 1j * np.random.randn(M)
    y = np.random.randn(N) + 1j * np.random.randn(N)
    x/=np.linalg.norm(x)
    y/=np.linalg.norm(y)
    #x/=np.linalg.norm(x)/np.linalg.norm(y)

    c = np.random.randn(K,M) + 1j * np.random.randn(K,M)
    d = np.random.randn(K,N) + 1j * np.random.randn(K,N)
    measurements = np.zeros([K],dtype=float)
    for i in range(K):
        measurements[i] = np.sum(np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2)
    return x,y,c,d,measurements

def generate_blind_FROG(M,N,K, bw):
    x = np.random.randn(M) + 1j * np.random.randn(M)
    win = 1/bw
    #x *= np.fft.fftshift(np.exp(-1 * np.linspace(-win,win,M)**2))
    y = np.random.randn(N) + 1j * np.random.randn(N)
    #y *= np.fft.fftshift(np.exp(-1 * np.linspace(-win,win,M)**2))
    x = np.fft.ifft(x)
    y = np.fft.ifft(y)
    x/=np.linalg.norm(x)/np.linalg.norm(y)
    c = np.ones([K,M],dtype=complex)
    omega = np.linspace(-.5,.5, K)
    T = np.arange(N)
    d = np.exp(-1j * 2 * np.pi * np.outer(omega,T))
    measurements = np.zeros([K, N + M -1],dtype=float)
    for i in range(K):
        measurements[i,:] = np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2
    return x,y,c,d,measurements

def generate_FROG(M,K,bw):
    x = np.random.randn(M) + 1j * np.random.randn(M)
    win = 1/bw
    x *= np.fft.fftshift(np.exp(-1 * np.linspace(-win,win,M)**2))
    x = np.fft.ifft(x)
    y = np.copy(x).conj()
    c = np.ones([K,M],dtype=complex)
    omega = np.linspace(0,1 - 1./K, K)
    T = np.arange(M)
    d = np.exp(1j * 2 * np.pi * np.outer(omega,T))
    measurements = np.zeros([K, M + M -1],dtype=float)
    for i in range(K):
        measurements[i,:] = np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2
    return x,c,d,measurements

def measurement_iiac(x,c):
    M = c.shape[1]
    I = c.shape[0]
    measurements = np.zeros([I],dtype=float)
    for i in range(I):
        g_i = np.convolve(x * c[i,:], x * c[i,:],mode='full')[::-1]
        measurements[i] = np.sum(np.abs(g_i)**2)
    return measurements
    
def measurement_iicc(x,y,c,d):
    M = c.shape[1]
    N = d.shape[1]
    K = c.shape[0]
    measurements = np.zeros([K],dtype=float)
    for i in range(K):
        measurements[i] = np.sum(np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2)
    return measurements

def measurement(x,y,c,d):
    M = c.shape[1]
    N = d.shape[1]
    K = c.shape[0]
    measurements = np.zeros([K, N + M -1],dtype=float)
    for i in range(K):
        measurements[i,:] = np.abs(np.correlate(y*d[i,:],(x*c[i,:]),mode='full').conj())**2
    return measurements

def test_measurement():
    M=32
    K = 128
    bw = .1
    x,c,d,measurements = generate_FROG(M,K,bw)

    plt.imshow(measurements)
    plt.show()

def test_iiac():
    M = 32
    I = 1024
    #x,c,meas = generate_complex_gaussian_IIAC_half_bandwidth(M,I)
    #meas1 = measurement_iiac(x,c)
    #meas2 = measurement_iiac(x.conj()[::-1],c)
    ##meas3 = measurement_iiac(x[::-1],c)
    #meas4 = measurement_iiac(x * np.exp(1j * 2 * np.pi * .1 * np.arange(M)),c)
    #plt.plot(meas1 - meas2)
    ##plt.plot(meas1 - meas3)
    #plt.plot(meas1 - meas4)
    #plt.show()

    x = np.zeros(M,dtype=complex)
    c = np.random.randn(M-1,M) + 1j * np.random.randn(M-1,M)
    u,s,v = np.linalg.svd(c)
    x = v[-1,:].conj()

    inner = np.abs(c @ x)
    meas1 = measurement_iiac(x,c)
    print(meas1)
    plt.plot(meas1)
    plt.plot(inner)
    plt.show()


if __name__ == '__main__':
    test_iiac()

