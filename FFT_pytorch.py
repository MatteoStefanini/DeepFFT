import numpy as np
import torch
from torch.autograd import Variable


def FFT_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT with Numpy"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above, and should be a power of 2
    N_min = min(N, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd, X_even - factor * X_odd])

    return X.ravel()


def FFT_torch_simple_signal(x):
    # Enable next 2 lines for if you come in with a numpy tensor otherwise MUST be commented
    #x = np.asarray(x, dtype=float)
    #x = Variable(torch.from_numpy(x), requires_grad=True).double().cuda()
    N_samples = x.shape[0]

    if np.log2(N_samples) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above, and should be a power of 2
    N_min = min(N_samples, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = Variable(torch.arange(N_min), requires_grad = False).double() # .view(1,N,1)
    k = n[:, None]
    M_r = torch.cos(-2 * np.pi * n * k / N_min)  
    M_i = torch.sin(-2 * np.pi * n * k / N_min) 

    X_r = torch.mm(M_r, x.view(N_min, -1)) 
    X_i = torch.mm(M_i, x.view(N_min, -1)) 

    # build-up each level of the recursive calculation all at once
    while X_r.shape[0] < N:
        X_r_even = X_r[:, : X_r.shape[1] // 2]
        X_r_odd = X_r[:, X_r.shape[1] // 2:]
        X_i_even = X_i[:, : X_i.shape[1] // 2]
        X_i_odd = X_i[:, X_i.shape[1] // 2 :]
        factor_r = Variable(torch.cos(-1 * np.pi * torch.arange(X_r.shape[0]).double() / X_r.shape[0]), requires_grad = False).double().view(-1,1)
        factor_i = Variable(torch.sin(-1 * np.pi * torch.arange(X_i.shape[0]).double() / X_i.shape[0]), requires_grad = False).double().view(-1,1)
        P_r = factor_r * X_r_odd - factor_i * X_i_odd
        P_i = factor_r * X_i_odd + factor_i * X_r_odd
        X_r = torch.cat([X_r_even + P_r, X_r_even - P_r]) 
        X_i = torch.cat([X_i_even + P_i, X_i_even - P_i])

    R = X_r.data.numpy()
    I = X_i.data.numpy()
    X_ = np.complex128(R + 1j* I)
    return X_.ravel()


def FFT_torch(x):
    '''
    FFT through pytorch cuda 9.1
    :param x: a variable Cuda with samples in time domain and shape (batch_size, feature_size, samples_size)
    :return: 2 variables Cuda with real and imaginary part of coefficients in frequency domain with same input size
    '''

    batch_size = x.shape[0]
    features = x.shape[1]
    N_samples = x.shape[2]

    if np.log2(N_samples) % 1 > 0:
        raise ValueError("size of sequence must be a power of 2")

    # N_min should be a power of 2 and is equivalent to the stopping condition above
    N_min = min(N_samples, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = Variable(torch.arange(N_min), requires_grad = False).double().cuda()
    k = n[:, None]
    M_r = torch.cos(-2 * np.pi * n * k / N_min).view(N_min, -1)
    M_i = torch.sin(-2 * np.pi * n * k / N_min).view(N_min, -1)
    X_r = torch.matmul(M_r, x.view(batch_size, features, N_min, -1))
    X_i = torch.matmul(M_i, x.view(batch_size, features, N_min, -1))

    # build-up each level of the recursive calculation all at once
    while X_r.shape[2] < N_samples:
        X_r_even = X_r[:,:,:, : X_r.shape[3] // 2]
        X_r_odd =  X_r[:,:,:, X_r.shape[3] // 2:]
        X_i_even = X_i[:,:,:, : X_i.shape[3] // 2]
        X_i_odd =  X_i[:,:,:, X_i.shape[3] // 2:]
        factor_r = Variable(torch.cos(-1 * np.pi * torch.arange(X_r.shape[2]).double() / X_r.shape[2]),
                            requires_grad = False).cuda().view(-1, 1)
        factor_i = Variable(torch.sin(-1 * np.pi * torch.arange(X_i.shape[2]).double() / X_i.shape[2]),
                            requires_grad = False).cuda().view(-1, 1)
        P_r = factor_r * X_r_odd - factor_i * X_i_odd
        P_i = factor_r * X_i_odd + factor_i * X_r_odd
        X_r = torch.cat([X_r_even + P_r, X_r_even - P_r], dim=2)
        X_i = torch.cat([X_i_even + P_i, X_i_even - P_i], dim=2)

    # return the torch output tensors real and imag parts of coefficients :
    return X_r.view(batch_size,features,N_samples), X_i.view(batch_size,features,N_samples)


def IFFT_torch(x_r, x_i):

    batch_size = x_r.shape[0]
    features = x_r.shape[1]
    N_samples = x_r.shape[2]

    if np.log2(N_samples) % 1 > 0:
        raise ValueError("size of sequence must be a power of 2")

    # N_min should be a power of 2 and is equivalent to the stopping condition above
    N_min = min(N_samples, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = Variable(torch.arange(N_min), requires_grad = False).double().cuda()
    k = n[:, None]
    M_r = torch.cos(2 * np.pi * n * k / N_min).view(N_min, -1)
    M_i = torch.sin(2 * np.pi * n * k / N_min).view(N_min, -1)
    X_r = torch.matmul(M_r, x_r.view(batch_size, features, N_min, -1))
    X_i = torch.matmul(M_i, x_i.view(batch_size, features, N_min, -1))
    X_r = X_r - X_i
    X_i = torch.matmul(M_i, x_r.view(batch_size, features, N_min, -1)) + torch.matmul(M_r, x_i.view(batch_size, features, N_min, -1))

    # build-up each level of the recursive calculation all at once
    while X_r.shape[2] < N_samples:
        X_r_even = X_r[:,:,:, : X_r.shape[3] // 2]
        X_r_odd =  X_r[:,:,:, X_r.shape[3] // 2:]
        X_i_even = X_i[:,:,:, : X_i.shape[3] // 2]
        X_i_odd =  X_i[:,:,:, X_i.shape[3] // 2:]
        factor_r = Variable(torch.cos(1 * np.pi * torch.arange(X_r.shape[2]).double() / X_r.shape[2]),
                            requires_grad = False).cuda().view(-1,1)
        factor_i = Variable(torch.sin(1 * np.pi * torch.arange(X_i.shape[2]).double() / X_i.shape[2]),
                            requires_grad = False).cuda().view(-1,1)
        P_r = factor_r * X_r_odd - factor_i * X_i_odd
        P_i = factor_r * X_i_odd + factor_i * X_r_odd
        X_r = torch.cat([X_r_even + P_r, X_r_even - P_r], dim=2)
        X_i = torch.cat([X_i_even + P_i, X_i_even - P_i], dim=2)

    # rescaling the samples dividing by N
    X_r /= N_samples
    X_i /= N_samples # must be very close to 0 so we can omit it
    # And return only the real value that are our original samples in time domain
    return X_r.view(batch_size,features,N_samples)
