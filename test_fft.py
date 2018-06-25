import torch
from torch.autograd import Variable
import FFT_pytorch as fft
import numpy as np
import time


def test_fft_numpy():
    """ This test is for numpy correctness and time comparison"""
    #np.random.seed(1)
    x = np.random.random((4, 2048, 1024))  # batch, features, sequence
    start_time = time.clock()
    nump = np.fft.fft(x)
    nump_time = time.clock() - start_time
    print(nump, '\n\n')

    x = Variable(torch.from_numpy(x), requires_grad=True).double().cuda()
    start_time = time.clock()
    torc_r, torc_i = fft.FFT_torch(x)
    torch_time = time.clock() - start_time
    R = torc_r.cpu().data.numpy()
    I = torc_i.cpu().data.numpy()
    torc = np.complex128(R + 1j * I)
    print(torc)

    print("\nFFT_torch is equal to np.fft? : ", np.allclose(nump, torc))
    print("\nTime numpy fft: {:.3f}ms ".format((nump_time) * 1000))
    print("\nTime my FFT_torch: {:.3f}ms".format((torch_time) * 1000))


def test_backprop():
    """This test is for backprop gradient check """
    x = np.random.random((2, 8, 1024))  # batch, features, sequence
    x_input = Variable(torch.from_numpy(x).cuda(), requires_grad = True).double()#.cuda()

    np.random.seed(1)
    x_t = np.random.random((2, 8, 1024))
    nump = np.fft.fft(x_t)
    x_target_real = Variable(torch.from_numpy(nump.real).cuda()).double()#.cuda()
    x_target_imm = Variable(torch.from_numpy(nump.imag).cuda()).double()#.cuda()

    # Next part is for backprop gradient check
    for i in range(250):
        torc_r, torc_i = fft.FFT_torch(x_input)

        loss = torch.nn.MSELoss()
        output = loss(torc_r, x_target_real)
        output += loss(torc_i, x_target_imm)

        output.backward()
        print("MSE loss at iteration ", i, " : ", output.data)

        x_input.data = x_input.data - 0.06 * x_input.grad.data
        x_input.grad = 0 * x_input.grad


def test_IFFT():
    """Next Part is for correctness of Inverse FFT"""
    import matplotlib.pyplot as plt
    np.random.seed(1)
    x_t = np.random.random((1, 1, 64))
    x_t = Variable(torch.from_numpy(x_t), requires_grad=True).double().cuda()
    torc_r, torc_i = fft.FFT_torch(x_t)

    x_reconstructured = fft.IFFT_torch(torc_r, torc_i)

    plt.plot(x_t.view(-1).cpu().data.numpy(),color='red')
    plt.plot(x_reconstructured.view(-1).cpu().data.numpy(),color='blue')
    plt.show()
    #energy = torc_r.view(-1)**2 + torc_i.view(-1)**2
    #plt.plot(energy.cpu().data.numpy())
    #plt.show()
    print("np all close: ", np.allclose(x_t.view(-1).cpu().data.numpy(),x_reconstructured.view(-1).cpu().data.numpy()))
    print("tot Squared Error : ", sum(sum(sum(pow(x_t - x_reconstructured,2)))).data)


def test_fft_cuFFT():
    """Next part is for comparison between this FFT and cuFFT pytorch-fft https://github.com/locuslab/pytorch_fft"""
    import pytorch_fft.fft as pyfft
    import time

    # A = time domain data  B = frequency domain data
    A_real, A_imag = torch.randn(4, 2048, 1024).cuda().double(), torch.zeros(4, 2048, 1024).cuda().double()
    start_time = time.clock()
    B_real, B_imag = pyfft.fft(A_real, A_imag)
    py_fft_time = time.clock() - start_time

    x = Variable(A_real)  # my FFT takes a Variable directly, not a tensor
    start_time = time.clock()
    my_B_real, my_B_imag = fft.FFT_torch(x)
    my_fft_time = time.clock() - start_time

    B_real = Variable(B_real.double().cuda())
    B_imag = Variable(B_imag.double().cuda())

    print("FFT_torch is equal to pytorch-fft? : ", np.allclose(my_B_real.cpu().data.numpy(), B_real.cpu().data.numpy()))
    print("\nTime pytorch-fft: {:.3f}ms ".format((py_fft_time) * 1000))
    print("\nTime my FFT_torch: {:.3f}ms".format((my_fft_time) * 1000))


def test_torch_ftt():
    A_real, A_imag = torch.randn(4, 2048, 1024).cuda().double(), torch.zeros(4, 2048, 1024).cuda().double()
    start = time.time()
    x = Variable(A_real).to('cuda:0')
    xr, xi = fft.FFT_torch(x)
    time1 = time.time() - start
    print("time my fft: ", time1)

    im = torch.zeros_like(x).view(x.shape[0],x.shape[1],x.shape[2],1)
    x2 = x.view(x.shape[0],x.shape[1],x.shape[2],1)
    xtt = torch.cat([x2,im], dim=3)

    start = time.time()
    tr = torch.fft(xtt, 1)
    time2 = time.time() - start
    print("time torch fft: ", time2)

    xm = torch.cat([xr.view(x.shape[0],x.shape[1],x.shape[2],1),xi.view(x.shape[0],x.shape[1],x.shape[2],1)], dim=3)
    print(np.allclose(tr.detach().cpu().numpy(), xm.detach().cpu().numpy()))


if __name__ == "__main__":
    #test_fft_numpy()
    #test_backprop()
    #test_IFFT()
    #test_fft_cuFFT()
    test_torch_ftt()