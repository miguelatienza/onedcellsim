from onedcellsim.compression import compress
import torch

def test_output_shape():

    net = compress.SummaryNet()
    shape = (2, 3*2*60*15)
    x = torch.randn(shape)
    x = net(x)
    print(x.shape)
    return
