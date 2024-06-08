import numpy as np
from functools import partial
import warnings

import torch
import torch.optim as optim

from src.examples.common.tfn_utils import createGraphData, createOnesTensor
from src.examples.tensor_field_networks.nonlinearity_layer import TfnNonlinearityLayer
from src.examples.tensor_field_networks.point_convolution_layer import PointConvolutionLayer
from src.examples.tensor_field_networks.self_interaction_layer import SelfInteractionLayer
from src.examples.tensor_field_networks.concatenation_layer import ConcatenationLayer

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings(action = "ignore", message=".*ATen tensor of dims.*has strides.*")

tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
          [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
          [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
          [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
          [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L

dataset = [ np.array(points_) for points_ in tetris ]
num_classes = len(dataset)

class Readout(torch.nn.Module):
    def __init__(self, channels_in, num_classes):
        super(Readout, self).__init__()
        
        self.lin = torch.nn.Linear(
            channels_in, num_classes, bias=True, dtype=torch.cfloat)
        self.input_dims = channels_in
        self.num_classes = num_classes

    def reset_parameters(self):
        self.lin.reset_parameters()
        
    def forward(self, inputs):
        inputs = torch.mean(inputs.squeeze(),dim=0)
        inputs = self.lin.forward(inputs).unsqueeze(0)
        return inputs
    
class TetrisLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, l_value):
        super().__init__()

        self.point_convolution_ = PointConvolutionLayer(in_channels, l_value)
        self.concat_ = ConcatenationLayer()
        self.self_interation_ = \
            SelfInteractionLayer(in_channels, out_channels, l_value)
        self.nonlinearity_ = TfnNonlinearityLayer(out_channels, l_value)

    def reset_parameters(self):
        self.point_convolution_.reset_parameters()
        self.self_interation_.reset_parameters()
        self.nonlinearity_.reset_parameters()

    def forward(self, input):
        input = self.point_convolution_.forward(input)
        # input = self.concat_.forward(input.x)
        input = self.self_interation_.forward(input)
        input = self.nonlinearity_.forward(input)
        return input
    
class TetrisNetwork(torch.nn.Module):
    def __init__(self, l_value, num_classes_in = num_classes):
        super().__init__()

        # Create all layers
        assert l_value != None
        self.layers_ = [TetrisLayer(1, 4, l_value),
                       TetrisLayer(4, 4, l_value),
                       TetrisLayer(4, 4, l_value)]
        self.layers_ = torch.nn.ModuleList(self.layers_)

        # Set the readout function to be called at the end
        self.readout_ = Readout(4, num_classes_in)

    def reset_parameters(self):
        for layer in self.layers_:
            layer.reset_parameters()
        self.readout_.reset_parameters()
        
    def forward(self, data):
        for layer in self.layers_:
            data = layer.forward(data)

        return self.readout_.forward(data.x[0][0])

if __name__=="__main__": 
  kLValue = 1
  model = TetrisNetwork(l_value = kLValue)
  tetris_tensor = torch.Tensor(tetris)
  labels = torch.arange(len(tetris_tensor)).unsqueeze(-1)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)

  input_tensor = createOnesTensor(kLValue, 4)  # Each piece has 4 elements.
  tetris_tensor = torch.Tensor(tetris)
  for epoch in range(2001):  # loop over the dataset multiple times
      running_loss = 0.0
      order = np.arange(len(tetris_tensor))
  #     np.random.shuffle(order)
      for i in order:
          label = labels[i]
          # rij, rbf = rij_list[i].unsqueeze(0), rbf_list[i].unsqueeze(0)
          # zero the parameter gradients
          optimizer.zero_grad()

          data = createGraphData(tetris_tensor[i], input_tensor)
          outputs = model.forward(data)
          loss = criterion(torch.abs(outputs), label)
    
          loss.backward()
          optimizer.step()

          running_loss += loss.item()
          assert False
      
      if epoch%100 == 0:
          print('{:3.3f}'.format(running_loss/len(tetris_tensor)))
  print('Finished Training')