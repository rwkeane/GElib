import numpy as np
from functools import partial

import torch
import torch.optim as optim

from src.examples.tensor_field_networks.nonlinearity_layer import TfnNonlinearityLayer
from src.examples.tensor_field_networks.point_convolution_layer import PointConvolutionLayer
from src.examples.tensor_field_networks.self_interaction_layer import SelfInteractionLayer
from src.examples.tensor_field_networks.concatenation_layer import ConcatenationLayer
from src.examples.tensor_field_networks.tfn_utils import createGraphDataFactory

tetris = [[(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 1, 0)],  # chiral_shape_1
          [(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, -1, 0)], # chiral_shape_2
          [(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)],  # square
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)],  # line
          [(0, 0, 0), (0, 0, 1), (0, 1, 0), (1, 0, 0)],  # corner
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 0)],  # T
          [(0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 1, 1)],  # zigzag
          [(0, 0, 0), (1, 0, 0), (1, 1, 0), (2, 1, 0)]]  # L

dataset = [np.array(points_) for points_ in tetris]
num_classes = len(dataset)

class Readout(torch.nn.Module):
    def __init__(self, channels_in, num_classes):
        super(Readout, self).__init__()
        
        self.lin = torch.nn.Linear(channels_in, num_classes, bias=True)
        self.input_dims = channels_in
        self.num_classes = num_classes
        
    def forward(self, inputs):
        inputs = torch.mean(inputs.squeeze(),dim=0)
        inputs = self.lin.forward(inputs).unsqueeze(0)
        return inputs
    
class TetrisLayer(torch.nn.module):
    def __init__(self, in_channels, out_channels, l_value, data_factory):
        self.data_factory = partial(data_factory, in_channels)

        self.point_convolution = \
            PointConvolutionLayer(in_channels, in_channels, l_value)
        self.concat = ConcatenationLayer()
        self.self_interation = \
            SelfInteractionLayer(in_channels, out_channels, l_value)
        self.nonlinearity = TfnNonlinearityLayer(out_channels, l_value)

    def forward(self, input):
        input = self.point_convolution.forward(self.data_factory(input))
        input = self.concat.forward(input.x)
        input = self.self_interation.forward(input)
        return self.nonlinearity.forward(input)
    
class TetrisNetwork(torch.nn.Module):
    def __init__(self, data_factory, num_classes_in = num_classes):
        super().__init__()
        self.as_embedding = SelfInteractionLayer(input_dim = 1, output_dim = 1, bias = False)

        # Create all layers
        self.layers = [TetrisLayer(1,4, data_factory),
                       TetrisLayer(4,4, data_factory),
                       TetrisLayer(4,4, data_factory)]
        self.layers = torch.nn.ModuleList(self.layers)

        # Set the readout function to be called at the end
        self.readout = Readout(4, num_classes_in)
        
    def forward(self, rbf, rij):
        # TODO: Replace this with a call to GELib.

        # Start with all 1s, because the tetris blocks are always seen from the
        # same direction here.
        embed = self.as_embedding.forward(
            torch.ones(1,4,1,1).repeat([rbf.size()[0],1,1,1]))   
        result = {0: [embed]}
        for layer in self.layers:
            result = layer.forward(result)

        assert result.dim() == 2
        first, second = result.size()
        assert first == 1
        assert second == 1
        return self.readout.forward(result[0][0])

if __name__=="__main__": 
  model = TetrisNetwork(createGraphDataFactory(TODO))
  tetris_tensor = torch.Tensor(tetris)

  criterion = torch.nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  for epoch in range(2001):  # loop over the dataset multiple times
  # for epoch in range(100):  # loop over the dataset multiple times
      running_loss = 0.0
      order = np.arange(len(tetris_tensor))
  #     np.random.shuffle(order)
      for i in order:
          label = labels[i]
          # rij, rbf = rij_list[i].unsqueeze(0), rbf_list[i].unsqueeze(0)
          # zero the parameter gradients
          optimizer.zero_grad()
          outputs = model(rbf, rij)
          loss = criterion(outputs, label)
    
          loss.backward()
          optimizer.step()

          running_loss += loss.item()
      
      if epoch%100 == 0:
          print('{:3.3f}'.format(running_loss/len(tetris_tensor)))
  print('Finished Training')