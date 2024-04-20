
import torch
from torch.nn import Linear, Module, Parameter, ReLU, Sequential
from torch_geometric.data import Data, DataLoader
from typing import Any, Callable, Generic, List, TypeVar

from cmath import sqrt
import math
import random
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


from src.examples.tensor_field_networks import TfnNonlinearityLayer
from src.examples.tensor_field_networks import PointConvolutionLayer
from src.examples.tensor_field_networks import SelfInteractionLayer
from src.examples.tensor_field_networks import createGraphData



# This file defines an implementation of the "gravity" test described in the
# original Tensor Field Networks paper. This test was chosen because the data
# used is minimal enough for a CPU to be sufficient.

class GravityTFN(Module):
  def __init__(self, num_classes, num_features, l_filter, point_positions):
    super().__init__()

    self.conv = PointConvolutionLayer(num_features, num_features, l_filter, point_positions)
    self.self_interaction = SelfInteractionLayer(num_features, 16)
    self.nonlin = TfnNonlinearityLayer(16, 16, torch.relu)
    self.final_lin = Linear(16, num_classes)  # Output layer for classification

  def forward(self, data):
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    # Pass through network layers
    x = self.conv(x, edge_index, edge_attr)
    x = self.self_interaction(x)
    x = self.nonlin(x)
    out = self.final_lin(x)
    return out

def accelerations(points, masses=None):
    """
    inputs:
    -points: a list of 3-tuples of point coordinates
    -masses: a list (of equal length N) of masses
    
    returns: 
    -shape [N, 3] numpy array of accelerations under Newtonian gravity
    """
    EPSILON = 0.0000001
    accels = []
    if masses is None:
        masses = [1.0 for _ in range(len(points))]
    for i, ri_ in enumerate(points):
        accel_vec = np.array((0., 0., 0.))
        for j, rj_ in enumerate(points):
            rij_ = ri_ - rj_
            dij_ = np.linalg.norm(rij_)
            if (ri_ != rj_).any():
                accel_update = -rij_ / (np.power(dij_, 3) + EPSILON) * masses[j]
                accel_vec += accel_update
        accels.append(accel_vec)
    assert len(accels) == len(points)
    return np.array(accels)


def random_points_and_masses(max_points=10, min_mass=0.5, max_mass=2.0, 
                             max_coord=5, min_separation=0.5):
    """
    returns:
    -shape [N, 3] numpy array of points, where N is between 2 and max_points
    -shape [N] numpy array of masses
    """
    num_points = random.randint(2, max_points)
    candidate_points = []
    for point in range(num_points):
        candidate_points.append(
            np.array([random.uniform(-max_coord, max_coord) for _ in range(3)]))
    
    # remove points that are closer than min_separation
    output_points = []
    for point in candidate_points:
        include_point = True
        for previous_point in output_points:
            if np.linalg.norm(point - previous_point) < min_separation:
                include_point = False
        if include_point:
            output_points.append(point)
    
    points_ = np.array(output_points)
    masses_ = np.random.rand(len(output_points)) * (max_mass - min_mass) + min_mass
    return points_, masses_

def visualize_radial_function(model, min_separation):
    # Hyperparameters
    rbf_low = 0.0  # Lower bound of RBF range
    rbf_high = 2.0  # Upper bound of RBF range
    rbf_count = 30  # Number of RBFs    

    # Calculate RBF spacing
    rbf_spacing = (rbf_high - rbf_low) / rbf_count
    x_vals = np.linspace(rbf_low, rbf_high, 100)

    radial_fig = plt.figure(figsize=(5, 2.5))
    ax = radial_fig.add_subplot(1, 1, 1)

    min_index_cutoff = int((1 / sqrt(40.) - rbf_low) / rbf_spacing)
    ax.plot(x_vals[min_index_cutoff:], [-1 / r_**2 for r_ in x_vals[min_index_cutoff:]], "r:", lw=3, label="$-1/r^2$")

    # Accessing the radial function parameters
    # Adjust this based on your implementation
    r_layer = model.conv.mlp  # Assuming the MLP is within the PointConvolutionLayer

    y_vals = []
    for x_val in x_vals:
        # Evaluate the radial function for each x_val
        # Adjust this based on how you input data to the R function MLP
        input_tensor = torch.tensor([x_val]).unsqueeze(0)  # Assuming the MLP takes a single distance value as input
        y_val = r_layer(input_tensor).detach().numpy()  # Detach from gradient and convert to NumPy
        y_vals.append(y_val)

    ax.plot(x_vals, y_vals, 'b-', label="Learned Radial Function")

    ax.plot([min_separation, min_separation], [10, -50], 'k--', label="Minimum distance of points")
    ax.set_ylabel("Output of learned radial function.")
    ax.set_xlabel("Radial distance")
    ax.set_xlim(0., 2.0)
    ax.set_ylim(-10, 1.)
    ax.legend()
    plt.show()

if __name__=="__main__": 
    # Hyperparameters
    max_steps = 1001
    validation_size = 1000
    print_freq = 25
    num_radial_basis = 30  # Adjust as needed
    in_channels = 1  # Mass as input
    out_channels = 3  # Acceleration vector as output

    # Model Initialization
    num_points = 10  # Maximum number of points
    num_features = 1  # Mass as the feature
    l_filter = 1  # Filter order for learning accelerations (vector quantity)
    point_positions = torch.randn(num_points, 3)  # Initialize random positions
    model = GravityTFN(out_channels, num_features, l_filter, point_positions)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Training Loop
    min_separation = 0.5
    for step in range(max_steps):
        if step % 100 == 0:
            print(step)

        # Generate random points and masses
        rand_points, rand_masses = \
            random_points_and_masses(max_points=10,
                                     min_separation=min_separation)
        
        # Create a complete graph with masses as features
        data = createGraphData(torch.tensor(rand_masses).unsqueeze(1))
        
        # Calculate accelerations 
        accel = torch.tensor(accelerations(rand_points)).unsqueeze(1)

        # Train the model
        optimizer.zero_grad()
        outputs = model(data.x, data.edge_index, data.pos)
        loss = criterion(outputs, accel)
        loss.backward()
        optimizer.step()

        # Validation
        if step % print_freq == 0:
            loss_sum = 0.0
            for _ in range(validation_size):
                # Generate validation points and masses
                val_points, val_masses = random_points_and_masses(max_points=10, min_separation=min_separation)
                
                # Create validation data
                val_data = createGraphData(torch.tensor(val_masses).unsqueeze(1))
                val_data.pos = torch.tensor(val_points)  # Set node positions

                # Calculate validation accelerations
                val_accel = torch.tensor(accelerations(val_points)).unsqueeze(1)

                # Validate the model
                val_outputs = model(val_data)
                validation_loss = criterion(val_outputs, val_accel)
                loss_sum += validation_loss.item()

            avg_val_loss = loss_sum / validation_size
            print(f"Step {step}: validation loss = {avg_val_loss:.3f}")

    # Draw a plot to show it visually, like in the original paper.
    visualize_radial_function(model, min_separation)
    