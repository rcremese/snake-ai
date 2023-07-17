from pathlib import Path
import torch
import os


# SILU = SWISH with \beta=1
class MLP(torch.nn.Module):
    def __init__(
        self, input_size, output_size, hidden_sizes=[128], act_fn=torch.nn.SiLU()
    ):
        """
        Args:
            act_fn: Object of the activation function that should be used as non-linearity in the network.
            input_size: Size of the input images in pixels
            num_classes: Number of classes we want to predict
            hidden_sizes: A list of integers specifying the hidden layer sizes in the NN
        """
        super().__init__()
        # Create the network based on the specified hidden sizes
        layers = []
        layer_sizes = [input_size] + hidden_sizes
        layer_size_last = layer_sizes[0]
        for layer_size in layer_sizes[1:]:
            layers += [torch.nn.Linear(layer_size_last, layer_size), act_fn]
            layer_size_last = layer_size
        layers += [torch.nn.Linear(layer_sizes[-1], output_size)]
        # nn.Sequential summarizes a list of modules into a single module, applying them in sequence
        self.layers = torch.nn.Sequential(*layers)

        # We store all hyperparameters in a dictionary for saving and loading of the model
        self.config = {
            "input_size": input_size,
            "num_classes": output_size,
            "hidden_sizes": hidden_sizes,
        }

    def forward(self, x: torch.Tensor):
        return self.layers(x)

    def save(self, filename="mlp.pth"):
        model_path = os.getenv("MODEL_PATH")
        if model_path is None:
            model_path = Path(__file__).parents[4].joinpath("models")
        filepath = Path(model_path).joinpath(filename).resolve()
        torch.save(self.state_dict(), filepath)
