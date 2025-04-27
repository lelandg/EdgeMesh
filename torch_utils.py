import torch
from torch.nn import Module


class TorchModelClass(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)

        # Fully connected layers will be initialized lazily
        self.fc1 = None
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)

    def forward(self, x):
        # Pass through convolutional layers
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))

        # Flatten the tensor
        x = x.view(x.size(0), -1)

        # Initialize the first fully connected layer dynamically if not already done
        if self.fc1 is None:
            self.fc1 = torch.nn.Linear(x.size(1), 128).to(x.device)

        # Pass through fully connected layers
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)

        return x


# Example usage
if __name__ == "__main__":
    # Input should match the expected dimensions
    model = TorchModelClass(input_height=32, input_width=32)
    test_input = torch.rand(1, 3, 32, 32)
    output = model(test_input)
    print("Output shape:", output.shape)


def load_model_with_fixes(state_dict_path):
    try:
        # Load the model and corresponding state_dict
        state_dict = torch.load(state_dict_path)
        model = TorchModelClass()

        # Check for key mismatches
        model_keys = set(model.state_dict().keys())
        state_dict_keys = set(state_dict.keys())

        # Find missing and unexpected keys
        missing_keys = model_keys - state_dict_keys
        unexpected_keys = state_dict_keys - model_keys

        if missing_keys:
            print(f"Missing keys in state_dict: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys in state_dict: {unexpected_keys}")

        # Filter the state_dict to remove unexpected keys if necessary
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

        # Load the filtered state_dict
        model.load_state_dict(filtered_state_dict, strict=False)

        print("Model loaded successfully!")
        return model

    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return None


# Example usage:
# Replace "path_to_state_dict.pth" with the actual path to your state_dict
# model = load_model_with_fixes("path_to_state_dict.pth")
