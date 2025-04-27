import sys

import torch
import os
import torchvision.models as models


dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Depth Models")

# Load the model
path = os.path.join(dir_path, 'metric_depth_vit_large_800k.pth')
state_dict = torch.load(path)['model_state_dict']
for key in state_dict.keys():
    print(key)
    print(state_dict[key])

sys.exit(0)

model.eval()

