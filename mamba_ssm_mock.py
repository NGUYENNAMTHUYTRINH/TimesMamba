# Mock implementation of mamba-ssm for testing purposes
# This allows the project to import without the actual mamba-ssm package

import torch
import torch.nn as nn

class selective_scan_fn:
    """Mock selective scan function"""
    @staticmethod
    def apply(*args, **kwargs):
        # Return dummy tensor for testing
        return torch.randn(args[0].shape[0], args[0].shape[1], args[0].shape[2])

# Create mock module structure
import sys
import types

# Create mamba_ssm module
mamba_ssm = types.ModuleType('mamba_ssm')
sys.modules['mamba_ssm'] = mamba_ssm

# Create ops submodule
ops = types.ModuleType('ops')
mamba_ssm.ops = ops
sys.modules['mamba_ssm.ops'] = ops

# Create selective_scan_interface submodule
selective_scan_interface = types.ModuleType('selective_scan_interface')
ops.selective_scan_interface = selective_scan_interface
sys.modules['mamba_ssm.ops.selective_scan_interface'] = selective_scan_interface

# Add the mock function to the module
selective_scan_interface.selective_scan_fn = selective_scan_fn

print("Mamba SSM mock module loaded successfully")