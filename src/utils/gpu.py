"""GPU utilities and verification."""

import torch


def check_gpu_availability():
    """
    Check GPU availability and print device information.
    
    Returns:
        dict: GPU availability status and device information
    """
    cuda_available = torch.cuda.is_available()
    
    info = {
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "devices": []
    }
    
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            info["devices"].append({
                "index": i,
                "name": device_name,
                "capability": device_capability
            })
    
    return info


def print_gpu_info():
    """Print GPU information to console."""
    info = check_gpu_availability()
    
    print("=" * 60)
    print("GPU Configuration")
    print("=" * 60)
    print(f"CUDA Available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA Version: {info['cuda_version']}")
        print(f"cuDNN Version: {info['cudnn_version']}")
        print(f"Device Count: {info['device_count']}")
        print("\nAvailable Devices:")
        for device in info['devices']:
            print(f"  - Device {device['index']}: {device['name']}")
            print(f"    Capability: {device['capability']}")
    else:
        print("WARNING: CUDA not available. CPU-only mode will be used.")
    
    print("=" * 60)


def get_device():
    """
    Get the appropriate device (GPU or CPU).
    
    Returns:
        torch.device: Device to use for training/inference
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
