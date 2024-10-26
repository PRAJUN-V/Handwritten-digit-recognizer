import h5py

def print_group_contents(group, indent=0):
    """Recursively print the contents of an HDF5 group."""
    for item_name in group:
        item = group[item_name]
        if isinstance(item, h5py.Dataset):
            print("  " * indent + f"Dataset '{item_name}': shape {item.shape}, dtype {item.dtype}")
        elif isinstance(item, h5py.Group):
            print("  " * indent + f"Group '{item_name}':")
            # Recursively explore the nested group
            print_group_contents(item, indent + 1)

# Open the .h5 file
file_path = 'mnist.h5'  # Replace with your file path
with h5py.File(file_path, 'r') as f:
    # Explore the model weights
    model_weights_group = f['model_weights']
    
    print("Contents of 'model_weights':")
    print_group_contents(model_weights_group)
