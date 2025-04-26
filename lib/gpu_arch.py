import subprocess
import os
import torch

class GPUArch:
    def __init__(self):
        self.arch = self.get_arch()

    def get_arch(self):
        try:
            # Step 1: Clone the repository
            print("Cloning the NVIDIA CUDA samples repository...")
            subprocess.run(['git', 'clone', 'https://github.com/NVIDIA/cuda-samples'], check=True)
            
            repo_dir = 'cuda-samples'
            
            # Full path for subsequent commands
            device_query_dir = os.path.join(repo_dir, 'Samples', '1_Utilities', 'deviceQuery')
            build_dir = os.path.join(device_query_dir, 'build')
            
            # Step 3: Create the build directory
            os.makedirs(build_dir, exist_ok=True)
            
            # Step 4: Run cmake in the build directory
            print("Running cmake...")
            subprocess.run(['cmake', '..'], cwd=build_dir, check=True)
            
            # Step 5: Run make in the build directory
            print("Running make...")
            subprocess.run(['make'], cwd=build_dir, check=True)
            
            # Step 6: Run ./deviceQuery and capture output
            print("Running deviceQuery...")
            result = subprocess.run(['./deviceQuery'], cwd=build_dir, capture_output=True, text=True, check=True)
            
            # Step 7: Write the output to a text file
            output_file = 'deviceQuery_output.txt'
            with open(output_file, 'w') as f:
                f.write(result.stdout)
            
            print(f"Output from deviceQuery has been saved to {output_file}")
            
            # New code added here to write the PyTorch version
            with open('import_torch.txt', 'w') as f:
                f.write(torch.__version__)
            print("PyTorch version has been saved to import_torch.txt")
            
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
            # You can add more detailed error handling here if needed
        except Exception as e:
            print(f"Unexpected error: {e}")
    
    def device_query(self):
        with open('deviceQuery_output.txt', 'r') as f:
            return f.read()
    
    def torch_version(self):
        with open('import_torch.txt', 'r') as f:
            return f.read()


