import subprocess
import os
import torch

class Profilers:
    def __init__(self):
        self.nsys_available = self.check_nsys()
        self.ncu_available = self.check_ncu()
        
    def check_nsys(self):
        try:
            result = subprocess.run(['nsys'], capture_output=True, text=True, check=False)
            if "The most commonly used nsys commands are" in result.stdout:
                print("nsys ✅")
                return True
            else:
                print("nsys is available but did not produce the expected output.")
                return False
        except FileNotFoundError:
            print("Error: nsys executable not found on the system.")
            raise RuntimeError("nsys is not available on the system.") from None  # This will stop the script
        except Exception as e:
            print(f"Unexpected error while checking nsys availability: {str(e)}")
            raise RuntimeError("nsys is not available on the system.") from None  # This will stop the script
    def check_ncu(self):
        try:
            ncu_result = subprocess.run(['ncu'], capture_output=True, text=True, check=True)
            if "usage: ncu [options]" in ncu_result.stdout:
                print("ncu ✅")  # You can modify this as needed
                return True
            else:
                print("ncu is available but did not produce the expected output.")
                return False
        except subprocess.CalledProcessError:
            print("ncu is not available on the system.")  # No error raised
        
