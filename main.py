from lib.gpu_arch import GPUArch
from lib.profilers import Profilers
from lib.models import Models
from lib.prompts import Prompts
from lib.initialization import initialize_active_directory

def main():
    # gpu_arch = GPUArch()
    # profilers = Profilers()
    
    # print(gpu_arch.device_query())
    # print(gpu_arch.torch_version())

    # profilers.nsys_available
    # profilers.ncu_available

    initialize_active_directory()

    models = Models()
    print(models.gemini_2_5_flash("Hello, how are you?"))

if __name__ == "__main__":
    main()