kernel_optimization_tricks = """
shared memory caching
coalescing memory access
vectorized loads/stores
thread divergence
thread coarsening
"""

generate_naive_kernel = """
generate a naive CUDA kernel for the following pytorch operation in row major order:
{}
"""

class Prompts:
    def __init__(self):
        self.prompts = {
            "kernel_optimization_tricks": self.kernel_optimization_tricks
        }

    def get_prompt(self, prompt_name):
        return self.prompts[prompt_name]

