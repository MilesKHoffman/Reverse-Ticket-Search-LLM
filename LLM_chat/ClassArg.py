# Allows for quick configuration throughout the program.
class Args:
    def __init__(
        self, 
        directory: str = "./data", 
        verbose: bool = True, 
        dev_verbose: bool = True, 
        similarity: float = 0.0, 
        timings: bool = True, 
        save_dir: str = "./saved_stores", 
        top_k: int = 5, 
        llm_verbose: bool = True, 
        top_p: float = 0.8, 
        temp: float = 1.0, 
        top_docs: int = 3,
        model_name: str = "phi3:3.8b"
    ):
        self.directory = directory
        self.verbose = verbose
        self.dev_verbose = dev_verbose
        self.similarity = similarity
        self.timings = timings
        self.save_dir = save_dir
        self.top_k = top_k
        self.llm_verbose = llm_verbose
        self.top_p = top_p
        self.temp = temp
        self.top_docs = top_docs
        self.model_name = model_name
