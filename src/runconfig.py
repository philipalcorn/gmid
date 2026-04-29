from dataclasses import dataclass
from pathlib import Path


@dataclass 
class RunConfig:
    def __init__(self):
        # Path to LTSPICE executable

        self.exe_path = Path(r"C:\Program Files\ADI\LTspice\LTspice.exe")
        self.asc_path = Path(__file__).parent.parent / "simulations" / "example_closed_loop.asc"
        if not self.asc_path.exists():
            raise FileNotFoundError(f"Can't find {self.asc_path.resolve()}")
        # Path to folder for output files 
        self.out_path = Path("./out")
        self.num_parallel_sims = 1
        self.debug_mode=False
        self.make_plots = False

        """ 
        Parameter Selection

        This allows you to select what values you are interested 
        in receiving for the SimResults. Setting a variable to 
        None will automatically grab everything available.
        for example, 
        'self.meas_names=None'
        will pull every .meas value automatically. While convenient, 
        this can make data parsing cluttered so it is reccomended
        to only grab what you are interested in. 
        """
        self.meas_names=None                     
        self.trace_names=["v(vy)", "Id(M7)"]   
        self.device_names=["M6"]
        self.device_values=["Model", "Id", "Vgs", "Vdsat"]
