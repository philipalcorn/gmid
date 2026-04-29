from callbacks import CallbackGetAllData
from callbacks import SimResult, ParsedSimResult
from pathlib import Path
import matplotlib.pyplot as plt

from typing import Dict, Iterable, Iterator, List, Optional, Union, Any

import numpy as np
import os
from PyLTSpice import SimRunner, SpiceEditor, RawRead
from PyLTSpice.log.ltsteps import LTSpiceLogReader as LogRead
from PyLTSpice.sim.process_callback import ProcessCallback

class Spice:
    def __init__(
            self,
            exe_path: Path,
            asc_path: Union[str, Path],  
            output_folder: Union[str, Path],
            callback_proc, 
            parallel_sims: int = 8 ) -> None:

        self.asc_path = Path(asc_path)
        self.exe_path = Path(exe_path)
        #Automatically get .net paths
        self.net_path = self.asc_path.with_suffix(".net")

        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.sim_runner = SimRunner(
            output_folder=str(self.output_folder),
            simulator=str(exe_path),
            parallel_sims=parallel_sims)

        # It will create a netlist from a .asc and then assign itself that 
        # netlist
        self.sim_runner.create_netlist(str(self.asc_path))
        self.netlist = SpiceEditor(str(self.net_path))

        self.callback = callback_proc 


    def set_parameter(self, parameter:str, 
                      value:Union[int, float, str]) -> None:
        self.netlist.set_parameters(**{parameter:value})

    def set_component_value(self, components:Union[str, Iterable[str]], 
                            value:Union[int, float, str]) -> None:
        if(isinstance(components, str)):
           components = [components]
        for c in components:
            self.netlist.set_component_value(c,value)

    def simulate(self) -> None:
        self.sim_runner.run(self.netlist, callback=self.callback)



    def run_sweep(self, param: str, values: List[str]) -> None:
        for v in values:
            self.set_parameter(param, v)
            self.simulate()

