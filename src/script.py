from callbacks import CallbackGetAllData
from parser import Parser
from PyLTSpice import SpiceEditor
from spice import Spice
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from helpers import Helpers
from runconfig import RunConfig


def main() -> None:
    config=RunConfig()
    Helpers.clean_directory(config.out_path)
    spice = Spice(config.exe_path, 
                  asc_path=config.asc_path, 
                  output_folder=config.out_path, 
                  callback_proc=CallbackGetAllData,
                  parallel_sims=config.num_parallel_sims)
    if config.debug_mode: print("Spice initialized (netlist created).")


    # --- small sweep ---
    #values = ["0.3u"]
    values = ["0.3u", "0.5u", "1.0u"]

    for v in values:
        try:
            print(f"Changing .param {{l}} to {v}")
            spice.set_parameter("l", v)
            spice.simulate()
        except KeyboardInterrupt:
            spice.sim_runner.kill_all_ltspice()
            raise

    spice.sim_runner.wait_completion()
    if config.debug_mode: print("Sims complete")

    raw_results = list(spice.sim_runner) 
    if not raw_results:
        print("no results yielded from sim runner")
        return
    if config.debug_mode: print("Raw Results collected")    



    parser = Parser()
    parsed_results = list(parser.parse(
        raw_results,
        config.trace_names, 
        config.meas_names,
        config.device_names,
        config.device_values))

    if not parsed_results:
        print("no results yielded from parsing")
        return
    if config.debug_mode: print("Parsed Results collected")    


    Helpers.print_stats(parsed_results, show_nums=False, show_semi_params=True, show_semi_values=True)

    if not config.make_plots:
        return
    Helpers.plot_op_points(
        parsed_results,
        "V(vx)",
        "Id(M7)",
        title="nmos id vs vds"
        )
   
    return


if __name__ == '__main__':
    main()
