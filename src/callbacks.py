from parser import SimResult, ParsedSimResult
from PyLTSpice import RawRead
from PyLTSpice.log.ltsteps import LTSpiceLogReader as LogRead
from PyLTSpice.log.semi_dev_op_reader import opLogReader
import numpy as np
from PyLTSpice.sim.process_callback import ProcessCallback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


def remove_step_lines(log_path: Path):
    lines = log_path.read_text().splitlines()

    filtered = [line for line in lines if not line.strip().startswith(".step")]

    log_path.write_text("\n".join(filtered) + "\n")


# returns the raw file and the log file.
class CallbackGetAllData(ProcessCallback):
    @staticmethod
    def callback(raw_file, log_file):
        #print("CALLBACK STARTS HERE #############################")
        file_id = int(str(raw_file).split("_")[-1].split(".")[0]) 
        #TODO: Might need to get this changed that we can only remove step
        #lines if desired
        remove_step_lines(Path(log_file))
        try:
            raw_data = RawRead(raw_file)
        except Exception:
            print(f"Could not read raw file {raw_file}")
        try:
            log_data = LogRead(log_file)
        except Exception:
            print(f"Could not read log file {log_file}")
        try:
            semi_ops = opLogReader(str(log_file))
        except Exception:
            print(f"Could not get semiconductor ops from {log_file}")
        return SimResult(
            file_id=file_id,
            raw_data=raw_data,
            log_data=log_data,
            semi_ops=semi_ops
        )


