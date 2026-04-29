from dataclasses import dataclass, field
from PyLTSpice import RawRead
from PyLTSpice.log.ltsteps import LTSpiceLogReader as LogRead
from typing import Dict, Any, Optional, Iterable, Iterator
import numpy as np

@dataclass
class SimResult:
    file_id: int
    raw_data: Optional[RawRead] = None
    log_data: Optional[LogRead] = None
    semi_ops: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)


@dataclass
class ParsedSimResult:
    file_id: int
    traces: Dict[str, np.ndarray]
    meas: Dict[str, Any]
    semi_ops: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)


class Parser:
    def __init__(self) -> None:
        pass
    @staticmethod
    def filter_semi_ops(semi_ops, semi_names=None, semi_values=None):
        if not semi_ops:
            return {}

        # normalize to sets for fast lookup
        name_set = set(semi_names) if semi_names is not None else None
        val_set = set(semi_values) if semi_values is not None else None

        filtered = {}

        for section, devices in semi_ops.items():
            kept_devices = {}

            for dev_name, params in devices.items():
                # filter device names
                if name_set is not None and dev_name not in name_set:
                    continue

                if val_set is None:
                    kept_params = dict(params)
                else:
                    kept_params = {
                        param_name: value
                        for param_name, value in params.items()
                        if param_name in val_set
                    }

                # only keep device if something remains
                if kept_params:
                    kept_devices[dev_name] = kept_params

            # only keep section if something remains
            if kept_devices:
                filtered[section] = kept_devices

        return filtered


    def _parse_one(
        self,
        result: SimResult,
        trace_names=None,
        meas_names=None,
        semi_names=None,
        semi_values=None
    ) -> ParsedSimResult:
        traces: Dict[str, np.ndarray] = {}
        meas: Dict[str, Any] = {}

        # raw traces
        if getattr(result, "raw_data", None) is not None:
            names = trace_names
            if names is None:
                try:
                    names = list(result.raw_data.get_trace_names())
                except Exception:
                    print(f"Warning: no trace names found for file_id={result.file_id}")
                    names = []

            for name in names:
                try:
                    traces[name] = np.asarray(
                        result.raw_data.get_trace(name).get_wave()
                    )
                except Exception:
                    continue

        # log / measures
        if getattr(result, "log_data", None) is not None:
            if meas_names is None:
                keys = []
                try:
                    keys.extend(result.log_data.get_step_vars() or [])
                except Exception:
                    pass
                try:
                    keys.extend(result.log_data.get_measure_names() or [])
                except Exception:
                    pass

                seen = set()
                keys = [k for k in keys if not (k in seen or seen.add(k))]
            else:
                keys = list(meas_names)

            for k in keys:
                try:
                    vlist = result.log_data[k]
                except Exception:
                    continue

                if isinstance(vlist, (list, tuple)) and len(vlist) > 0:
                    v0 = vlist[0]
                else:
                    v0 = vlist

                try:
                    meas[k] = float(v0)
                except Exception:
                    try:
                        meas[k] = complex(v0)
                    except Exception:
                        meas[k] = v0

        all_semi_ops = getattr(result, "semi_ops", None) or {}
        semi_ops = self.filter_semi_ops(
            all_semi_ops,
            semi_names=semi_names,
            semi_values=semi_values,
        )

        return ParsedSimResult(
            file_id=result.file_id,
            traces=traces,
            meas=meas,
            semi_ops=semi_ops,
        )

    def parse(
        self,
        results: Iterable[SimResult],
        trace_names=None,
        meas_names=None,
        semi_names=None,
        semi_values=None
    ) -> Iterator[ParsedSimResult]:
        for result in results:
            if result is None:
                continue

            if hasattr(result, "success") and not result.success:
                continue

            has_raw = getattr(result, "raw_data", None) is not None
            has_log = getattr(result, "log_data", None) is not None
            has_raw_file = getattr(result, "raw_file", None) is not None
            has_log_file = getattr(result, "log_file", None) is not None

            if not (has_raw or has_log or has_raw_file or has_log_file):
                continue

            yield self._parse_one(
                result,
                trace_names=trace_names,
                meas_names=meas_names,
                semi_names=semi_names,
                semi_values=semi_values,
            )

    
