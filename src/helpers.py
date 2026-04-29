from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import shutil

class Helpers:
    def init():
        pass

    @staticmethod
    def print_stats(results, show_nums=False, show_semi_params=False, show_semi_values=False):
        print("\n========== RESULTS SUMMARY ==========")
        print("Total ParsedSimResults:", len(results))

        all_trace_names = sorted({
            name for r in results for name in r.traces.keys()
        })
        all_meas_names = sorted({
            name for r in results for name in r.meas.keys()
        })
        all_semi_sections = sorted({
            section for r in results for section in r.semi_ops.keys()
        })

        for r in sorted(results, key=lambda x: x.file_id):
            print(f"\nSIMULATION FILE #: {r.file_id}")
            #print("  Results fields:", list(vars(r).keys()))
            print("  Traces:", list(r.traces.keys()))
            print("  Measures:", list(r.meas.keys()) if r.meas else "(none)")

            if show_nums:
                print("  # traces:", len(r.traces))
                print("  # meas  :", len(r.meas))

            if r.semi_ops:
                print("  Semiconductor sections:", list(r.semi_ops.keys()))

                if show_nums:
                    print("  # semi sections:", len(r.semi_ops))

                for section, devices in r.semi_ops.items():
                    dev_names = list(devices.keys())
                    print(f"    [{section}] devices:", dev_names if dev_names else "(none)")

                    if show_nums:
                        print(f"      # devices: {len(dev_names)}")

                    if show_semi_params:
                        param_names = sorted({
                            param
                            for dev_data in devices.values()
                            for param in dev_data.keys()
                        })
                        print(f"      params: {param_names if param_names else '(none)'}")

                    if show_semi_values:
                        for dev_name, dev_data in devices.items():
                            print(f"      {dev_name}:")
                            for param, value in dev_data.items():
                                print(f"        {param} = {value}")
            else:
                print("  Semiconductor sections: (none)")

        print("\nAll trace names seen:", all_trace_names if all_trace_names else "(none)")
        print("All measure names seen:", all_meas_names if all_meas_names else "(none)")
        print("All semiconductor sections seen:", all_semi_sections if all_semi_sections else "(none)")

    @staticmethod
    def clean_directory(folder: Path) -> None:
        folder = Path(folder)
        if folder.exists():
            shutil.rmtree(folder)          # deletes folder + all contents
        folder.mkdir(parents=True, exist_ok=True)

    @staticmethod # Plot trace versus trace
    def plot_op_points(results, x_trace, y_trace, scale_y=1.0, title=None):
        xs = []
        ys = []

        for r in results:
            x_raw = r.traces.get(x_trace)
            y_raw = r.traces.get(y_trace)

            if x_raw is None or y_raw is None:
                print(f"Skipping file_id={r.file_id}: missing {x_trace} or {y_trace}")
                continue

            x = float(np.asarray(x_raw).ravel()[0])
            y = float(np.asarray(y_raw).ravel()[0]) * scale_y

            xs.append(x)
            ys.append(y)

        if not xs:
            print("No valid points to plot.")
            return

        plt.plot(xs, ys, marker='o')
        if title:
            plt.title(title)
        plt.xlabel(x_trace)
        plt.ylabel(y_trace)
        plt.grid(True)
        plt.show()
