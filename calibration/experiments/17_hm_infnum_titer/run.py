"""Exp 17 reproduction — HM posterior (NROY) for infnum+titer under the cohort observation.
Driver is the shared calibration/hm_calibrate.py (--model infnum). Run on a 120-core box in
the pinned env (calibration/hm_env_pins.txt). See SUMMARY.md for results.

Waves 1-6 (AUTO feature selection: mean_sq_z, 1/wave, cooldown 2):
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HM_WORKERS=118 \
    python ../../hm_calibrate.py --model infnum --max-iter 6 --n-samples 1500
Waves 7-9 (FORCE the auto-skipped targets; resume from the wave-6 checkpoint):
  ... --model infnum --max-iter 9 --n-samples 1500 \
      --features repeat_detected_frac,first_inf_median --resume
"""
import subprocess, sys, pathlib
CAL = pathlib.Path(__file__).resolve().parents[2]   # .../calibration
if __name__ == "__main__":
    base = [sys.executable, str(CAL / "hm_calibrate.py"), "--model", "infnum", "--n-samples", "1500"]
    subprocess.run(base + ["--max-iter", "6"], cwd=CAL, check=True)
    subprocess.run(base + ["--max-iter", "9", "--features",
                           "repeat_detected_frac,first_inf_median", "--resume"], cwd=CAL, check=True)
