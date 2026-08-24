"""exp30 entry point. The forward-prediction lives in uk_vaccine_predict_run.py (run on a VM);
this thin wrapper is the canonical `run.py` and also regenerates the figure from saved outputs.

  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HM_WORKERS=110 python run.py        # full VM run
  python run.py --figure-only                                                       # just replot
"""
import sys
import uk_vaccine_predict_run as predict

if __name__ == '__main__':
    if '--figure-only' not in sys.argv:
        predict.main()
    import fig_uk_vaccine_validation   # noqa: F401  (writes figures/uk_vaccine_validation.png on import)
