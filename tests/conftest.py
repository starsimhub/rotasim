"""
Settings for the tests: draw figures off screen, so that the docs' code (e.g. sim.plot(), run by test_docs.py) never opens windows
"""
import matplotlib
matplotlib.use('agg')
