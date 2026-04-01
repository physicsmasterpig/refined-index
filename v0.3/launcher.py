"""PyInstaller entry point — launches the ManifoldIndex GUI."""
import multiprocessing
multiprocessing.freeze_support()

from manifold_index.app import main
main()
