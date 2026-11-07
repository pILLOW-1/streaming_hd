"""Load the original supervised probe from the pinned ARS dependency."""
from common.upstream import definitions
NonLinearClassifier, = definitions('ars', 'detectors/train_probing.py', ('NonLinearClassifier',))
