"""Load ARS shaping definitions from the user's pinned local dependency."""
from common.upstream import definitions
LinearMap, icm_sameid_ars_loss = definitions('ars', 'train_ars/train_ARS.py', ('LinearMap', 'icm_sameid_ars_loss'))
