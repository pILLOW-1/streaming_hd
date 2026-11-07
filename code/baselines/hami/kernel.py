"""Load the original HaMI definitions from the user's pinned local dependency."""
from common.upstream import definitions
HaMI, MIL_loss = definitions('hami', 'hami.py', ('HaMI', 'MIL_loss'))
