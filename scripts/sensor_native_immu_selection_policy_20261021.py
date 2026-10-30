"""Pure deterministic shared-epoch selection policy for native IMMU screen."""
from decimal import Decimal

SEEDS = (17, 23, 29)


def synchronized_epoch_rows(raw_by_seed, ema_by_seed):
    """Return shared raw/EMA three-seed means; incomplete epochs are invalid."""
    if len(raw_by_seed) != len(SEEDS) or len(ema_by_seed) != len(SEEDS):
        raise ValueError('A synchronized epoch requires exactly three seed scores per view')
    raw = tuple(float(value) for value in raw_by_seed)
    ema = tuple(float(value) for value in ema_by_seed)
    if any(not 0 <= value <= 1 for value in raw + ema):
        raise ValueError('Macro-F1 values must be finite scores in [0,1]')
    return float(sum(Decimal(str(value)) for value in raw) / Decimal(3)), \
        float(sum(Decimal(str(value)) for value in ema) / Decimal(3))


def should_stop_group(epoch, stale_epochs, maximum=100, minimum=20, patience=10):
    if not 1 <= epoch <= maximum or stale_epochs < 0:
        raise ValueError('Invalid group epoch state')
    return epoch == maximum or (epoch >= minimum and stale_epochs >= patience)


def choose_recipe(rows):
    """Select maximum Decimal score; exact ties use epoch/raw/lr ordering."""
    if not rows:
        raise ValueError('No candidate recipe scores')
    for row in rows:
        if row['view'] not in ('raw', 'ema') or row['epoch'] < 1 or row['lr_index'] < 0:
            raise ValueError('Invalid recipe score record')
        score = Decimal(str(row['score']))
        if not score.is_finite() or not Decimal(0) <= score <= Decimal(1):
            raise ValueError('Macro-F1 must be a finite score in [0,1]')
    return min(rows, key=lambda row: (-Decimal(str(row['score'])), row['epoch'],
                                     0 if row['view'] == 'raw' else 1, row['lr_index']))
