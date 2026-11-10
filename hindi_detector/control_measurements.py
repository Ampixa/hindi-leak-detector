"""Retain every attempted post-edit outcome, including explicit abstentions."""
from __future__ import annotations

import logging


def attempt(record, measurement):
    try:
        result = measurement()
    except ValueError as error:
        logging.warning('Retained post-edit abstention %s/%s: %s', record['pair_id'], record['family'], error)
        return {**record, 'status': 'abstain', 'features': None, 'reason': str(error)}
    return {**record, 'status': 'measured', **result}
