
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from datetime import datetime
from src.dexter.tools.yf_shared import (
    frame_to_records,
    to_python,
    format_period_label,
    apply_period_filters,
    safe_get,
    limit_records,
)

def test_frame_to_records_empty():
    assert frame_to_records(None) == []
    assert frame_to_records(pd.DataFrame()) == []

def test_frame_to_records():
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    records = frame_to_records(df)
    assert len(records) == 2
    assert records[0]['period'] == 'col1'
    assert records[0]['values'] == {'0': 1, '1': 2}

def test_frame_to_records_with_limit():
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    records = frame_to_records(df, limit=1)
    assert len(records) == 1
    assert records[0]['period'] == 'col1'

def test_to_python():
    assert to_python(None) is None
    assert to_python(np.int64(5)) == 5
    assert to_python(pd.Timestamp('2023-01-01')) == '2023-01-01T00:00:00'
    assert to_python(np.nan) is None

def test_format_period_label():
    assert format_period_label(pd.Timestamp('2023-01-01')) == '2023-01-01'
    assert format_period_label(datetime(2023, 1, 1)) == '2023-01-01'
    assert format_period_label('2023-01') == '2023-01'

def test_apply_period_filters():
    records = [
        {'period': '2023-01-01'},
        {'period': '2023-02-01'},
        {'period': '2023-03-01'},
    ]
    assert len(apply_period_filters(records, report_period_gt='2023-01-15')) == 2
    assert len(apply_period_filters(records, report_period_gte='2023-02-01')) == 2
    assert len(apply_period_filters(records, report_period_lt='2023-02-15')) == 2
    assert len(apply_period_filters(records, report_period_lte='2023-02-01')) == 2
    assert len(apply_period_filters(records, report_period_gt='2023-01-01', report_period_lt='2023-03-01')) == 1

def test_safe_get():
    data = {'col1': [10, 20]}
    df = pd.DataFrame(data, index=['row1', 'row2'])
    assert safe_get(df, ['row1'], 'col1') == 10
    assert safe_get(df, ['non_existent', 'row2'], 'col1') == 20
    assert safe_get(df, ['non_existent'], 'col1') is None
    assert safe_get(df, ['row1'], 'non_existent_col') is None
    assert safe_get(None, ['row1'], 'col1') is None

def test_limit_records():
    records = [1, 2, 3, 4, 5]
    assert limit_records(records, 3) == [1, 2, 3]
    assert limit_records(records, 0) == records
    assert limit_records(records, -1) == records
    assert limit_records(records, 10) == records
    assert limit_records(records, None) == records
