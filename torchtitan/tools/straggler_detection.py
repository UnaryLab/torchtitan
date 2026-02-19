import numpy as np
import pandas as pd
import json
from decimal import Decimal
from os import path

from torchtitan.tools.logging import logger

def dtoi(ts):
    return np.int64(ts * Decimal('1000'))


def kernel_link(line, data):
    data.append({
        'name': line['name'],
        'ts': dtoi(line['ts']),
        'dur': dtoi(line['dur']),
        'stream': line['args']['stream'],
    })


def json_to_pandas(json_fn: str) -> pd.DataFrame:
    kernel_data = []

    with open(json_fn, 'r') as fp:
        for line in json.load(fp, parse_float=Decimal)['traceEvents']:
            if 'cat' in line and line['cat'] == 'kernel':
                kernel_link(
                    line,
                    kernel_data,
                )
            else:
                continue

    return pd.DataFrame(kernel_data)


def merge_traces(pytorch_traces):
    gpu_map = {i: trace_fn for i, trace_fn in enumerate(pytorch_traces)}
    for gpu, trace_fn in gpu_map.items():
        logger.info(f'Mapping {path.basename(trace_fn)} to GPU{gpu}')
    return pd.concat(
        (
            df.assign(gpu=i)
            for i, df in
            enumerate(map(json_to_pandas, pytorch_traces))
        ),
        ignore_index=True
    )


def leader_value(df, agg=True, use_last=False, use_max=False, use_sum=False):
    df_compute = df[~df['name'].str.startswith('nccl')].copy()
    df_compute['_ki'] = (
        df_compute
        .groupby(['gpu', 'name'])
        .cumcount()
    )
    straggler = (
        df_compute
        .groupby(['name', '_ki'])
        ['ts']
        .max()
        .rename('straggler_ts')
    )
    df = (
        df_compute
        .merge(
            straggler,
            on=['name', '_ki'],
            how='left'
        )
    )
    df['lead'] = df['straggler_ts'] - df['ts']
    df.drop(columns='straggler_ts', inplace=True)
    df.sort_values(['gpu', 'ts'], inplace=True)
    if agg:
        if use_sum:
            return (
                df
                .groupby('gpu')['lead']
                .sum()
                .reset_index()
            )
        elif use_max:
            return (
                df
                .groupby('gpu')['lead']
                .max()
                .reset_index()
            )
        elif use_last:
            return (
                df
                .groupby('gpu')['lead']
                .last()
                .reset_index()
            )
    else:
        return df[['gpu', 'lead']]


def no_overlap(df):
    df = df.copy()
    df['end'] = df['ts'] + df['dur']

    df_compute = df[df['stream'] == 0].copy()
    df_compute['_ki'] = (
        df_compute
        .groupby(['gpu', 'name'])
        .cumcount()
    )

    df_overlap = df[df['stream'] != 0].copy()

    no_overlap = pd.Series(True, index=df_compute.index)
    for _, row_overlap in df_overlap.iterrows():
        overlap_condition = (df_compute['ts'] < row_overlap['end']) & (
            df_compute['end'] > row_overlap['ts'])
        no_overlap &= ~overlap_condition
    df_compute = df_compute[no_overlap].drop(columns=['end'])

    gpu_count = (
        df_compute
        .groupby(['name', '_ki'])['gpu']
        .transform('count')
    )

    n_gpus = df['gpu'].nunique()
    return df_compute[gpu_count == n_gpus].copy()


def get_straggler_gpus(
    pytorch_traces,
    max_adj,
    invert=True,
    max_lead=0,
    use_sum=False,
    use_max=False,
    use_last=False,
):
    assert (use_max and not (use_last or use_sum)) or (use_last and not (
        use_max or use_sum)) or (use_sum and not (use_max or use_last)), "pick one to use"
    df = merge_traces(pytorch_traces)
    lv = leader_value(df, agg=True, use_last=use_last,
                      use_max=use_max, use_sum=use_sum)
    assert lv is not None

    gpu_leads = lv.groupby('gpu')['lead'].sum().reset_index()
    max_lead = max(max_lead, gpu_leads['lead'].max())
    value = ((gpu_leads['lead'] - gpu_leads['lead'].min()) /
             (gpu_leads['lead'].max() - gpu_leads['lead'].min()))

    if invert:
        value = 1 - value

    lv['freq_inc'] = (
        value
        * gpu_leads['lead'].max()/max_lead
        * max_adj
    )

    return lv.set_index('gpu')['freq_inc'].to_dict(), max_lead
