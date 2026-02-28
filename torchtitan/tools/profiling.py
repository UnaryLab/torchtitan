# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import pickle
import time

import torch

from torchtitan.config import Profiling as ProfilingConfig
from torchtitan.tools.logging import logger
from torchtitan.tools.straggler_detection import get_straggler_gpus

import numpy as np
import math

# Do this if you have admin privileges:
# from amdsmi import amdsmi_get_processor_handles, amdsmi_get_gpu_kfd_info, amdsmi_init, amdsmi_shut_down, amdsmi_set_power_cap

# Do this if you don't have admin privileges:
import grpc
import grpc.experimental

# from importlib.resources import files

# protos, services = grpc.protos_and_services(
#     str(files("torchtitan.tools").joinpath("power.proto")))
protos, services = grpc.protos_and_services("power.proto")

# how much memory allocation/free ops to record in memory snapshots
MEMORY_SNAPSHOT_MAX_ENTRIES = 100000

pending_counter = 0
wait_counter = 0
max_lead = 0

# Do this if you have admin privileges:
# def set_pow(gpu_num: int, val: int):
#     amdsmi_init()

#     devices = amdsmi_get_processor_handles()
#     gpu_ids = {}
#     for device in devices:
#         gpu_ids[amdsmi_get_gpu_kfd_info(device)['node_id']-2] = device

#     device = gpu_ids[gpu_num]
#     logger.info(f"Setting GPU{gpu_num} power to {val/1000000:.3f} W")
#     amdsmi_set_power_cap(device, 0, int(val))

#     amdsmi_shut_down()


def set_pow(gpu_num: int, val: int, grpc_socket: str):
    response = services.PowerServer.SetPower(
        protos.PowerReq(gpu=gpu_num, watts=val),
        f'unix://{grpc_socket}',
        insecure=True,
    )
    if response.ack == 0:
        logger.info("Successfully set power")
    elif response.ack == 1:
        logger.error("Failed to set power")
    else:
        raise ValueError(f"Unexpected ack: {response.ack}")


@contextlib.contextmanager
def maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
    # get user defined profiler settings
    enable_profiling = profiling_config.enable_profiling

    if enable_profiling:
        trace_dir = os.path.join(base_folder, profiling_config.save_traces_folder)
        profile_freq, warmup, active = (
            profiling_config.profile_freq,
            profiling_config.profiler_warmup,
            profiling_config.profiler_active,
        )

        gpu_power = [profiling_config.initial_power_cap for _ in range(8)]
        gpu_pending = [[] for _ in range(8)]

        rank = torch.distributed.get_rank()

        if rank == 0 and profiling_config.power_man:
            for gpu_num in range(8):
                logger.info(f"Setting initial power cap for GPU{gpu_num}: {gpu_power[gpu_num]:.3f}")
                set_pow(gpu_num, gpu_power[gpu_num], profiling_config.grpc_socket)

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_num)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name, leaf_folder)
            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)

            logger.info(f"Dumping profiler traces at step {prof.step_num}")
            begin = time.monotonic()

            output_file = os.path.join(curr_trace_dir, f"rank{rank}_trace.json")
            prof.export_chrome_trace(output_file)
            logger.info(
                f"Finished dumping profiler traces in {time.monotonic() - begin:.2f} seconds"
            )

            global pending_counter
            global wait_counter
            global max_lead
            if profiling_config.power_man:
                torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    logger.info("Tweaking frequency...")
                    # WARN hardcoded for 8 GPUs
                    gpu_traces = [os.path.join(curr_trace_dir, f"rank{rank_}_trace.json") for rank_ in range(8)]
                    for gpu_trace in gpu_traces:
                        assert os.path.exists(gpu_trace)

                    straggler_gpus, max_lead = get_straggler_gpus(
                        gpu_traces,
                        profiling_config.max_adj,
                        invert=True,
                        max_lead=max_lead if profiling_config.use_global else 0,
                        use_sum=profiling_config.use_sum,
                        use_max=profiling_config.use_max,
                        use_last=profiling_config.use_last,
                    )

                    for gpu_num, delta in straggler_gpus.items():
                        logger.info(f"Pending delta GPU{gpu_num}: {delta:.3f} W")
                    if wait_counter < profiling_config.wait_steps:
                        logger.info(f"Waiting steps {profiling_config.wait_steps - wait_counter} left...")
                        wait_counter += 1
                    elif pending_counter == profiling_config.adjust_steps - 1:
                        # Adjust power distribution
                        pending_counter = 0
                        avg_deltas = {}
                        for gpu_num, delta in straggler_gpus.items():
                            gpu_pending[gpu_num].append(delta)
                            avg_delta = np.median(gpu_pending[gpu_num]).astype(int)
                            gpu_pending[gpu_num] = []
                            avg_deltas[gpu_num] = avg_delta
                        for gpu_num, avg_delta in avg_deltas.items():
                            gpu_power[gpu_num] += avg_delta
                        total_power = sum(gpu_power)
                        power_delta = math.ceil((total_power - profiling_config.fake_max_power * 8)/8)
                        logger.info(f"Total Power: {(total_power - power_delta*8):.3f} W")
                        assert total_power-power_delta*8 <= profiling_config.fake_max_power * 8

                        # Uniformly raise or lower power distribution
                        gpu_delta = 0
                        for gpu_num in avg_deltas.keys():
                            gpu_power[gpu_num] -= power_delta
                            gpu_delta = max(gpu_delta, gpu_power[gpu_num] - profiling_config.max_power)
                        # Uniformly lower if any GPUs are above TDP
                        for gpu_num in avg_deltas.keys():
                            gpu_power[gpu_num] -= gpu_delta

                        underutil = profiling_config.fake_max_power * 8 - sum(gpu_power)
                        assert underutil >= 0, f"{-1 * underutil:.3f} W over the limit"
                        if underutil > 0:
                            logger.warning(f"Operating {underutil:.3f} W lower than node cap")
                        logger.info("Final power deltas:")
                        for gpu_num, avg_delta in avg_deltas.items():
                            logger.info(f"  GPU{gpu_num}: {avg_delta:.3f} W")
                            new_cap = gpu_power[gpu_num]
                            assert new_cap <= profiling_config.max_power
                            set_pow(gpu_num, new_cap, profiling_config.grpc_socket)
                    else:
                        pending_counter += 1
                        for gpu_num, delta in straggler_gpus.items():
                            gpu_pending[gpu_num].append(delta)
                torch.distributed.barrier()

        logger.info(f"Profiling active. Traces will be saved at {trace_dir}")

        if not os.path.exists(trace_dir):
            os.makedirs(trace_dir, exist_ok=True)

        wait = profile_freq - (active + warmup)
        assert (
            wait >= 0
        ), "profile_freq must be greater than or equal to warmup + active"
        gpu_device_profiled = None
        if torch.cuda.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.CUDA
        elif torch.xpu.is_available():
            gpu_device_profiled = torch.profiler.ProfilerActivity.XPU
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                gpu_device_profiled,
            ],
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active),
            on_trace_ready=trace_handler,
            record_shapes=True,
        ) as torch_profiler:
            torch_profiler.step_num = global_step
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield None


@contextlib.contextmanager
def maybe_enable_memory_snapshot(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
    enable_snapshot = profiling_config.enable_memory_snapshot
    if enable_snapshot:
        snapshot_dir = os.path.join(
            base_folder, profiling_config.save_memory_snapshot_folder
        )
        if not os.path.exists(snapshot_dir):
            os.makedirs(snapshot_dir, exist_ok=True)
        rank = torch.distributed.get_rank()

        class MemoryProfiler:
            def __init__(self, step_num: int, freq: int):
                torch.cuda.memory._record_memory_history(
                    max_entries=MEMORY_SNAPSHOT_MAX_ENTRIES
                )
                # when resume training, we start from the last step
                self.step_num = step_num
                self.freq = freq

            def step(self, exit_ctx: bool = False):
                self.step_num += 1
                if not exit_ctx and self.step_num % self.freq != 0:
                    return
                if not exit_ctx:
                    curr_step = self.step_num
                    dir_name = f"iteration_{curr_step}"
                else:
                    # dump as iteration_0_exit if OOM at iter 1
                    curr_step = self.step_num - 1
                    dir_name = f"iteration_{curr_step}_exit"
                curr_snapshot_dir = os.path.join(snapshot_dir, dir_name, leaf_folder)
                if not os.path.exists(curr_snapshot_dir):
                    os.makedirs(curr_snapshot_dir, exist_ok=True)
                logger.info(f"Dumping memory snapshot at step {curr_step}")
                begin = time.monotonic()
                output_file = os.path.join(
                    curr_snapshot_dir, f"rank{rank}_memory_snapshot.pickle"
                )
                with open(output_file, "wb") as output:
                    pickle.dump(torch.cuda.memory._snapshot(), output)
                logger.info(
                    f"Finished dumping memory snapshot in {time.monotonic() - begin:.2f} seconds"
                )

        logger.info(f"Memory profiler active. Snapshot will be saved at {snapshot_dir}")
        profiler = MemoryProfiler(global_step, profiling_config.profile_freq)
        try:
            yield profiler
        except torch.OutOfMemoryError as e:
            profiler.step(exit_ctx=True)
    else:
        yield None
