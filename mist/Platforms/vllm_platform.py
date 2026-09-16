import os
import csv
import itertools
from typing import TYPE_CHECKING, ClassVar, Dict, Iterable, List, Optional
import pandas as pd
import ast
from sklearn.ensemble import RandomForestRegressor
import numpy as np

from mist.Request import Request
from .platforms import PlatformConfig, PlatformType


class vLLMPlatformConfig(PlatformConfig):
    """Configuration for vLLM Platform.

    Args:
    """

    def __init__(
        self,
        vllm_df_path: str = None,
        device: str = "H100_GPU",
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        sys_eff: Optional[float] = 0.75,
        model: Optional[str] = None,
        engine_type: Optional[PlatformType] = PlatformType.MIXED_MACHINE,
        chunk_size: Optional[int] = 2048,
    ) -> None:

        def _parse_list(val):
            if pd.isna(val):
                return []
            if isinstance(val, (list, tuple)):
                return list(val)
            if isinstance(val, str):
                s = val.strip()
                if s == "" or s == "[]":
                    return []
                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    try:
                        return [int(s)]
                    except Exception:
                        return []
                if isinstance(parsed, (list, tuple)):
                    return list(parsed)
                return [parsed]
            return [val]

        def compute_metrics(row):
            """
            Parse Prefill and Context and compute:
            - tokens (total token count)
            - kv_length (sum of KV lengths)
            - attn_size (approx attention work = sum((kv+tokens)*tokens) for prefill and (context+1)*1 for context)
            Returns (tokens, kv_length, attn_size)
            """
            prefill = _parse_list(row.get("Prefill", []))
            context = _parse_list(row.get("Context", []))

            tokens = 0
            kv_length = 0
            attn_size = 0
            batches = 0

            for item in prefill:
                # expected item like (kv_len, tokens)
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    try:
                        kv = int(item[0])
                    except Exception:
                        kv = 0
                    try:
                        tok = int(item[1])
                    except Exception:
                        tok = 0
                else:
                    # fallback: treat single numeric as tokens
                    try:
                        tok = int(item)
                    except Exception:
                        tok = 0
                    kv = 0

                tokens += tok
                kv_length += kv
                attn_size += (kv + tok) * tok
                batches += 1

            for item in context:
                # context entries appear to be integers (token lengths)
                try:
                    c = int(item)
                except Exception:
                    # if it's not directly int, attempt to extract from sequence
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        try:
                            c = int(item[0])
                        except Exception:
                            c = 0
                    else:
                        c = 0

                batches += 1
                tokens += 1  # original logic counted 1 token per context item
                kv_length += c
                attn_size += (
                    c + 1
                ) * 1  # assuming context items have length 1 for attention multiplier

            return batches, tokens, kv_length, attn_size

        def add_columns_to_df(df):
            df[
                [
                    "total_batches",
                    "total_tokens",
                    "total_KV_length",
                    "total_attention_size",
                ]
            ] = df.apply(compute_metrics, axis=1).to_list()
            return df

        self.use_vllm = True
        if vllm_df_path is None:
            base_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "./vllm_runtime_data/"
            )
            if device in ["H100", "A100", "L40S"] and model is not None:
                model_str = model if isinstance(model, str) else getattr(model, "model", None)
                if isinstance(model_str, str):
                    csv_name = f"{model_str.split('/')[-1].replace('_', '-')}_NVIDIA_{device}_{tensor_parallel_size}.csv"
                    vllm_df_path = os.path.join(base_dir, csv_name)
            if vllm_df_path is None or not os.path.exists(vllm_df_path):
                print(f"vLLM data file not found at {vllm_df_path}")
                print(f"Following data exists:")
                if os.path.exists(base_dir):
                    for files in os.listdir(base_dir):
                        print(files)
                print("Falling back to PlatformConfig.")
                self.use_vllm = False
        else:
            if vllm_df_path is None or not os.path.exists(vllm_df_path):
                print(f"vLLM data file not found at {vllm_df_path}")
                print("Falling back to PlatformConfig.")
                self.use_vllm = False

        super().__init__(
            device=device,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            sys_eff=sys_eff,
            engine_type=engine_type,
            chunk_size=chunk_size,
            model=model,
        )

        if self.use_vllm:
            df = pd.read_csv(vllm_df_path, delimiter=";")
            df = df[df["Time (ms)"] > 2]  # Filter out invalid entries
            df = add_columns_to_df(df)

            self.prefill_df = df[(df["Context"] == "[]") & (df["Prefill"] != "[]")]
            self.chunked_df = df[(df["Context"] != "[]") & (df["Prefill"] != "[]")]
            self.decode_df = df[(df["Context"] != "[]") & (df["Prefill"] == "[]")]

            self.prefill_predictor = RandomForestRegressor(
                n_estimators=100, random_state=42
            )
            self.decode_predictor = RandomForestRegressor(
                n_estimators=100, random_state=42
            )
            self.chunked_predictor = RandomForestRegressor(
                n_estimators=100, random_state=42
            )

            self.prefill_predictor.fit(
                self.prefill_df[
                    [
                        "total_batches",
                        "total_tokens",
                        "total_KV_length",
                        "total_attention_size",
                    ]
                ],
                self.prefill_df["Time (ms)"],
            )
            self.decode_predictor.fit(
                self.decode_df[
                    [
                        "total_batches",
                        "total_tokens",
                        "total_KV_length",
                        "total_attention_size",
                    ]
                ],
                self.decode_df["Time (ms)"],
            )
            self.chunked_predictor.fit(
                self.chunked_df[
                    [
                        "total_batches",
                        "total_tokens",
                        "total_KV_length",
                        "total_attention_size",
                    ]
                ],
                self.chunked_df["Time (ms)"],
            )
            print(
                "vLLM PlatformConfig: Trained RandomForestRegressor models for prefill, decode, and chunked time prediction."
            )

    def get_log_prefix(self) -> str:
        if getattr(self, "use_vllm", True):
            return "vllm_"
        return ""

    def _search_vllm_df(
        self,
        decode_only,
        prefill_only,
        total_tokens,
        total_batches,
        total_KV_length,
        total_attention_size,
    ):
        target_df = None
        if decode_only:
            target_df = self.decode_df
        elif prefill_only:
            target_df = self.prefill_df
        else:
            target_df = self.chunked_df

        similar_rows = target_df[
            (target_df["total_tokens"] == total_tokens)
            & (target_df["total_batches"] == total_batches)
        ]
        if not similar_rows.empty:
            matching_latencies = []
            for idx, row in similar_rows.iterrows():
                if (
                    abs(row["total_KV_length"] - total_KV_length)
                    <= 0.025 * total_KV_length
                    and abs(row["total_attention_size"] - total_attention_size)
                    <= 0.025 * total_attention_size
                ):
                    matching_latencies.append(row["Time (ms)"])
                    if len(matching_latencies) >= 5:
                        break
            if matching_latencies:
                import numpy as np

                return np.mean(matching_latencies)
        return 0

    def get_chunked_time(
        self, prefills: List[Request], decodes: List[Request]
    ) -> float:
        """
        API call to the speculative cost model for getting the chunked prefill/decode time.

        Inputs:
        prefills : List of all the prefill requests in the current cycle.
        decodes  : List of all the decode requests in the current cycle.

        Outputs:
        time: Time of complete the batch with chunked prefill and decode.
        """
        if not getattr(self, "use_vllm", True):
            return super().get_chunked_time(prefills, decodes)

        if (
            self.engine_type == PlatformType.PREFILL_MACHINE
            or self.engine_type == PlatformType.DECODE_MACHINE
        ):
            raise ValueError("Machine should be MIXED to process chunked requests.")
        total_tokens = 0
        total_batches = 0
        total_KV_length = 0
        total_attention_size = 0
        decode_only = len(prefills) == 0 and len(decodes) > 0
        prefill_only = len(decodes) == 0 and len(prefills) > 0
        for d in decodes:
            total_KV_length += d.get_current_kv_length()
            total_attention_size += d.get_current_kv_length() + 1
            total_tokens += 1
            total_batches += 1

        for p in prefills:
            total_KV_length += p.get_current_kv_length()
            total_attention_size += (
                p.get_current_kv_length() + p.get_current_chunk_size()
            ) * p.get_current_chunk_size()
            total_tokens += p.get_current_chunk_size()
            total_batches += 1

        latency = 0

        if decode_only:
            avg_latency, avg_energy = self.search_log_db(
                total_tokens, total_batches, total_KV_length, tolerance=0.025
            )
            if avg_latency > 0:
                return avg_latency, avg_energy

            latency = self._search_vllm_df(
                decode_only,
                prefill_only,
                total_tokens,
                total_batches,
                total_KV_length,
                total_attention_size,
            )
            if latency == 0:
                latency = self.decode_predictor.predict(
                    pd.DataFrame(
                        [
                            [
                                total_batches,
                                total_tokens,
                                total_KV_length,
                                total_attention_size,
                            ]
                        ],
                        columns=[
                            "total_batches",
                            "total_tokens",
                            "total_KV_length",
                            "total_attention_size",
                        ],
                    )
                )[0]
            self.log_chunked_output(
                True,
                False,
                total_tokens,
                total_batches,
                total_KV_length,
                0,
                0,
                total_KV_length,
                latency,
                0,
            )

        elif prefill_only:
            latency = self._search_vllm_df(
                decode_only,
                prefill_only,
                total_tokens,
                total_batches,
                total_KV_length,
                total_attention_size,
            )
            if latency == 0:
                latency = self.prefill_predictor.predict(
                    pd.DataFrame(
                        [
                            [
                                total_batches,
                                total_tokens,
                                total_KV_length,
                                total_attention_size,
                            ]
                        ],
                        columns=[
                            "total_batches",
                            "total_tokens",
                            "total_KV_length",
                            "total_attention_size",
                        ],
                    )
                )[0]

        else:
            latency = self._search_vllm_df(
                decode_only,
                prefill_only,
                total_tokens,
                total_batches,
                total_KV_length,
                total_attention_size,
            )
            if latency == 0:
                latency = self.chunked_predictor.predict(
                    pd.DataFrame(
                        [
                            [
                                total_batches,
                                total_tokens,
                                total_KV_length,
                                total_attention_size,
                            ]
                        ],
                        columns=[
                            "total_batches",
                            "total_tokens",
                            "total_KV_length",
                            "total_attention_size",
                        ],
                    )
                )[0]

        return latency, 0.0
