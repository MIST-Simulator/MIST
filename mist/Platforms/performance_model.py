"""
    PerformanceModel helps estimate the duration of tasks or iterations 
    Adapted from Splitwise
"""

import math
import os

from abc import ABC, abstractmethod

import pandas as pd

from scipy.interpolate import interp1d

from mist.Platforms import PlatformType


performance_model = None


class PerformanceModel(ABC):
    """
    PerformanceModel helps estimate the duration of tasks or iterations,
    under given hardware, model, and parallelism configurations.
    Abstract class that must be subclassed.
    """
    def __init__(self):
        global performance_model
        performance_model = self



class SplitwisePerformanceModel(PerformanceModel):
    """
        PerformanceModel based on a CSV database of characterization runs.
        Interpolates between data points and updates the database correspondingly.
        The underlying predictor could be changed for different interpolation strategies.
    """
    def __init__(self, db_path):
        super().__init__()

        # Data is assumed to be stored in a CSV file of a subfolder
        self.db = pd.read_csv(db_path)

        # ensure the database has the correct columns
        # and remove extraneous columns
        self.db = self.db[["model",
                           "hardware",
                           "tensor_parallel",
                           "prompt_size",
                           "batch_size",
                           "token_size",
                           "prompt_time",
                           "token_time"]]

        # Keep the time in ms
        self.db["prompt_time"] = self.db["prompt_time"]
        self.db["token_time"] = self.db["token_time"]

        self.init_predictor()

    def init_predictor(self):
        """
        Predict using number of tokens in the batch.
        """
        self.prompt_time_predictors = {}
        self.token_time_predictors = {}
        self.prompt_time_cache = {}
        self.token_time_cache = {}

        for model in self.db["model"].unique():
            for hardware in self.db["hardware"].unique():
                for tensor_parallel in self.db["tensor_parallel"].unique():
                    mask = (self.db["model"] == model) & \
                            (self.db["hardware"] == hardware) & \
                            (self.db["tensor_parallel"] == tensor_parallel)
                    db_subset = self.db[mask].copy()
                    if len(db_subset) == 0:
                        continue
                    db_subset["batch_tokens"] = db_subset["prompt_size"] * db_subset["batch_size"]
                    x = db_subset[["batch_tokens", "prompt_time"]].groupby("batch_tokens").median().index
                    y = db_subset[["batch_tokens", "prompt_time"]].groupby("batch_tokens").median()["prompt_time"]
                    self.prompt_time_predictors[(model, hardware, tensor_parallel)] = interp1d(
                                                                    x, y, fill_value="extrapolate")
                    x = db_subset[["batch_tokens", "token_time"]].groupby("batch_tokens").median().index
                    y = db_subset[["batch_tokens", "token_time"]].groupby("batch_tokens").median()["token_time"]
                    self.token_time_predictors[(model, hardware, tensor_parallel)] = interp1d(
                                                                    x, y, fill_value="extrapolate")

    def _match(self, **kwargs):
        """
        Returns a boolean mask for the database from kwargs.
        """
        mask = True
        for k, v in kwargs.items():
            mask &= (self.db[k] == v)
        return mask

    def predict_new_row(self, **kwargs):
        """
        Predicts the prompt and token time for a new row.
        Inserts the new row into the database.
        """
        model = kwargs["model"]
        hardware = kwargs["hardware"]
        tensor_parallel = kwargs["tensor_parallel"]
        batch_tokens = kwargs["batch_tokens"]
        new_row = pd.DataFrame(kwargs, index=[0])

        prompt_time = self.prompt_time_predictors[(model, hardware, tensor_parallel)](batch_tokens)
        token_time = self.token_time_predictors[(model, hardware, tensor_parallel)](batch_tokens)

        new_row["prompt_time"] = prompt_time
        new_row["token_time"] = token_time
        self.db = pd.concat([self.db, new_row], ignore_index=True)
        return new_row

    def get_prompt_time(self, **kwargs):
        """
        Returns the prompt time from the database.
        """
        prompt_time = self.db[self._match(**kwargs)]["prompt_time"].median()
        # if not found, predict
        if math.isnan(prompt_time):
            new_row = self.predict_new_row(**kwargs)
            prompt_time = new_row["prompt_time"][0]
        return prompt_time

    def get_token_time(self, **kwargs):
        """
        Returns the prompt time from the database.
        """
        token_time = self.db[self._match(**kwargs)]["token_time"].median()
        # if not found, predict
        if math.isnan(token_time):
            new_row = self.predict_new_row(**kwargs)
            token_time = new_row["token_time"][0]
        return token_time