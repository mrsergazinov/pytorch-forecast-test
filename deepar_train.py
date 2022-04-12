import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pylab as plt

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch

from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

if __name__ == "__main__":
    path = './gludata/data'
    with open(path+"/train_data_pyforecast.pkl", 'rb') as f:
        train_data_raw = pickle.load(f)
    with open(path+"/val_data_pyforecast.pkl", 'rb') as f:
        val_data_raw = pickle.load(f)
    with open(path+"/test_data_pyforecast.pkl", 'rb') as f:
        test_data_raw = pickle.load(f)

    def read_data(data, id_start):
        data_len = sum([len(data[i][1]) for i in range(len(data))])
        data_pd = pd.DataFrame(index = range(data_len),
                            columns = ["timeidx", "id", "subject", "CGM", 
                                        "dayofyear", "dayofmonth", "dayofweek", "hour", 
                                        "minute", "date"])
        start = 0
        for i in range(len(data)):
            block_len = len(data[i][1]) 
            data_pd["timeidx"][start:(start+block_len)] = range(block_len)
            data_pd["id"][start:(start+block_len)] = [id_start + i] * block_len
            data_pd["subject"][start:(start+block_len)] = [data[i][0]] * block_len
            data_pd["CGM"][start:(start+block_len)] = data[i][1].flatten() 
            data_pd["date"][start:(start+block_len)] = data[i][3]
            start += block_len
        
        # set format
        data_pd["id"] = data_pd["id"].astype(str).astype("string").astype("category")
        data_pd["subject"] = data_pd["subject"].astype(str).astype("string").astype("category")
        data_pd["CGM"] = data_pd["CGM"].astype("float")
        data_pd["timeidx"] = data_pd["timeidx"].astype("int")
        
        #extract time features
        data_pd["date"] = pd.to_datetime(data_pd["date"])
        data_pd["dayofyear"] = data_pd["date"].dt.dayofyear.astype("string").astype("category")
        data_pd["dayofmonth"] = data_pd["date"].dt.day.astype("string").astype("category")
        data_pd["dayofweek"] = data_pd["date"].dt.dayofweek.astype("string").astype("category")
        data_pd["hour"] = data_pd["date"].dt.hour.astype("string").astype("category")
        data_pd["minute"] = data_pd["date"].dt.minute.astype("string").astype("category")
        
        # reset index
        data_pd = data_pd.reset_index()
        data_pd = data_pd.drop(columns=["index"])
        return data_pd

    train_data_pd = read_data(train_data_raw, 0)
    val_data_pd = read_data(val_data_raw, len(train_data_raw))
    test_data_pd = read_data(test_data_raw, len(train_data_raw)+len(val_data_raw))

    train_data = TimeSeriesDataSet(
        train_data_pd,
        time_idx="timeidx",
        target="CGM",
        group_ids=["id"],
        max_encoder_length=180,
        max_prediction_length=12,
        static_categoricals=["subject"],
        time_varying_known_categoricals= ["dayofyear", 
                                        "dayofmonth", 
                                        "dayofweek", 
                                        "hour",
                                        "minute"],
        time_varying_known_reals=["timeidx"],
        time_varying_unknown_reals = ["CGM"],
        scalers=[],
        # add_relative_time_idx=True,
        # add_encoder_length=True,
    )
    train_dataloader = train_data.to_dataloader(train=True, batch_size=64, num_workers=24)

    val_data = TimeSeriesDataSet(
        val_data_pd,
        time_idx="timeidx",
        target="CGM",
        group_ids=["id"],
        max_encoder_length=180,
        max_prediction_length=12,
        static_categoricals=["subject"],
        time_varying_known_categoricals= ["dayofyear", 
                                        "dayofmonth", 
                                        "dayofweek", 
                                        "hour",
                                        "minute"],
        time_varying_known_reals=["timeidx"],
        time_varying_unknown_reals = ["CGM"],
        scalers=[],
        # add_relative_time_idx=True,
        # add_encoder_length=True,
    )
    val_dataloader = val_data.to_dataloader(train=False, batch_size=64, num_workers=24)

    test_data = TimeSeriesDataSet(
        test_data_pd,
        time_idx="timeidx",
        target="CGM",
        group_ids=["id"],
        max_encoder_length=180,
        max_prediction_length=12,
        static_categoricals=["subject"],
        time_varying_known_categoricals= ["dayofyear", 
                                        "dayofmonth", 
                                        "dayofweek", 
                                        "hour",
                                        "minute"],
        time_varying_known_reals=["timeidx"],
        time_varying_unknown_reals = ["CGM"],
        scalers=[],
        # add_relative_time_idx=True,
        # add_encoder_length=True,
    )
    test_dataloader = test_data.to_dataloader(train=False, batch_size=64, num_workers=24)

    # configure network and trainer
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath='./saved_models',
        filename="deepar-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min")
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")

    trainer = pl.Trainer(
        max_epochs=100,
        progress_bar_refresh_rate=10,
        log_every_n_steps=1,
        gpus=[0],
        weights_summary="top",
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    deepar = DeepAR.from_dataset(
        train_data,
        cell_type='LSTM',
        hidden_size=60,
        rnn_layers=4,
        dropout=0.1,
        learning_rate=0.001,
    )
    print(f"Number of parameters in network: {deepar.size()/1e3:.1f}k")

    # fit network
    trainer.fit(
        deepar,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )