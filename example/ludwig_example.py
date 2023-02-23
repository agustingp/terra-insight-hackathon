from ludwig.api import LudwigModel
import os
import pandas as pd
import numpy as np


def evaluate_model(ludwig_model, dataset, id):

    exp_results = ludwig_model.evaluate(
        dataset=dataset,
        data_format=None,
        split='full',
        batch_size=128,
        skip_save_unprocessed_output=True,
        skip_save_predictions=False,
        skip_save_eval_stats=False,
        collect_predictions=True,
        collect_overall_stats=True,
        output_directory='results' + str(id),
        debug=False
    )

    return exp_results

def train_dl_model(dataset, exp_name, index, previous_directory = None):

    os.environ['NUMEXPR_NUM_THREADS'] = '12'
    config = "config.yaml"

    ludwig_model = LudwigModel(
        config,
        logging_level=100,
        backend=None,
        gpus=None, # if using GPU change to [0]
        gpu_memory_limit=None,
        allow_parallel_threads=True,
        callbacks=None)


    exp_results = ludwig_model.experiment(
        dataset=dataset,
        training_set_metadata=None,
        data_format=None,
        experiment_name= exp_name,
        model_name='run'+str(index),
        model_load_path= previous_directory , # useful for transfer learning
        model_resume_path=None,
        skip_save_training_description=False,
        skip_save_training_statistics=False,
        skip_save_model=False,
        skip_save_progress=False,
        skip_save_log=False,
        skip_save_processed_input=True,
        skip_save_unprocessed_output=True,
        skip_save_predictions=False,
        skip_save_eval_stats=False,
        skip_collect_predictions=False,
        skip_collect_overall_stats=False,
        output_directory=exp_name,
        random_seed=42
    )

    return exp_results

# See we are only loading NDVI feature

df = pd.read_csv('../sample_train_set.csv', usecols=["ndvi", "label"])

# Train a model defined in config.yaml See more details: https://ludwig.ai/latest/user_guide/api/LudwigModel/

_,_,_, output_directory = train_dl_model(df, "example_1", 0)


ludwig_model = LudwigModel.load("./example_1/example_1_run0/model")
df_evaluate = pd.read_csv('../sample_train_set.csv', usecols=["ndvi", "label"])

evaluate_model(ludwig_model, df_evaluate, '1')

# predictions, _ = ludwig_model.predict(dataset=file_path)

