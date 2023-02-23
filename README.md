# Terra-Insight Hackathon Sample Dataset

## This repository provides sample datasets for the Insight-Terra hackathon to be held on the 25th of February at the Insight Centre for Data Analytics, Galway 

We provide a code snippet to use this datasets within Ludwig declarative machine learning framework. This is just an example of how these datasets can be used to train land use classification models, the participants can use any library or libraries of their preference.

* The day of the hackathon we will provide you with two tabular datasets: train_set.csv and test_set.csv
* train_set.csv will contain a number of columns: “blue”, “green”, “red”, “nir”, “ndvi” and “label”
* Each row corresponds to a pixel in our study region
* Each cell in the spectral bands’ columns contains the 55 datapoints corresponding to each spectral band, separated by a space.
* The column “label” depicts the actual pixel class belonging to one of the folloging 7 categories:

1 - “Forest & Semi-natural Vegetation”
2 - “Buildings - Urban”
3 - “Water”
4 - “Bare soil”
5 - “Corn”
6 - “Soybean”
7 - “Sunflower”

## Evaluation

* test_set.csv will NOT contain the column “label”, instead, it will contain a “GlobalID” column
* In order to evaluate your model's performance, you will be able to submit your results in a website
* We will evaluate your results by using the “GlobalID” and the “label” (predicted by your model)
* During the hackathon time, we will use a subset off this test dataset to provide you feedback on your models, using the website
* We will reserve the entire test set for the final evaluation of the teams
* The metric to be used will be “Weighted F1 Score”


## Next, we provide a code snippet to showcase the use of the datasets within Ludwig


### Creating a conda environment https://docs.conda.io/en/latest/miniconda.html

```console
foo@bar:~$ conda create -n hk python=3.8
foo@bar:~$ conda activate hk
```

### Installing Ludwig https://ludwig.ai/

```console
foo@bar:~$ pip install ludwig
or
foo@bar:~$ pip install 'ludwig[full]'
```

### Installing Ludwig https://ludwig.ai/ and https://ludwig.ai/latest/getting_started/installation/

```console
foo@bar:~$ pip install ludwig
or
foo@bar:~$ pip install 'ludwig[full]'
```

### If using GPU install the following

```console
foo@bar:~$ pip install torch -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

