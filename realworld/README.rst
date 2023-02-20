======================
Real-world time series
======================

In this, we look to analyze real-world time series through the lense of
chaos and information theory. Specifically, we are looking at the new cases
and hospital patients for COVID-19.

We want to see if these time series exhibit stable, periodic, or chaotic
behavior. We also want to characterize each time series in terms of Shannon
entropy.

We also wish to analyze the mutual information and transfer entropy between
indiviual countries and the total (similar to Walker et al.'s `"Evolutionary
Transitions and Top-Down Causation" <https://arxiv.org/pdf/1207.4808.pdf>`_), as well as the mutual information and
transfer entropy between the two variables.

Replicating our numbers
-----------------------
First, download the `"Our World in Data" COVID-19 dataset <https://covid.ourworldindata.org/data/owid-covid-data.csv>`_.

Next, install the python dependencies::

        $ pip install -r realworld/requirements.txt

Then run::

        $ python -m realworld --file <path-to-dataset>

For example, if you saved the dataset to ``data/owid-covid-data.csv``,
then you would run::

        $ python -m realworld --file data/owid-covid-data.csv
 
The default behavior is to assume the dataset is saved to ``data/owid-covid-data.csv``,
in which case you can simply run::
 
         $ python -m realworld
