## Overview

LIMIT-v2 is the second version of a project with the purpose of stress testing embedding models. The first version sits next to this project, i.e. at 
../LIMIT

## Design Philosophy

All functions in the code must fit into ONLY ONE of three categories:

Dataset creation -> Embedding -> Evaluation. 

This way, multiple tests can be done on multiple models. The embedding has to be able to run locally and on a cluster. And different metrics have to be able to evaluate performance without recomputing embeddings every time.


## Project structure

- `explore_phantom_wiki.ipynb` — exploratory notebook for the PhantomWiki dataset (kilian-group/phantom-wiki-v1 on HuggingFace)

**You have to keep this relevant**. Every time you delete files or create new files that are relevant and meant to stay, update this (no throwaway or miscellanious files)

## Coding practices

**IMPORTANT:** Use the correct venv based on the machine name. To get it, run `hostname` in bash:
- `EDGAR-PC`: `C:\Users\edgar\Projekte\Python\Machine_Learning_venv\Scripts\python.exe`
- `EDGAR_LAPTOP`: `C:\Users\EdgarLangwald\OneDrive - neuland AI AG\Coding\.venv\Scripts\python.exe`
- `*.rwth-aachen.de` (RWTH cluster): `/rwthfs/rz/cluster/home/nld68820/.venv/bin/python`

Run all scripts and pip installs using the correct python for the current device.

## Other

EDGAR-PC and EDGAR_LAPTOP run on windows. Keep in mind that spaces in paths compicate the bash path finding.

