#!/bin/bash
conda create -n dmlab0713 python=3.6
conda activate dmlab0713
python3 -m ipykernel install --user --name dmlab0713 --display-name "dmlab0713"
pip install --upgrade pip
pip install -r requirements.txt

