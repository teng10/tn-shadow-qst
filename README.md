# tn-shadow-qst
Code and data for **Learning topological states from randomized measurements using
variational tensor network tomography** (arXiv link: )

- A tutorial of how to use the code base is in [add link]().
- For reproducing figures in the paper with dataset linked [ADD LINK]() , see [`Draft_figures.ipynb`](https://github.com/teng10/tn-shadow-qst/blob/bd3f62930849889fba854b96f6da129fc1c99e51/Draft_figures.ipynb).


## Commands to run the code

### Set up the environment with python version control.


   1. Git clone the current repository
```
git clone https://github.com/teng10/tn-shadow-qst.git
```
   2. Create virtual environment 
```
python3 -m venv venv/
```
   3. Install the necessary packages listed in `requirements.txt` by
```
pip install -r requirements.txt
```

### Generate training dataset
To simulate measurements of a $3 \times 3$ surface code state, run the following command in the terminal
```
python -m tn_generative.run_data_generation\
--data_config=tn_generative/data_configs/surface_code_data_config.py\
--data_config.job_id=0 --data_config.task_id=0
```
In `run_data_generation.py`, we usually first run density matrix renormalization group (DMRG) to find the ground state of a specified hamiltonian as our target state. Then we simulate measurements of the target state by sampling the target matrix product state (MPS). 
In `surface_code_data_config.py`, we reconfigure settings for our measurements, determined by `data_config.task_id`.

The config files enables systematic dataset generation on FASRC Cannon cluster at Harvard University.
For instance, to generate data for surface codes of various preconfigured dimensions, use the following command
```
sbatch --array=0-6 --mem=30000 generate_dataset.sh 'surface_code' 'sweep_sc_size_y_3_fn’
```

### Training matrix product state (MPS)
Once we have the dataset available (this step could be skipped using `example_dataset.nc`), we can run the training process.

```
python -m tn_generative.run_training \
--train_config=tn_generative/train_configs/surface_code_training_config.py \
--train_config.job_id=0 \
--train_config.task_id=0 \
--train_config.sweep_name="sweep_sc_3x3_fn" \
--train_config.training.num_training_steps=20 \
```
In `surface_code_training_config.py`, we reconfigure settings for the training process, determined by `train_config.task_id`.
