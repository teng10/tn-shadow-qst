# tn-shadow-qst
Code and data for **Learning topological states from randomized measurements using
variational tensor network tomography** (arXiv link: )

- A tutorial on how to use the codebase is in [draft_tutorials.ipynb](https://github.com/teng10/tn-shadow-qst/blob/e0347b7d64ef86c7564efa0a95e13008d9dfeab8/draft_tutorial.ipynb).
- To reproduce figures from the paper with the dataset linked [here](https://doi.org/10.5281/zenodo.11397880) , see [`Draft_figures.ipynb`](https://github.com/teng10/tn-shadow-qst/blob/bd3f62930849889fba854b96f6da129fc1c99e51/Draft_figures.ipynb).


## Getting started

### Set up the environment with python version control.


   1. Git clone the current repository:
```
git clone https://github.com/teng10/tn-shadow-qst.git
```
   2. Create a virtual environment:
```
python3 -m venv venv/
```
   3. Install the necessary packages listed in `requirements.txt`:
```
pip install -r requirements.txt
```

## Command line demo
In this demo below, we will perform tomography for the ground state of a a $3 \times 3$ surface code Hamiltonian. For a notebook version, see [draft_tutorials.ipynb](https://github.com/teng10/tn-shadow-qst/blob/e0347b7d64ef86c7564efa0a95e13008d9dfeab8/draft_tutorial.ipynb). This involves two steps:
1. Using run_data_generation.py, we will run the density matrix renormalization group (DMRG) to find the ground state of a specified Hamiltonian as our target state. Then, we simulate measurements of the target state by sampling the target matrix product state (MPS). (This step can be skipped if you already have a dataset, as we demonstrate next!)
2. After generating this dataset, we can perform training with a matrix product state ansatz by runnnig `run_training.py`.

### Generate training dataset
To simulate measurements of a $3 \times 3$ surface code state, run the following command in the terminal
```
python -m tn_generative.run_data_generation \
--data_config=tn_generative/data_configs/surface_code_data_config.py \
--data_config.job_id=0 --data_config.task_id=0 \
--data_config.output.data_dir=./
```
In `surface_code_data_config.py`, we reconfigure settings for our measurements, determined by `data_config.task_id`.
The above command will output a file such as `0_surface_code_xz_basis_sampler_size_x=3_size_y=3_d=5_onsite_z_field=0.000.nc`

 - The config file enables systematic dataset generation via a clustering computing platform.
   For instance, on FASRC Cannon cluster at Harvard University, to generate data for surface codes of various preconfigured dimensions, use the following command:
   ```
   sbatch --array=0-6 --mem=30000 generate_dataset.sh 'surface_code' 'sweep_sc_size_y_3_fn’
   ```

### Training matrix product state (MPS)
Once we have the dataset available (this step could be skipped using a precomputed dataset `surface_code_xz.nc` or an dataset from experiments as long as they are in the same format), we can run the training process.

```
python -m tn_generative.run_training \
--train_config=tn_generative/train_configs/surface_code_train_config.py \
--train_config.job_id=0530 \
--train_config.task_id=0 \
--train_config.data.dir=tn_generative/test_data/ \
--train_config.data.filename=surface_code_xz.nc \
--train_config.results.experiment_dir=./ \
--train_config.training.steps_sequence="(5000,400)" \
--train_config.data.num_training_samples=10000 \
--train_config.model.bond_dim=10
```
The results are saved in directory specified by `train_config.results.experiment_dir`. This command outputs the following files:
1. `530_0_eval.csv`: Contains evaluation results, including fidelity (`fidelity`) and train and test losses (`model_ll`).
2. `530_0_mps_lbfgs_1.nc`: Contains the trained MPS state after the L-BFGS sequence.
3. `530_0_train.csv`: Contains the training results, containing `loss` for each training step, as well as comprehensive descriptions of all the parameter configurations for the training process. 
