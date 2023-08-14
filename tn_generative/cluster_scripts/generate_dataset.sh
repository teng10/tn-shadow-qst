#!/bin/bash
#SBATCH -n 1                # Number of cores (-n)
#SBATCH -N 1                # Ensure that all cores are on one Node (-N)
#SBATCH -t 2-00:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p serial_requeue #shared #  # Partition to submit to
#SBATCH --mail-user=y_teng@g.harvard.edu #Email for notifications
#SBATCH --mail-type=END #This command would send an email when the job ends.
#SBATCH --mem=50000
#SBATCH -c 1
#SBATCH --array=0-11 # 30 different parameter settings
# enumerating in parameter {bond_dimension}, {onsite_z_field} index.
# e.g. 2 x 3=6 This enumerates 6 parameter setting
#SBATCH -o /n/home11/yteng/experiments/TNS/logsTNS/%j.out # Standard out
#SBATCH -e /n/home11/yteng/experiments/TNS/logsTNS/%j.err # Standard err
module load python/3.10.9-fasrc01
source activate tn-shadow-qst
package_path="/n/home11/yteng/tn-shadow-qst/"
cd ${package_path}
FILEPATH="/n/home11/yteng/experiments/TNS"
data_dir="$FILEPATH/dataTNS/%CURRENT_DATE/"
mkdir -p ${data_dir}
echo "Data saving directory is ${data_dir}"
echo "array_task_id=${SLURM_ARRAY_TASK_ID}"

python -m tn_generative.run_data_generation \
--data_config=tn_generative/data_configs/surface_code_data_config.py \
--data_config.job_id=${SLURM_ARRAY_JOB_ID} \
--data_config.task_id=${SLURM_ARRAY_TASK_ID} \
--data_config.output.data_dir=${data_dir} \
--data_config.sweep_name = "sweep_sc_5x5_fn"

echo "job finished"
> "$FILEPATH/logsTNS/${SLURM_ARRAY_JOB_ID}_log.txt"
