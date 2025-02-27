echo "--- STARTING SETUP ---"
# Init Miniconda in the script
source ~/miniconda3/etc/profile.d/conda.sh

# Create environment
conda env create -f config/environment.yml
conda activate drl-frameworks-env
pip install lib/horizon-0.1.tar.gz
conda deactivate
echo "--- SETUP COMPLETED ---"
