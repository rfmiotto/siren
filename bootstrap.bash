

# This bootstrap aims at combining Conda + Poetry. Conda is usefull to manage non-python
# dependencies, while python dependencies will be mostly taken care by Poetry.
# Conda will be used as a virtual environment manager, while Poetry will handle most of the
# package management stuff. Hence, there is NO need to use `poetry run` or `poetry shell`.
#
# This script generates the lock files to make sure we have a reproducible env on any machine.
# The bootstrap step creates the lock-files. Then, at the end of the script, we create and
# initialize a new environment based on these lock-files.
#
# Conda-specific files needed for the bootstrap:
#   environment.yml  virtual-packages.yml
#
# Conda-specific files (automatically generated):
#   conda-lock.yml  conda-win-64.lock
# Poetry-specific files:
#   pyproject.toml  poetry.lock
#
# After executing this script, it is a good practive to commit these files on git.


#######################################################################################################
#                                            Bootstrap
#######################################################################################################

# Create a bootstrap environment (python version must match the one from environment.yml)
conda create -p $PWD/bootstrap -c conda-forge mamba conda-lock poetry='1.*' python='3.11' -y
conda activate $PWD/bootstrap

# Create Conda lock file(s) from environment.yml
conda-lock -f environment.yml -p win-64 -p linux-64 --conda mamba

# Set-up Poetry
poetry init --python=^3.11  # this version should match the one from environment.yml
poetry add --lock pytorch=2.2.0 pytorch-cuda=11.8  # prevent upgrades of packages installed by Conda
poetry add --lock conda-lock 
# Add other packages here
# poetry add --lock 

# Display env information
conda info

# Render platform-specific lockfile:
conda-lock render -p linux-64

# Remove bootstrap env
conda deactivate
conda remove -p $PWD/bootstrap --all -y



#######################################################################################################
#                                            Initialization
#######################################################################################################
# This step is independent from bootstrap. You can execute it afterwards if needed, as long as the
# bootstrap has been executed before.

YELLOW='\033[1;33m'
RED='\033[0;31m'
NO_COLOR='\033[0m'

printf "
${RED} 
WARNING:
${YELLOW} 
If any Python packages were specified in environment.yml, it is necessary to add an entry of these packages in Poetry's pyproject.toml, with the same version (without ^ or ~ before the version number). This will let Poetry know that the package is there and should not be upgraded.

Once the pyproject.toml was properly modified, a new environment can be initialized using the lockfiles by running:
    conda create -p $PWD/env --file conda-linux-64.lock
    conda activate $PWD/env
    poetry install

If there are still python packages left to be installed, you can do so by running:
    poetry add --lock <packages>
    poetry install


UPDATING THE ENVIRONMENT:
- Re-generate Conda lock file(s) based on environment.yml:
    conda-lock -k explicit --conda mamba
- Update Conda packages based on re-generated lock file:
    mamba update --file conda-linux-64.lock
- Update Poetry packages and re-generate poetry.lock:
    poetry update

${NO_COLOR}
Make changes then proceed with instalation. 
"
