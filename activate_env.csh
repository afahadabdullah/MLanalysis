# tcsh/csh version of activate_env.sh:   source activate_env.csh
module load miniforge >& /dev/null
set _cb = `conda info --base`
if ( -f $_cb/etc/profile.d/conda.csh ) source $_cb/etc/profile.d/conda.csh
conda activate /home/afahad/project/MLanalysis/envs/gc
unset _cb
setenv PYTHONNOUSERSITE 1
setenv PROJ /home/afahad/project/MLanalysis
setenv XLA_PYTHON_CLIENT_PREALLOCATE false
echo "Environment active: `which python`"
