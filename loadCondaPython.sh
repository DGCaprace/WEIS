module load Miniconda3/4.9.2
module load GCC/11.2.0 OpenMPI/4.1.2-GCC-11.2.0 
module load OpenBLAS/0.3.18-GCC-11.2.0

#PATH=/usr/local/anaconda3/bin/:$PATH 
#PYTHONPATH=/usr/local/anaconda3/bin

##INSTALL
# conda config --add channels conda-forge
# conda create --name weis-env-TMP2 -y python=3.9
# conda activate weis-env-TMP2 
# conda install -y --file ./environment_nic5.yml
# conda uninstall -y wisdem pyhams
# pip install dearpygui marmot-agents
# conda install -y petsc4py mpi4py ipopt pyoptsparse==2.10.2
# python setup.py develop

 # >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/cecisw/noarch/easybuild/2018.01/software/Miniconda3/4.9.2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/cecisw/noarch/easybuild/2018.01/software/Miniconda3/4.9.2/etc/profile.d/conda.sh" ]; then
        . "/opt/cecisw/noarch/easybuild/2018.01/software/Miniconda3/4.9.2/etc/profile.d/conda.sh"
    else
        export PATH="/opt/cecisw/noarch/easybuild/2018.01/software/Miniconda3/4.9.2/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate weis-env-TMP2



# Making hifi setup available so that we can use it when projecting HiFi DVs onto LoFi yaml
export PYTHONPATH=$PYTHONPATH:~/ATLANTIS/HiFi/ATLANTIS_UM-BYU_utils/SETUP

#For Linux 64, Open MPI is built with CUDA awareness but this support is disabled by default.
#To enable it, please set the environment variable OMPI_MCA_opal_cuda_support=true before
#launching your MPI processes. Equivalently, you can set the MCA parameter in the command line:
#mpiexec --mca opal_cuda_support 1 ...
# 
#In addition, the UCX support is also built but disabled by default.
#To enable it, first install UCX (conda install -c conda-forge ucx). Then, set the environment
#variables OMPI_MCA_pml="ucx" OMPI_MCA_osc="ucx" before launching your MPI processes.
#Equivalently, you can set the MCA parameters in the command line:
#mpiexec --mca pml ucx --mca osc ucx ...
#Note that you might also need to set UCX_MEMTYPE_CACHE=n for CUDA awareness via UCX.
#Please consult UCX's documentation for detail.
