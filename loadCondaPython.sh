PATH=/usr/local/anaconda3/bin/:$PATH 
PYTHONPATH=/usr/local/anaconda3/bin

#create envs
#conda env create --name weis-env-v1.1 -f https://raw.githubusercontent.com/WISDEM/WEIS/develop/environment.yml python=3.9


# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<


# Making hifi setup available so that we can use it when projecting HiFi DVs onto LoFi yaml
export PYTHONPATH=$PYTHONPATH:/Users/dcaprace/OneDrive\ -\ BYU/BYU_ATLANTIS/ATLANTIS_UM-BYU_utils/SETUP


conda activate weis-env-v1.1
