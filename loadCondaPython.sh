PATH=/usr/local/anaconda3/bin/:$PATH 
PYTHONPATH=/usr/local/anaconda3/bin

#create envs
#conda config --add channels conda-forge
#conda create -y --name weis-env python=3.8

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


#conda activate weis-env
conda activate weis-env-v1
