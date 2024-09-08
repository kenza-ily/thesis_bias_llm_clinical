# .bash_profile
# Get the aliases and functions
if [ -f ~/.bashrc ]; then
. ~/.bashrc
fi
# User specific environment and startup programs
PATH=$PATH:$HOME/.local/bin:$HOME/bin
export PATH

# Add BIAS_LLM_CONFIG environment variable
export BIAS_LLM_CONFIG="/Users/kenzabenkirane/Desktop/GitHub/24ucl_thesis/bias_llm_clinical_nle/config"