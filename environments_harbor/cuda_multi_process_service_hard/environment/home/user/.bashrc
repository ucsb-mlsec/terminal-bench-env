# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
    . /etc/bashrc
fi

# User specific environment
if ! [[ "$PATH" =~ "$HOME/.local/bin:$HOME/bin:" ]]
then
    PATH="$HOME/.local/bin:$HOME/bin:$PATH"
fi
export PATH

# Add common binary paths
export PATH="/usr/local/bin:/usr/bin:$PATH"

# Prompt configuration
export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "

# History settings
export HISTSIZE=1000
export HISTFILESIZE=2000
export HISTCONTROL=ignoredups:ignorespace

# Append to history file, don't overwrite
shopt -s histappend

# Check window size after each command
shopt -s checkwinsize

# User specific aliases
alias ls='ls --color=auto'
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias fgrep='fgrep --color=auto'
alias egrep='egrep --color=auto'

# Safety aliases
alias rm='rm -i'
alias cp='cp -i'
alias mv='mv -i'

# Additional useful aliases
alias ..='cd ..'
alias ...='cd ../..'
alias df='df -h'
alias du='du -h'
alias free='free -h'