# Isolation PvP

Modified Isolation library and scripts to run a PvP Isolation competition for the AIND.

# Setup

- Clone this repository
- Put a copy of a default "competition_agent.py" implementation in the base directory (this file is used to "balance" tournaments with an odd number of agents)
- Put all competing agents in their own directories within a target folder. For example:
```
    AIND-Isolation_PvP/  
        competition_agent.py  # default agent for balance  
        agents/  # player agents go here  
            agent1/  
                competition_agent.py    
            agent2/  
                competition_agent.py  
            ...  
            agentN/  
                competition_agent.py
```

# Usage

Run the script by completing the steps in setup, then executing `python run.py <options>` as shown below.

    usage: run.py [-h] [-i INPUT] [-p PROCESSES] [-d] [-t]
                  [-l {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}]

    optional arguments:
      -h, --help            show this help message and exit
      -i INPUT, --input INPUT
                            Name of the input folder containing agent folders
      -p PROCESSES, --processes PROCESSES
                            Number of processes to use for running matches
      -d, --database        Name to use for sqlite database of results (ignored if
                            -t is used)
      -t, --temp            Use an in-memory database (results not saved)
      -l {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}, --loglevel {CRITICAL,ERROR,WARNING,INFO,DEBUG,NOTSET}
