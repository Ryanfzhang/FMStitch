#!/bin/bash
SEED=${1:-2}  

# FQL on D4RL antmaze-umaze-v2
python main.py --env_name=antmaze-umaze-v2 --agent=agents/iql.py --offline_steps=500000 --agent.alpha=10 --seed $SEED
# FQL on D4RL antmaze-umaze-diverse-v2
python main.py --env_name=antmaze-umaze-diverse-v2 --agent=agents/iql.py --offline_steps=500000 --agent.alpha=10 --seed $SEED
# FQL on D4RL antmaze-medium-play-v2
python main.py --env_name=antmaze-medium-play-v2  --agent=agents/iql.py --offline_steps=500000 --agent.alpha=10 --seed $SEED
# FQL on D4RL antmaze-medium-diverse-v2
python main.py --env_name=antmaze-medium-diverse-v2 --agent=agents/iql.py --offline_steps=500000 --agent.alpha=10 --seed $SEED
# FQL on D4RL antmaze-large-play-v2
python main.py --env_name=antmaze-large-play-v2 --agent=agents/iql.py --offline_steps=500000 --agent.alpha=10 --seed $SEED
# FQL on D4RL antmaze-large-diverse-v2
python main.py --env_name=antmaze-large-diverse-v2 --agent=agents/iql.py --offline_steps=500000 --agent.alpha=10 --seed $SEED
