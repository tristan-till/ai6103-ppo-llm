#!/bin/bash
#SBATCH --partition=M1
#SBATCH --qos=q_d8_norm
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=6G
#SBATCH --job-name=ppo-llm
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
#SBATCH --time=06:00:00

module load cuda/12.2.2

eval "$(conda shell.bash hook)"
echo "we are here!"
# conda activate ppo-llm

# Run ollama server
ollama serve &

# Run the required models
ollama run llama3.2:1b


echo "Loaded Model!"
full_response=""
done=false

# Verify model is working
curl -s http://localhost:11434/api/generate \
  -d '{"model": "llama3.2:1b","prompt":"Ready?"}' | while read -r line; do

  response_part=$(echo "$line" | jq -r '.response')
  done=$(echo "$line" | jq -r '.done')

  full_response+="$response_part"

  if [[ $done == "true" ]]; then
    echo "Full Response: $full_response"
    break
  fi
done

python3 main.py

