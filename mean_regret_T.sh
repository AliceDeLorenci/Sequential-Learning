# Bash script to run in parallel the simulations for different ranges of T

# Usage: bash mean_regret_T.sh <python_script> <folder>
# <folder> is the folder where the results will be saved

mkdir $2 -p

for i in {0..99}
do
    python3 $1".py" $2 $(($i*10+1)) $(($i*10+10)) &
done
wait