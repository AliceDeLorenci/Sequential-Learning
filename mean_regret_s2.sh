# Bash script to run in parallel the simulations for different ranges of sigma^2 (UCB algorithm)

# Usage: bash mean_regret_s2.sh <python_script> <folder> <p1> <p2>
# <folder> is the folder where the results will be saved

mkdir $2 -p

for i in 0 0.03125 0.0625 0.25 1
do
    python3 $1".py" $2 $i $3 $4 &
done
wait