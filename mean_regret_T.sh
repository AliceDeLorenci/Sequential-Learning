# Bash script to run in parallel the simulations for different ranges of T

# Usage: bash mean_regret_T.sh <python_script> <folder> <p1> <p2> <s2>
# <folder> is the folder where the results will be saved
# <p1> and <p2> are the success probabilities of the arms (optional)
# <s2> is the sub-gaussianity parameter (optional)

mkdir $2 -p

for i in {0..99}
do
    python3 $1".py" $2 $(($i*10+1)) $(($i*10+10)) $3 $4 &
done
wait