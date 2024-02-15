for i in {0..99}
do
    python3 question1d.py $(($i*10+1)) $(($i*10+10)) &
done
wait