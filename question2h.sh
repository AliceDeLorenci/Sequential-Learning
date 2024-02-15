for i in 0 $((1/32)) $((1/16)) $((1/4)) 1
do
    python3 question2h.py $i &
done
wait