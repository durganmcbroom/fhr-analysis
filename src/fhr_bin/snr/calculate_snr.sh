FILES=(
  "draw1_Y2_3500_outsideS/draw1_Y2_3500_outsideS_1"
  "draw1_Y2_3500_outsideS/draw1_Y2_3500_outsideS_2"
)

for NAME in "${FILES[@]}"
do
    echo "python3 calculate_snr.py data/$NAME.csv out/$NAME.png --window=10-17"
    python3 calculate_snr.py data/$NAME.csv out/$NAME.png --window=10-17 --floor=3
done

FILES=(
  "H4_undershirt_new_connection/H4_undershirt_new_connection_2"
  "H4_undershirt_new_connection/H4_undershirt_new_connection_3"
  "15-2"
  "draw2_y1_5800_inside_sweater"
  "H3_undershirt_new_connection"
  "Y4_undershirt_new_connection"
  "Y3_undershirt_grounding"
  "draw1Y1_on_H"
  "H3"
  "Y4"
  "H4_new/H4_new_1"
  "H4_new/H4_new_2"
  "Y4_after_braiding/Y4_after_braiding_1"
  "Y4_after_braiding/Y4_after_braiding_2"
  "H3_new"
)

for NAME in "${FILES[@]}"
do
    echo "python3 calculate_snr.py data/$NAME.csv out/$NAME.png"
    python3 calculate_snr.py data/$NAME.csv out/$NAME.png --floor=3
done

python3 calculate_snr.py data/Y4_higher.csv out/Y4_higher.png --window=1-15 --floor=3
python3 calculate_snr.py data/H4.csv out/H4.png --window=3-20 --floor=3
python3 calculate_snr.py data/draw2_Y1_5800V_outside_sweater.csv out/draw2_Y1_5800V_outside_sweater.png --window=1-20 --floor=3
python3 calculate_snr.py data/H4_undershirt_new_connection/H4_undershirt_new_connection_1.csv out/H4_undershirt_new_connection/H4_undershirt_new_connection_1.png --window=1-10 --floor=3
python3 calculate_snr.py data/117-2.csv out/117-2.png --window=2-10 --floor=3
python3 calculate_snr.py data/Y2_3300.csv out/Y2_3300.png --window=1-10 --floor=3

