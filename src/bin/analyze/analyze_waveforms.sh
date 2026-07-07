BANNER_DATA="../../../Banner_data/Banner_test_20251220"

echo python3 analyze_waveforms.py "$BANNER_DATA/Patient 6" "$BANNER_DATA/Patient 7" --out-dir="./out"
python3 analyze_waveforms.py "$BANNER_DATA/Patient 6" "$BANNER_DATA/Patient 7" "$BANNER_DATA/patient8-session1" --out-dir="./out"
#python3 analyze_waveforms.py "$BANNER_DATA/patient8-session1" "$BANNER_DATA/Patient 7" --out-dir="./out"
