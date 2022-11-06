python prune.py --method fixed --fixed 0.99
python prune.py --method fixed --fixed 0.98
python prune.py --method fixed --fixed 0.95
python prune.py --method fixed --fixed 0.9
python prune.py --method fixed --fixed 0.85
python prune.py --method fixed --fixed 0.8
python prune.py --method fixed --fixed 0.75

python prune.py --method wcss --wcss_from 0.9  --wcss_to 1.0
python prune.py --method wcss --wcss_from 0.75 --wcss_to 1.0

python prune.py --method inertia --inertia 1
python prune.py --method inertia --inertia 2
python prune.py --method inertia --inertia 3
python prune.py --method inertia --inertia 5
python prune.py --method inertia --inertia 20


python prune.py --only_first_layer --method fixed --fixed 0.95
python prune.py --only_first_layer --method fixed --fixed 0.75
python prune.py --only_first_layer --method fixed --fixed 0.5

python prune.py --only_first_layer --method wcss  --wcss_from 0.7 --wcss_to 0.8
python prune.py --only_first_layer --method wcss  --wcss_from 0.4 --wcss_to 0.6
