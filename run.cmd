:all_layers
python run.py --method fixed --fixed 0.99
python run.py --method fixed --fixed 0.98
python run.py --method fixed --fixed 0.95
python run.py --method fixed --fixed 0.9
python run.py --method fixed --fixed 0.85
python run.py --method fixed --fixed 0.8
python run.py --method fixed --fixed 0.75

python run.py --method wcss --wcss_from 0.9  --wcss_to 1.0
python run.py --method wcss --wcss_from 0.75 --wcss_to 1.0

python run.py --method inertia --inertia 1
python run.py --method inertia --inertia 2
python run.py --method inertia --inertia 3
python run.py --method inertia --inertia 5

:only_first_layer
python run.py --only_first_layer --method fixed --fixed 0.95
python run.py --only_first_layer --method fixed --fixed 0.75
python run.py --only_first_layer --method fixed --fixed 0.5

python run.py --only_first_layer --method wcss --wcss_from 0.7 --wcss_to 0.8
python run.py --only_first_layer --method wcss --wcss_from 0.4 --wcss_to 0.6

python run.py --only_first_layer --method inertia --inertia 1
python run.py --only_first_layer --method inertia --inertia 2
python run.py --only_first_layer --method inertia --inertia 3
python run.py --only_first_layer --method inertia --inertia 5
