# IDD-Lite-Unet

# Training procedure 

1. Download the dataset in a folder "dataset" and arrange the data in the following structure:

```bash
├── dataset
│   ├── images
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   ├── annotation
│   │   ├── train
│   │   ├── val
```
2. Execute the command 
```bash
bash run.sh
```
# Testing procedure

1. For single scale testing:
```bash
python test.py
```
2. For multi-single scale testing:
```bash
python multiscale_testing.py
```
3. For single scale testing with CRF post processing 
```bash
python test_with_postprocessing.py
```

# Visualization 

Left side : predicted classes  :  Right side : Ground Truth 

![Visualization during training](images/visual.png?raw=true "sample of generated data")

