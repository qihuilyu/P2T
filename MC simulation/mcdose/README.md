# Deep Monte Carlo Dose Predictor

**install python pre-requisites:**

0. (optional) initialize a virtual environment and activate it
1. run: `pip3 install -r requirements.txt`

## Actions
### Training
Some examples of common _training_ operations are presented below:
```bash
# Begin training on data in <train-data-dir> 
#   and store model checkpoints/logs/tflogs in <run-output-dir>
python main.py train --datadir "<train-data-dir>" --rundir "<run-output-dir>"
```

```bash
# Resume previously initiated training run (with run number "<run-number>") from last checkpoint
# (--resume will raise an error if resume fails, otherwise it will warn and restart training)
python main.py train --datadir "<train-data-dir>" --rundir "<run-output-dir>" \
    --run "<run-number>" --resume
```

```bash
# Begin training with selected learning rate and stop after selected number of epochs
python main.py train --datadir "<train-data-dir>" --rundir "<run-output-dir>" \
    --epochs 150 --lr 1e-4
```

```bash
# Begin training with non-default configuration file
python main.py --config "<config-file>" train --datadir "<train-data-dir>" \
    --rundir "<run-output-dir>" --epochs 150 --lr 1e-4
```

```bash
# Begin training on CPU only (experimental)
python main.py --cpu train --datadir "<train-data-dir>" --rundir "<run-output-dir>"
```

```bash
# Training with a larger file cache (default=3; higher requires more RAM 
#   but should speed up training)
python main.py --cache-size 6 train --datadir "<train-data-dir>" --rundir "<run-output-dir>"
```

Monitoring the training progress:
```bash
# Start tensorboard instance and specify the <run-output-dir> used during training
# (optionally: add --bind_all to make accessible from other computers on the same network)
tensorboard --logdir "<run-output-dir>" --bind_all
```


### Testing
Once the model is trained and the model weights have been saved to a checkpoint file within a _run_ folder, 
performance testing can be carried out using one of the forms below:

```bash
# Load previously trained model (with run number "<run-number>") and calculate performance 
# metrics on the testing dataset (optional tests such as gamma analysis, plot output can be
# enabled with cmd line flags)
python main.py test --datadir "<data-dir>" --rundir "<run-output-dir>" --run "<run-number>" \
    --gamma --plots
```


### Prediction
Once the model is trained and the model weights have been saved to a checkpoint file within a _run_ folder, 
it can be loaded and used to perform inference on new examples.

**COMING SOON**


## Running with Nvidia-Docker
First build the image from Dockerfile
```bash
./docker-build.sh
```

Then start an interactive session to be placed in a shell inside the running container. Be sure to map a volume to access training data from the docker container's filesystem and make the results visible to your local filesystem.
```
# Start the container and attach to it
./docker-attach.sh  -v "<local-data-dir>:/data"  [optional 'docker run' args]

# Run the desired action from witihin the container
python main
```

* `<local-dir>` can be anywhere on the host machine but must be pre-configured with a child directory called `traindata` that contains three required directories `train`, `test`, and `validate`. Each of these must contain at least one .npy file. The identification of .npy files is recursive and based only on the presence of the extension `.npy`, so they may named arbitrarily and optionally organized into sub-folders as shown below:
```
<local-dir>/data/
├── test
│   └── subdir_1
│       ├── test_000.npy
│       ├── test_000.txt
│       └── ...
├── train
│   └── subdir_1
│       ├── train_000.npy
│       ├── train_000.txt
│       └── ...
└── validate
    └── subdir_1
        ├── train_001.npy
        ├── train_001.txt
        └── ...
```
