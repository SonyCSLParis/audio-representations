# Audio representation learning with JEPAs

This repository contains the PyTorch code associated to the paper [Investigating Design Choices in Joint-Embedding Predictive Architectures for General Audio Representation Learning](), presented at the [SASB workshop](https://sites.google.com/view/icasspsasb2024/description) at [ICASSP 2024](https://2024.ieeeicassp.org/).

## Usage

- Clone the repository and install the requirements using the provided `requirements.txt` or `environment.yml`.

- Then, preprocess your dataset to convert audios into mel-spectrograms:

  ```sh
  python wav_to_lms.py /your/local/audioset /your/local/audioset_lms
  ```
  
- Write the list of files to use as training data in a csv file
  ```sh
  cd data
  echo file_name > files_audioset.csv
  find /your/local/audioset_lms -name "*.npy" >> files_audioset.csv
  ```

- You can now start training! We rely on [Dora](https://github.com/facebookresearch/dora/tree/main) for experiment scheduling. For start an experiment locally, just type: 

  ```sh
  dora run
  ```

  Under the hood, [Hydra](https://hydra.cc/) is used for handle configurations, so you can override configurations via CLI or build your own YAML config files. For example, type:

  ```sh
  dora run data=my_dataset model.encoder.embed_dim=1024
  ```

  to train our model with a larger encoder on your custom dataset.

  Moreover, you can seamlessly launch SLURM jobs on a cluster thanks to Dora:

  ```sh
  dora launch -p partition-a100 -g 4 data=my_dataset
  ```

  We refer to the respective documentations of [Hydra](https://hydra.cc/) and [Dora](https://github.com/facebookresearch/dora/tree/main) for more advanced usage.

## Performances

Our model is evaluated on 8 various downstream tasks, including environmental, speech and music classification ones. Please refer to our paper for additional details.

![alt text](https://github.com/SonyCSLParis/audio-representations/blob/master/images/results.png?raw=true)

## Checkpoints

Will be available soon...

## Credits

- This great [Lightning+Hydra template](https://github.com/ashleve/lightning-hydra-template)
- [EVAR](https://github.com/nttcslab/eval-audio-repr) for evaluating our representations

