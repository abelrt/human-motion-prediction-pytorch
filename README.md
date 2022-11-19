# human-motion-prediction 🕺🏻

> **Note:** This repo is a fork of this one: https://github.com/cimat-ris/human-motion-prediction-pytorch
>
> The code has been refactored preserving the logic and structure, but adding functionalities to run it in Google Colab.

> Pytorch implementation of:
>
> &nbsp;&nbsp; Julieta Martinez, Michael J. Black, Javier Romero. _**On human motion prediction using recurrent neural networks**_. In CVPR 17.

The paper can be found on arXiv: [https://arxiv.org/pdf/1705.02445.pdf](https://arxiv.org/pdf/1705.02445.pdf)

The code in the original repository was written by [Julieta Martinez](https://github.com/una-dinosauria/) and [Javier Romero](https://github.com/libicocco/) and is accessible [here](https://github.com/enriccorona/human-motion-prediction-pytorch).

### Dependencies

- [h5py](https://github.com/h5py/h5py) -- to save samples
- [Pytorch](https://pytorch.org/)

> Some other dependencies have been listed in the `requirements.txt` file. If you want to create an environment (using conda) with the samples, you can use the following command:
>
> ```sh
> conda create -n muframex python=3.9 -y
> pip install -r requirements.txt
> ```

### Get this code and the data

First things first, clone this repo and get the human3.6m dataset on exponential map format.

To download code:

```sh
git clone https://github.com/RodolfoFerro/human-motion-prediction-pytorch.git
cd human-motion-prediction-pytorch
```

To download data:

```sh
mkdir data
cd data

# Install gdown: pip install gdown
gdown https://drive.google.com/uc?id=1hqE6GrWZTBjVzmbehUBO7NTrbEgDNqbH
unzip -q h3.6m.zip
rm h3.6m.zip
cd ..
```

> You can also download the zip file from [here](https://drive.google.com/file/d/1hqE6GrWZTBjVzmbehUBO7NTrbEgDNqbH/view?usp=sharing)

### Quick demo and visualization

For a quick demo, you can train for a few iterations and visualize the outputs
of your model.

To train the model, run

```bash
python src/train.py --action walking --seq_length_out 25 --iterations 10000
```

To test the model on one sample, run

```bash
python src/test.py --action walking --seq_length_out 25 --iterations 10000 --load 10000
```

Finally, to visualize the samples run

```bash
python src/animate.py
```

This should create a visualization similar to this one

<p align="center">
  <img src="https://raw.githubusercontent.com/una-dinosauria/human-motion-prediction/master/imgs/walking.gif"><br><br>
</p>

You can substitute the `--action walking` parameter for any action in

```
["directions", "discussion", "eating", "greeting", "phoning",
 "posing", "purchases", "sitting", "sittingdown", "smoking",
 "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
```

or `--action all` (default) to train on all actions.

### Citing

If you use our code, please cite our work

```
@inproceedings{julieta2017motion,
  title={On human motion prediction using recurrent neural networks},
  author={Martinez, Julieta and Black, Michael J. and Romero, Javier},
  booktitle={CVPR},
  year={2017}
}
```

### Acknowledgments

The pre-processed human 3.6m dataset and some of our evaluation code (specially under `src/data_utils.py`) was ported/adapted from [SRNN](https://github.com/asheshjain399/RNNexp/tree/srnn/structural_rnn) by [@asheshjain399](https://github.com/asheshjain399).

### Licence

MIT
