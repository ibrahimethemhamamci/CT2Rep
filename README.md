# CT2Rep
MICCAI 2024 & CT2Rep: Automated Radiology Report Generation for 3D Medical Imaging
 
 
## Requirements

Before you start, you will need to install the necessary dependencies. To do so, execute the following commands:

```setup
# Navigate to the 'ctvit' directory and install the required packages
cd ctvit
pip install -e .

# Return to the root directory
cd ..
```

## Dataset

An example dataset is provided [example_data_ct2rep.zip](https://huggingface.co/generatect/GenerateCT/blob/main/example_data_ct2rep.zip). This is to show the required dataset structure for CT2Rep and CT2RepLong. For the full dataset, please see [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE).

## Train

To train the models, go to the corresponding directory, and run the command

```train
python main.py --max_seq_length 300 --threshold 10 --epochs 100 --save_dir results/test_ct2rep/ --step_size 1 --gamma 0.8 --batch_size 1 --d_vf 512
```
The threshold is the minimum number of instances in the dataset for a token to be put in the tokens dictionary. You can select the directories, xlsx files, longitudinal files, etc. with the corresponding keyword arguments.

## Citing Us
If you use CT2Rep, we would appreciate your references to [our paper](https://arxiv.org/pdf/2403.06801).

## License
Our codes are released under a [Creative Commons Attribution (CC-BY) license](https://creativecommons.org/licenses/by/4.0/). This means that anyone is free to share (copy and redistribute the material in any medium or format) and adapt (remix, transform, and build upon the material) for any purpose, even commercially, as long as appropriate credit is given, a link to the license is provided, and any changes that were made are indicated. This aligns with our goal of facilitating progress in the field by providing a resource for researchers to build upon. 


## Acknowledgements
This work is an extension of the following repositories: [GenerateCT](https://github.com/ibrahimethemhamamci/GenerateCT), [CT-CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP), [R2Gen](https://github.com/cuhksz-nlp/R2Gen), and [Longitudinal Chest X-ray](https://github.com/celestialshine/longitudinal-chest-x-ray).

