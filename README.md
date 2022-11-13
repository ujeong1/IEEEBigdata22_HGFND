# Setups
1. [Create environment using Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
: We use following command to make conda env on Ubuntu 20.04 using Conda with python 3.7.6

```bash
conda create --name HGFND python==3.7.6
source activate HGFND
```
2. [Pytorch Installation on Conda](https://pytorch.org/)
: We use following command to install pytorch based on CUDA 10.2 for nvidia driver (Please check your CUDA version by <code>nvcc --version</code> before the installation)

```bash
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=10.2 -c Pytorch
```

3. [PyG Installation on Conda](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
: We use Pytorch 1.9.0 Linux Conda CUDA 10.2

```bash
conda install pyg -c pyg
```

4. Install Dependencies using <code> pip install -r requirements.txt </code>

# Usage
1. Run the code with following commandline/parameters
```bash
 python main.py --use_user true --use_date true --use_entity true --dataset politifact
```

# Trouble Shooting
1. Please apply <code>pip uninstall torch_spline_conv</code> if you encounter the following error:

```bash
OSError: /lib64/libm.so.6: version `GLIBC_2.27' not found
```

2. Please refer to [PyG FAQ](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#id1) for any types of error caused by PyG. 


# Dataset
- The UPFD dataset will be automatically downloaded to <code> data/{dataset-name}/raw/</code> when runnig <code> main.py </code>
- You can also manually download the dataset from [UPFD github](https://github.com/safe-graph/GNN-FakeNews)

