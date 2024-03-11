### Implementation of the paper Variational Convolutional Neural Network Pruning, presented at CVPR 2019

The file ```layers.py``` implements Batch Normalization functions as proposed in the paper.

2D is used in the paper, added implementations for 1D and 3D variants.

Also added another implementation that might work better.

### Usage

Place the file to a folder of your choice, or copy its contents, import whichever function you wish.

The functions inherit from nn.BatchNorm so they should work as drop-in replacements.

At your model add the following code to get the KL divergence loss during training:

```
def get_kl(self):

    total_kl = 0.0
    for m in self.modules():
        if isinstance(m, BatchNorm2dPruning) or isinstance(m, BatchNorm1dPruning) or isinstance(m, BatchNorm3dPruning):
            total_kl += m.get_kl()

    return total_kl
```

Then call this function and add the KL loss to your training loss.

Finally, to prune channels:

```
def prune_channels(self):
        for m in self.modules():
            if isinstance(m, BatchNorm2dPruning) or isinstance(m, BatchNorm1dPruning) or isinstance(m, BatchNorm3dPruning):
                m.prune_channels()
        return

```

Same code applies for the alternative implementations (change names in the ```if``` statements).

### Citation
Cite the original paper as:
```
@inproceedings{zhao2019variational,
	title={Variational Convolutional Neural Network Pruning},
	author={Zhao, Chenglong and Ni, Bingbing and Zhang, Jian and Zhao, Qiwei and Zhang, Wenjun and Tian, Qi},
	booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
	pages={2775--2784},
	year={2019},
	month={June~15--19},
	location={Long Beach, CA, USA},
	doi={10.1109/CVPR.2019.00289},
	publisher={IEEE}
}
```
