# cmrf


The cmrf package provides a python implementation of our winning solution [1] for the [MSR-Bing Image Retrieval Challenge](http://research.microsoft.com/en-US/projects/irc/) in conjunction with ACM Multimedia 2015.
* Six individual methods (i.e. Image2text, Text2image, PSI, DeViSE, ConSE and Parzen window),
* Learning optimized weights for relevance fusion,
* Cross-platform support (linux, mac, windows).


### Dependency
* Install [Theano](http://deeplearning.net/software/theano/install.html#install) >= 0.7.0 if you would like to run PSI and DeViSE
* Download [data](http://) and extract into the cmrf directory.
* Download [dataset](http://www.mmc.ruc.edu.cn/research/irc2015/data/rucmmc_irc2015_data.tar.gz) without image visual feature.
* Download image visual feature [ [required (4.7GB)](http://www.mmc.ruc.edu.cn/research/irc2015/data/rucmmc_irc2015_required_feature.tar.gz) | [optional (7.4GB)](http://www.mmc.ruc.edu.cn/research/irc2015/data/rucmmc_irc2015_optional_feature.tar.gz) ].
* Add [simpleknn](simpleknn) to `PYTHONPATH`.
* Change `ROOT_PATH` in [basic/common.py](basic/common.py) to local folder where dataset are stored in.



### Description
In order to generate cross-media relevance, image and query have to be represented in a common space as they are of two distinct modalities. In our package, we implement six individual methods for cross-media relevance computation and a late-fusion method for cross-media relevance fusion.


#####Individual training methods
* [PSI](model/psi_model.py):  utilize stochastic gradient descent with mini-batches to minimize the margin ranking loss of PSI model.
* [DeViSE](model/devise_model.py):  utilize stochastic gradient descent with mini-batches to minimize the margin ranking loss of DeViSE model.
* Other methods have no training process.

#####Individual test methods
* [Image2text](image2text.py): project image into Bag-of-Words space.
* [Text2image](text2image.py): project query into visual feature space.
* [PSI](psi.py):  project image and query of Bag-of-Words into a learned latent space.
* [DeViSE](devise.py):  project image and query of word2vec feature into a learned latent space.
* [ConSE](conse.py):  project image and query into a learned word2vec space.
* [Parzen window](parzenWindow.py): an extreme case of text2image.

#####Relevance fusion
* [Weight optimization](weightOptimization.py): employ Coordinate Ascent to learn optimized weights
* [Relevance fusion](relevanceFusion.py): fuse relevance from different methods with optimized weights.



### Get Started
Please run [doit_all.sh](doit_all.sh) to see if everything is in place.
If it runs successfully, the cross-media relevance of all the query-imge pairs will be written in `result/final.result.txt` folder, and other intermediate results will also appear in `result` folder.


### Note
* If you have not installed the Theano, you could run [doit_4.sh](doit_4.sh) (only Image2text, Text2image, ConSE and Parzen window)
* As a show case, we only run 20 queries. If you want to run all the 1000 queries from Dev set, please rename  'qid.text.all.txt' in `/rootpath/msr2013dev/Annotations/` to 'qid.text.txt'. It will take a while.
* If you would like to use your own dataset, we recommand you to organize dataset in a fixed structure like our data, which can minimize your coding effort.
* The package does not include any visual feature extractors. Features of data need to be pre-computed, and converted to required binary format using [txt2bin.py](simpleknn/txt2bin.py).


### Reference

[1] Jianfeng Dong, Xirong Li, Shuai Liao, Jieping Xu, Duanqing Xu, Xiaoyong Du. [Image Retrieval by Cross-Media Relevance Fusion](http://www.mmc.ruc.edu.cn/research/irc2015/p173-dong.pdf). ACM Multimedia 2015 (Multimedia Grand Challenge Session)
