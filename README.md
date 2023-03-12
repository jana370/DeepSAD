# Deep Learning Architectures: DeepSAD

The code is written in `Python 3.10`, the required packages are given in `requirements.txt`.

To run the code clone the repository to the current directory of your machine:
```
git clone https://github.com/jana370/DeepSAD.git
```

It is recommended to set up a virtual environment to run the code: 
```
# pip install virtualenv
cd <path of DeepSAD directory>
python -m virtualenv env
.\env\Scripts\activate
```

Then the required packages can be installed:
```
pip install -r requirements.txt
```

After that the code can be run.  
`DeepSAD.py` is the Deep SAD implementation,  
`make_graphics.py` is the code used for creating the figure for the tests including labeled normal data,  
`make_graphic_pollution.py` is the code used for creating the figure for the tests with mislabeled data.  
```
python DeepSAD.py
python make_graphis.py
python make_graphic_pollution.py
```
&nbsp;

For the Deep SAD implementation different options in the command line can be used:  
`-d` or `--dataset`:&emsp;choose the dataset which will be used; either `"mnist"`, `"fmnist"`, or `"cifar10"` can be used;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;default is `"mnist"`  
`-m` or `--mode`:&emsp;choose the type of loss function, which will be used for Deep SAD;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;`"standard"` will treat labeled normal data the same as unlabeled data and use the weight only for  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;labeled anomalies;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;`"standard_normal"` will use the weight for both labeled normal data and labeled anomalies;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;`"extended"` will use the weight for the labeled normal data and the second weight for the labeled  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;anomalies;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;default is `"standard"`  
`-w` or `--weight`:&emsp;choose the weight that will be used in the loss function; Note, that this only defines the weight  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;for the labeled normal data if the `"extended"` mode is used;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;default is `3`   
`-sw` or `--second_weight`:&emsp; choose the second weight that will be used for the labeled anomalies if the `"extended"`  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;mode is used;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;default is `4`  
`-cn` or `--category_normal`:&emsp;choose category which will be used as the normal class, the following categories  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;are defined for each dataset:  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;MNIST: `0`: 0, 6, 8, and 9; `1`: 1, 4, and 7; `2`: 2, 3, and 5;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;F-MNIST: `0`: T_shirt, Pullover, Coat, and Shirt; `1`: Trouser, and Dress;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;`2`: Sandal, Sneaker, Bag, and Ankleboot;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;CIFAR-10: `0`: plane, car, ship, and truck; `1`: bird, and frog;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;`2`: cat, deer, dog, and horse;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;default is `0`  
`-ca` or `--category_anomaly`:&emsp;choose category which will be used as the anomaly class, the following categories are  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;defined for each dataset:  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;MNIST: `0`: 0, 6, 8, and 9; `1`: 1, 4, and 7; `2`: 2, 3, and 5;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;F-MNIST: `0`: T_shirt, Pullover, Coat, and Shirt; `1`: Trouser, and Dress;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;`2`: Sandal, Sneaker, Bag, and Ankleboot;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;CIFAR-10: `0`: plane, car, ship, and truck; `1`: bird, and frog;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;`2`: cat, deer, dog, and horse;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&ensp;default is `1`  
`-ra` or `--ratio_anomaly`:&emsp;choose the ratio of labeled anomalies that will be used; Note, that the value should  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;be between `0` and `1`;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;default is `0.05`  
`-rn` or `--ratio_normal`:&emsp;choose the ratio of labeled normal data that will be used; Note, that the value should be  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;between `0` and `1`;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;default is `0.0`  
`-rpu` or `--ratio_pollution_unlabeled`:&emsp;choose the ratio of pollution in the unlabeled data; Note, that the value  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;should be between `0` and `1`;   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;default is `0.1`    
`-rpl` or `--ratio_pollution_labeled`:&emsp;choose the ratio of pollution in the labeled anomalies; Note, that the value  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;should be between `0` and `1`;  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&nbsp;default is `0.0`  
&nbsp;

This means, that Deep SAD using MNIST, the `standard` mode, with the weight `3`, the `0` category as normal class, the `1` category as anomaly class, a labeled 
anomaly ratio of `0.05`, no labeled normal data, a pollution of `0.1` in the unlabeled data, and no pollution in the labeled anomalies, can for example be run by using:
```
python DeepSAD.py
```
Deep SAD using CIFAR-10, the `extended` mode, with the weight `2` for labeled normal data, and a secondary weight `4` for labeled anomalies, 
the `1` category as normal class, the `2` category as anomaly class, a labeled anomaly ratio of `0.01`, a labeled normal data ratio of `0.1`, 
a pollution of `0.1` in the unlabeled data, and a pollution of `0.01` in the labeled anomalies, can for example be run by using:
```
python DeepSAD.py -d "cifar10" -m "extended" -w 2 -sw 4 -cn 1 -ca 2 -ra 0.01 -rn 0.1 -rpu 0.1 -rpl 0.01
```

