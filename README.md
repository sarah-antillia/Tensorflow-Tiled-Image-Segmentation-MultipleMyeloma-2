<h2>Tensorflow-Tiled-Image-Segmentation-MultipleMyeloma-2 (2025/04/23)</h2>

Sarah T. Arai<br>
Software Laboratory antillia.com
<br><br>
This is the third experiment of Tiled Image Segmentation for Tiled-MultipleMyeloma
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and a tiled dataset <a href="https://drive.google.com/file/d/105Ppwc5X92_qJhreS1NWUx1-DaCuQd6I/view?usp=sharing">
Tiled-MultipleMyeloma-ImageMask-Dataset.zip</a>, which was derived by us from
<a href="https://segpc-2021.grand-challenge.org/">
Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images (SegPC-2021) 
</a>
<br>
<br>
Please see also our experiments:<br>
<li><a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-MultipleMyeloma">
Tensorflow-Tiled-Image-Segmentation-MultipleMyeloma</a></li>
<li><a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma">
Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma</a></li>

<br>
<b>Experiment Strategies</b><br>
In this experiment, we employed the following strategies.
<br>
<b>1. Tiled ImageMask Dataset</b><br>
We trained and validated a TensorFlow UNet model using the Tiled-MultipleMyeloma-ImageMask-Dataset, which was tiledly-splitted to 512x512 pixels image and mask dataset from the original 2560x1920 pixels images and mask files.<br>
<br>

<b>2. Tiled Image Segmentation</b><br>
We applied our Tiled-Image Segmentation inference method to predict the HardExudates regions for the mini_test images 
with a resolution of 2560x1920 pixels.<br><br>

<hr>
<b>Actual Tiled Image Segmentation for Images of 2560x1920 pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/102.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/102.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/102.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/106.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/106.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/106.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/207.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/207.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/207.jpg" width="320" height="auto"></td>
</tr>

</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Tiled-MultipleMyelomaSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The original dataset used here has been take from the following web site:
<br>
<b>SegPC-2021-dataset</b><br>
SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images<br>
https://www.kaggle.com/datasets/sbilab/segpc2021dataset<br>
<br>
<b>Citation:</b><br>

Anubha Gupta, Ritu Gupta, Shiv Gehlot, Shubham Goswami, April 29, 2021, "SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells <br>
in Microscopic Images", IEEE Dataport, doi: https://dx.doi.org/10.21227/7np1-2q42.<br>
<br>
<b>BibTex</b><br>
@data{segpc2021,<br>
doi = {10.21227/7np1-2q42},<br>
url = {https://dx.doi.org/10.21227/7np1-2q42},<br>
author = {Anubha Gupta; Ritu Gupta; Shiv Gehlot; Shubham Goswami },<br>
publisher = {IEEE Dataport},<br>
title = {SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images},<br>
year = {2021} }<br>
<br>
<b>IMPORTANT:</b><br>
If you use this dataset, please cite below publications-<br>
1. Anubha Gupta, Rahul Duggal, Shiv Gehlot, Ritu Gupta, Anvit Mangal, Lalit Kumar, Nisarg Thakkar, and Devprakash Satpathy,<br> 
 "GCTI-SN: Geometry-Inspired Chemical and Tissue Invariant Stain Normalization of Microscopic Medical Images," <br>
 Medical Image Analysis, vol. 65, Oct 2020. DOI: <br>
 (2020 IF: 11.148)<br>
2. Shiv Gehlot, Anubha Gupta and Ritu Gupta, <br>
 "EDNFC-Net: Convolutional Neural Network with Nested Feature Concatenation for Nuclei-Instance Segmentation,"<br>
 ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), <br>
 Barcelona, Spain, 2020, pp. 1389-1393.<br>
3. Anubha Gupta, Pramit Mallick, Ojaswa Sharma, Ritu Gupta, and Rahul Duggal, <br>
 "PCSeg: Color model driven probabilistic multiphase level set based tool for plasma cell segmentation in multiple myeloma," <br>
 PLoS ONE 13(12): e0207908, Dec 2018. DOI: 10.1371/journal.pone.0207908<br>
<br>
<b>License</b><br>
CC BY-NC-SA 4.0
<br>
<br>
<h3>
<a id="2">
2 Tiled-MultipleMyelomaImageMask Dataset
</a>
</h3>
 If you would like to train this Tiled-MultipleMyelomaSegmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/105Ppwc5X92_qJhreS1NWUx1-DaCuQd6I/view?usp=sharing">
Tiled-MultipleMyeloma-ImageMask-Dataset.zip</a>,
 expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─Tiled-MultipleMyeloma
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
This is a 512x512 pixels tiles dataset generated from 2560x1920 pixels <b>original images</b> and
their corresponding <b>masks</b>.<br>
.<br>
On the derivation of this tiled dataset, please refer to the following Python scripts.<br>
<a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-MultipleMyeloma">Tiled-ImageMask-Dataset-MultipleMyeloma</a>
<br>


<br>
<b>Tiled-MultipleMyeloma-Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/Tiled-MultipleMyeloma_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is enough 
to use for a training set of our segmentation model. <br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We have trained Tiled-MultipleMyelomaTensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyelomaand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Enabled Batch Normalization.<br>
Defined a small <b>base_filters=16</b> and large <b>base_kernels=(9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.0001
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Dataset class</b><br>
Specified ImageMaskDataset class.
<pre>
[dataset]
datasetclass  = "ImageMaskDataset"
resize_interpolation = "cv2.INTER_CUBIC"
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]

epoch_change_infer      = False
epoch_change_infer_dir  = "./epoch_change_infer"
epoch_change_tiledinfer = True
epoch_change_tiledinfer_dir = "./epoch_change_tiledinfer"
num_infer_images       = 6
</pre>

By using this callback, on every epoch_change, the epoch change tiledinfer procedure can be called
 for 6 image in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/epoch_change_tiled_infer_at_start.png" width="1024" height="auto"><br>
<br>
<br>
<b>Epoch_change_inference output at ending (38,39,40)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/epoch_change_tiled_infer_at_end.png" width="1024" height="auto"><br>
<br>
<br>

In this experiment, the training process was stopped at epoch 40 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/train_console_output_at_epoch_40.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Tiled-MultipleMyeloma.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/evaluate_console_output_at_epoch_40.png" width="720" height="auto">
<br><br>Image-Segmentation-Tiled-MultipleMyeloma

<a href="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Tiled-MultipleMyeloma/test was not low, and dice_coef not high as shown below.
<br>
<pre>
loss,0.2784
dice_coef,0.7115
</pre>
<br>

<h3>
5 Tiled inference
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Tiled-MultipleMyeloma.<br>
<pre>
./4.tiled_infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetTiledInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images (2560x1920 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Tiled inferred test masks (2560x1920 pixels)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/asset/mini_test_output_tiled.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 2560x1920 pixels</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Tiled-inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/102.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/102.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/102.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/106.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/106.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/106.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/109.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/109.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/109.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/112.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/112.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/112.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/202.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/202.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/202.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/207.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/207.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/207.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/images/213.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test/masks/213.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Tiled-MultipleMyeloma/mini_test_output_tiled/213.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. IDRiD: Diabetic Retinopathy – Segmentation and Grading Challenge</b><br>
Prasanna Porwal
, 
Samiksha Pachade, Manesh Kokare, Girish Deshmukh, Jaemin Son, Woong Bae, Lihong Liu<br>
, Jianzong Wang, Xinhui Liu, Liangxin Gao, TianBo Wu, Jing Xiao, Fengyan Wang<br>, 
Baocai Yin, Yunzhi Wang, Gopichandh Danala, Linsheng He, Yoon Ho Choi, Yeong Chan Lee<br>
, Sang-Hyuk Jung,Fabrice Mériaudeau<br>
<br>

DOI:<a href="https://doi.org/10.1016/j.media.2019.101561">https://doi.org/10.1016/j.media.2019.101561</a>

<br>
<a href="https://www.sciencedirect.com/science/article/abs/pii/S1361841519301033">
https://www.sciencedirect.com/science/article/abs/pii/S1361841519301033</a>
<br>
<br>

<b>2. Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer</b><br>

Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-Skin-Cancer
</a>
<br>
<br>
<b>3. Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma">
https://github.com/sarah-antillia/Tensorflow-Tiled-Image-Segmentation-Augmented-MultipleMyeloma
</a>
<br>
<br>

<b>4. Tiled-ImageMask-Dataset-Breast-Cancer</b><br>
Toshiyuki Arai @antillia.com<br>
<a href="https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Breast-Cancer">
https://github.com/sarah-antillia/Tiled-ImageMask-Dataset-Breast-Cancer
</a>
<br>
<br>

