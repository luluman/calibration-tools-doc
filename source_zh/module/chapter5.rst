.. _quantize_skill:

量化技巧
========

结合1684芯片的特点这里给出了3类优化网络量化效果的技巧：

- 阈值计算

- 混合执行

- 网络图优化

阈值计算
-----------------

分布统计
~~~~~~~~~~~~~~~~~~~~
calibration_use_pb命令行添加如下参数：

- -dump_dist: 指向一个输出文件的路径。将网络各层统计的最大值及feature的分布信息保存到文件。

- -load_dist: 指向一个-dump_dist参数生成的文件的路径。从-dump_dist参数生成的文件
  中读取网络各个层的最大值及feature分布信息。

针对有些网络需要量化调优进行多次量化的场景，采用此参数可以只统计一次分布信息并多
次使用。可以大大加快量化调优的速度。

    注：选择使用ADMM方法不支持保存和加载feature分布。

阈值调整
~~~~~~~~~~~~~~~~~~~~

阈值的选取对于网络量化效果有很大的影响，这里给出两种方式来对量化阈值进行调整。

针对所有的层进行调整
````````````````````````
calibration_use_pb命令行添加如下参数：

- -th_method:可选参数，指定计算各个层量化阈值的方法,可选参数：KL，SYMKL，JSD，ADMM，ACIQ以及MAX，默认值为KL。


针对具体的层进行调整
````````````````````````

为了更精细地对某个具体层的阈值进行调整，在layer_parameter中增加了部分参数，可以通
过下面方式在prototxt文件中进行使用。

.. _ch5-001:

  .. figure:: ../_static/ch5_001.png
     :width: 5.76806in
     :height: 3.36583in
     :align: center

     Prototxt文件中设置采用最大值作为阈值


- th_strategy:设置当前layer的量化阈值计算策略。

              可选参数：USE_DEFAULT，USE_KL，USE_MAX，默认值为USE_DEFAULT。

              USE_DEFAULT:采用calibration-tools量化程序内部定义的规则来计算当前
              layer的阈值。

              USE_KL:采用求KL分布的方式来计算当前layer的阈值，这里的KL是相对MAX
              而言，具体的阈值计算策略可能是KL，SYMKL或者JSD。

              USE_MAX:采用统计的最大值做为当前layer的最大值。

- th_scale:在计算得到的阈值上面乘以缩放系数th_scale，默认值为1.0。

- threshold：设置当前layer的量化阈值，无默认值。


混合执行
------------------

1684芯片内部集成了浮点计算单元，可以高效地利用浮点进行计算。根据芯片的这个特点，
这里提供了一种混合执行的方式来运行网络，允许部分层用定点进行计算，部分层用浮点进
行计算。通过允许部分层用浮点进行计算，可以有效地提高网络的整体量化精度。

复杂网络的前处理/ 后处理
~~~~~~~~~~~~~~~~~~~~~~~~

Tensorflow及Pytorch等基于python的框架灵活度较大，从这些框架转过来的网络模型中可
能包含前后处理相关的算子。对于这些算子做量化将很大程度上影响网络的量化精度。这里
提供了一种方式在网络中标记出前处理、后处理相关的层，并允许这些层以浮点运行。在calibration_use_pb命令行中使用如下参数：

- -fpfwd_inputs:用逗号分隔开的网络layer name，在网络中这些layer及它们之前的layer
  被标记为网络前处理。网络前处理不算做正式网络的一部分，在calibration过程中不被
  量化，在推理过程中保持使用浮点进行计算。

- -fpfwd_outputs:用逗号分隔开的网络layer name，在网络中这些layer及它们之后的
  layer被标记为网络后处理。网络后处理不算做正式网络的一部分，在calibration过程中
  不被量化，在推理过程中保持使用浮点进行计算。


对量化损失敏感的层
~~~~~~~~~~~~~~~~~~~~


通过命令行参数指定
````````````````````````
在calibration_use_pb命令行中使用如下参数：

- -fpfwd_blocks:用逗号分隔开的网络layer name，在网络中每个layer及它们之后直到下一
  个进行数据计算的层在calibration过程中都不被量化，在推理过程中保持使用浮点进行计算。

calibration-tools程序会根据指定的layer name自动判断这个layer后面有多少个layer需
要用浮点进行计算，把网络的这个block做为一个整体用浮点进行计算，来达到提高量化精
度的目的。如下图所示在命令行中用-fpfwd_blocks指定Softmax层的layer name，
calibration-tools程序会将图中红色框中的layer都标识为用浮点进行计算。
calibration-tools程序会在此block的输入处自动将输入数据转换成浮点格式，在输出位置
转换为int8数据格式。

.. _ch5-002:

  .. figure:: ../_static/ch5_002.jpg
   :height: 3.99876in
   :align: center

   通过命令行设置将对精度敏感的layer block用浮点执行

通过配置prototxt指定
````````````````````````

- forward_with_float:将当前layer用浮点进行计算，可选参数为True，False，默认值为False。

具体使用方法参考如下面图 :ref:`prototxt文件中设置forward_with_float <ch5-003>` 所示，这里的*_test_fp32.prototxt文件是指
calibration_use_pb命令的输入prototxt文件，见 :ref: `grenerate_fp32umodel` 。

.. _ch5-003:

  .. figure:: ../_static/ch5_003.jpg
     :height: 3.99876in
     :align: center

     prototxt文件中设置forward_with_float


网络图优化
------------------

本节所描述的所有量化参数都属于网络图优化部分内容，所有的操作都可以用
graph_transform命令实现。

精度相关优化
~~~~~~~~~~~~~~~~~~~~
在calibration_use_pb命令行中使用如下参数：

- -accuracy_opt:将网络中depthwise卷积采用浮点推理以提高精度，默认为false，关闭。

- -conv_group:将conv的输出channel按照输出幅值进行分组并拆分成不同的组分别进行量化，默认为false，关闭。

- -per_channel:开启convolution计算的per_channel功能，默认为false，关闭。

速度相关优化
~~~~~~~~~~~~~~~~~~~~
在calibration_use_pb命令行中使用如下参数：

- -fuse_preprocess:将前处理里面的线性计算部分融合到网络中，默认为false，关闭。
  开启前处理融合功能后，图 :ref:`ch4-004` 中的mean_value以及scale参数会被合并到网络
  的第一个Convolution的weight以及bias参数中。
