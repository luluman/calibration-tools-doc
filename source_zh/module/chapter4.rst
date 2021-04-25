量化步骤
========

转换Caffe框架下的网络，需要以下5步骤:

- 准备lmdb数据集

- 生成fp32umodel

- 生成int8umodel

- 精度测试（optional）

- int8 umodel的部署

.. _prepare-lmdb:

准备lmdb数据集
--------------

- Quantization-tools对输入数据的格式要求是[N,C,H,W]  (即先存放W数据，再存放H数据，依次类推)
- Quantization-tools对输入数据的C维度的存放顺序与原始框架保持一致。例如caffe框架要求的C维度存放顺序是BGR；tensorflow要求的C维度存放顺序是RGB

.. _convert-lmdb:

将数据转换成lmdb格式
~~~~~~~~~~~~~~~~~~~~

需要将图片转换成lmdb格式的才能进行量化。 将数据转换成lmdb数据集有两种方法：

a) 运用convert_imageset工具

b) 运用u_framework接口，在网络推理过程中将第一层输入抓取存成lmdb

.. _ch4-001:

.. figure:: ../_static/ch4_001.png
   :height: 3.99876in
   :align: center

   两种生成lmdb数据集的操作流程


运用convert_imageset工具
````````````````````````

章节 :ref:`create-lmdb-demo` 作为示例程序，描述了convert_imageset的使用Quantization-tool工具包包括了将图像转换成lmdb的工具：convert_imageset。

convert_imageset的使用方法见如下：

该工具通过命令行方式使用，命令行的格式如下：

  .. code-block:: shell

     $ convert_imageset \
                  ImgSetRootDir/ \     #图像集的根目录
                  ImgFileList.txt\     #图片列表及其标注
                  imgSetFolder   \     #生成的lmdb的文件夹
                  可选参数


- ImgSetRootDir：  图像集的根目录
- ImgFileList.txt：  该文件中记录了图像集中的各图样的相对路径（以ImgSetRootDir为根目录）和相应的标注，假如ImgFileList.txt 的内容为

  ::

     subfolder1/file1.JPEG 7
     subfolder1/file2.JPEG 1
     subfolder1/file3.JPEG 2
     subfolder1/file4.JPEG 6


对于第一行，表示图像文件存放的位置为ImgSetRootDir/subfoler1/file1.JPEG, 它的label为7。

**注意** “subfolder1/file1.JPEG” 与 “7”之间有一个空格。

- imgSetFolder：表示生成的lmdb的文件夹

- 可选参数设置

  .. code-block:: shell

    --gray：bool类型，默认为false，如果设置为true，则代表将图像当做灰度图像来处理，否则当做彩色图像来处理
      例如 --gray=false

    --shuffle：bool类型，默认为false，如果设置为true，则代表将图像集中的图像的顺序随机打乱
      例如 --shuffle=false

    --backend：string类型，可取的值的集合为{"lmdb", "leveldb"}，默认为"lmdb"，代表采用何种形式来存储转换后的数据
      例如 --backend=lmdb

    --resize_width：int32的类型，默认值为0，如果为非0值，则代表图像的宽度将被resize成resize_width
      例如  --resize_width = 200

    --resize_height：int32的类型，默认值为0，如果为非0值，则代表图像的高度将被resize成resize_height
      例如  --resize_height = 200

    --check_size：bool类型，默认值为false，如果该值为true，则在处理数据的时候将检查每一条数据的大小是否相同

    --encoded：bool类型，默认值为false，如果为true，代表将存储编码后的图像，具体采用的编码方式由参数encode_type指定
      例如 --encoded=false

    --encode_type：string类型，默认值为""，用于指定用何种编码方式存储编码后的图像，取值为编码方式的后缀（如'png','jpg',...）

.. _u_framework:

运用u_framework c++接口
`````````````````````

当网络是级联网络，或者网络有特殊的数据预处理而u_framework不支持的，可以考虑使用u_framework提供的接口存储lmdb数据集。

章节 :ref:`mtcnn-demo` 描述了级联网络如何通过该接口来存储lmdb。

此时需要基于u_framework搭建一个网络推理的框架，如图 :ref:`ch4-002` 所示

.. _ch4-002:

.. figure:: ../_static/ch4_002.png
   :width: 5.0in
   :align: center

   通过u_framework接口存储lmdb数据集框架

1) 包含必要头文件

  .. code-block:: c++

     #include <ufw/ufw.hop>
     using namespace ufw;

2) 设置模式

  .. code-block:: c++

     Ufw::set_mode(Ufw::FP32);                  #设置为Ufw::FP32

3) 设置存储的图片数量

  .. code-block:: c++

     max_iterations = 200

4) 建立A_net

  .. code-block:: c++

     A_net_= new Net<float>(proto_file, TEST);   # proto_file描述网络结构的文件
     A_net_-> CopyTrainedLayersFrom(model_file); # model_file描述网络系数的文件
     A_net_-> ExtractFeaturesInit();             # 完成存储lmdb功能模块的初始化

各函数的详细定义见章节“7.2c接口API函数”。

5) 建立B_net

同4)

6) 读入图片，预处理

该步骤与待测的检测网络本身特性有关。可以使用opencv的函数。

7) 给网络填充数据

将经过预处理的图片数据填充给网络。

  .. code-block:: c++

     //根据输入blob的名字（这里是“data”），得到该blob的指针
     Blob<float> *input_blob = (net_-> blob_by_name("data")).get();

     //根据输入图片的信息，对输入blob进行reshape
     input_blob->Reshape(net_b, net_c, net_h, net_w);

     //resized的类型为cv::Mat；其中存储了经过了预处理的数据信息
     // universe_fill_data()函数会将resized中的数据填充给网络的输入blob（这里是input_blob）
     input_blob->universe_fill_data(resized);

8) A_net推理

  .. code-block:: c++

     A_net_->Forward();
     A_net_-> ExtractFeatures();

9) 给B网络填充数据

10) B_net推理


运用u_framework Python接口
``````````````````````````

a) LMDB API组成

   - lmdb = ufw.io.LMDBDataset(path, queuesize=100, mapsize=20e6) # 建立一个LMDBDataset对象

     ::

        path: 建立LMDB的路径(会自建文件夹，并将数据内容存储在文件夹下的data.mdb)
        queue_size:  缓存队列，指缓存图片数据的个数。默认为100，增加该数值会提高读写性能，但是对内存消耗较大
        mapsize:  LMDB建立时开辟的内存空间，LMDBDataset会在内存映射不够的时候自动翻倍


   - put(images, labels=None, keys=None)  # 存储图片和标签信息

     ::

        images: 图片数据，接受numpay.array格式。需要使用CHW格式，如果不符合需要提前transpose一下。数据类型可以是float或是uint8。如果数据维度为3维，则认为是单张图片(batch=1)；如果是4维，认为是多组图片，会按照batch分别存储。
        lables: 图片的lable，需要是int类型，如果没有label不填该值即可。如果设定该值，需要其长度与images的batch一致。
        keys:   LMDB的键值，可以使用原始图片的文件名，但是需要注意LMDB数据会对存储的数据按键值进行排序，推荐使用唯一且递增的键值。如果不填该值，LMDB_Dataset会自动维护一个递增的键值。

   - close()

     ::

        将缓存取内容存储，并关闭数据集。如果不使用该方法，程序会在结束的时候自动执行该方法。
        但是如果python解释器崩溃，则会导致缓存区数据丢失。

b) LMDB API使用方式

   - import ufw
   - txn = ufw.io.LMDB_Dataset('to/your/path')
   - txn.put(images)  # 放置在循环中
   - 在pytorch和tensorflow中，images通常是xxx.Tensor，可以使用images.numpy()，将其转化为numpy.array格式
   - tensorflow的tensor通常是NHWC模式，可以使用transpose([2, 0, 1])[三维数据]，或transpose([0, 3, 1, 2])[四维数据]
   - txn.close()

示例代码

  .. code-block:: python

     import ufw
     import lmdb
     import torch

     images = torch.randn([3, 3,100,100])

     path = 'test__'
     txn = ufw.io.LMDB_Dataset(path)

     for i in range(1020):
         txn.put(images.numpy())
     txn.close()

     ## test LMDB key information
     def lmdbextractinfo(path):
         with lmdb.open(path, readonly=True) as txn:
             cursor = txn.begin().cursor()
             for key, value in cursor:
                 print(key)

d) 注意事项

   - 此功能不会检查给定路径下是否已有文件，如果之前存在LMDB文件，该文件会被覆盖。
   - python解释器崩溃会导致数据丢失。
   - 如果程序正常结束，LDMB_Dataset会自动将缓存区数据写盘。也可以使用close()安全关闭写盘。
   - 使用重复的key会导致数据覆盖或污染，使用非递增的key会导致写入性能下降。
   - 解析该LMDB的时候需要使用Data layer。
   - 输入数据类型支持float、uint8。




.. _using-lmdb:

使用lmdb数据集
~~~~~~~~~~~~~~

为了使用刚生成的lmdb数据集，需要对网络的*.prototxt文件作以下3方面的修改：

- 使用Data layer作为网络的输入。
- 使Data layer的参数data_param指向生成的lmdb数据集的位置。
- 修改Data layer的transform_param参数以对应网络对图片的预处理。

修改data_param指向生成的lmdb数据集
``````````````````````````````````

.. figure:: ../_static/ch4_011.png
   :height: 4.02083in
   :align: center

   修改source指向正确的LMDB位置

数据预处理
``````````
在量化网络前，需要修改网络的prototxt文件，在datalayer（或者AnnotatedData layer）添加其数据预处理的参数，以保证送给net的数据与原始框架的一致。


数据预处理参数
''''''''''''''

数据预处理通过transform_param参数来定义，其各参数的含义如下：

a) TransformationParameter定义

  .. code-block:: c++

     message TransformationParameter {
                  // For data pre-processing, we can do simple scaling and subtracting the
                  // data mean, if provided. Note that the mean subtraction is always carried
                  // out before scaling.
                  optional float scale = 1 [default = 1];
                  // Specify if we want to randomly mirror data.
                  optional bool mirror = 2 [default = false];
                  // Specify if we would like to randomly crop an image.
                  optional uint32 crop_size = 3 [default = 0];
                  // mean_file and mean_value cannot be specified at the same time
                  optional string mean_file = 4;
                  // if specified can be repeated once (would subtract it from all the channels)
                  // or can be repeated the same number of times as channels
                  // (would subtract them from the corresponding channel)
                  repeated float mean_value = 5;
                  // Force the decoded image to have 3 color channels.
                  optional bool force_color = 6 [default = false];
                  // Force the decoded image to have 1 color channels.
                  optional bool force_gray = 7 [default = false];
                  // Resize policy
                  optional ResizeParameter resize_param = 8;
                  // Noise policy
                  optional NoiseParameter noise_param = 9;
                  // Constraint for emitting the annotation after transformation.
                  optional EmitConstraint emit_constraint = 10;
                  optional uint32 crop_h = 11 [default = 0];
                  optional uint32 crop_w = 12 [default = 0];
                  // Distortion policy
                  optional DistortionParameter distort_param = 13;
                  // Expand policy
                  optional ExpansionParameter expand_param = 14;

                  // TensorFlow data pre-processing
                  optional float crop_fraction = 15 [default = 0];
                  // if the number of resize is 1 preserve the original aspect ratio
                  repeated uint32 resize = 16;
                  // less useful
                  optional bool standardization = 17 [default = false];
                  repeated TransformOp transform_op = 18;
                  }


b) ResizeParameter定义

  .. code-block:: c++

     // Message that stores parameters used by data transformer for resize policy
     message ResizeParameter {
     //Probability of using this resize policy
     optional float prob = 1 [default = 1];
     enum Resize_mode {
                  WARP = 1;
                  FIT_SMALL_SIZE = 2;
                  FIT_LARGE_SIZE_AND_PAD = 3;
     }
     optional Resize_mode resize_mode = 2 [default = WARP];
     optional uint32 height = 3 [default = 0];
     optional uint32 width = 4 [default = 0];
     // A parameter used to update bbox in FIT_SMALL_SIZE mode.
     optional uint32 height_scale = 8 [default = 0];
     optional uint32 width_scale = 9 [default = 0];

     enum Pad_mode {
                  CONSTANT = 1;
                  MIRRORED = 2;
                  REPEAT_NEAREST = 3;
    }
     // Padding mode for BE_SMALL_SIZE_AND_PAD mode and object centering
     optional Pad_mode pad_mode = 5 [default = CONSTANT];
     // if specified can be repeated once (would fill all the channels)
     // or can be repeated the same number of times as channels
     // (would use it them to the corresponding channel)
     repeated float pad_value = 6;

     enum Interp_mode { //Same as in OpenCV
                  LINEAR = 1;
                  AREA = 2;
                  NEAREST = 3;
                  CUBIC = 4;
                  LANCZOS4 = 5;
     }
     //interpolation for for resizing
     repeated Interp_mode interp_mode = 7;
     }


Pad_mode： 表示pad时的模式，含义如下

  .. table::
     :widths: 50 50

     ==================   ===============
     pad_MODE 参数         与opencv对应关系
     ------------------   ---------------
     CONSTANT = 1         cv::BORDER_CONSTANT
     MIRRORED = 2         cv::BORDER_REFLECT101
     REPEAT_NEAREST = 3	cv::BORDER_REPLICATE
     ==================   ===============


Resize_mode：表示resieze时候模式，含义如下

+---------------------------+----------------------------------------------------------------------------------------------------+
|Resize_mode 参数           |与opencv对应关系                                                                                    |
+---------------------------+----------------------------------------------------------------------------------------------------+
|WARP = 1                   |cv::resize()                                                                                        |
+---------------------------+----------------------------------------------------------------------------------------------------+
|FIT_SMALL_SIZE = 2         |a) 保持原始图片的长宽比，长宽等比例变化，resize后其中一边与目标长度相同，另一边要比目标长度要大     |
|                           |                                                                                                    |
|                           |b) if :math:`\frac{img_W}{img_H} > \frac{new_W}{new_H}`                                             |
|                           |  - 则resize后的H要比new_H要大                                                                      |
|                           |  - resize后的 :math:`[W,H] = [new_W, new_W * \frac{img_H}{img_W}]`                                 |
|                           |                                                                                                    |
|                           |c) if :math:`\frac{img_W}{img_H} < \frac{new_W}{new_H}`                                             |
|                           |  - 则resize后的W要比new_W要大                                                                      |
|                           |  - resize后的 :math:`[\frac{img_W}{img_H}*new_H,new_H]`                                            |
+---------------------------+----------------------------------------------------------------------------------------------------+
|FIT_LARGE_SIZE_AND_PAD = 3 |a) 保持原始图片的长宽比,resize后其中一边与目标长度相同，另一边比目标长度要小，该边通过pad的方式达到 |
|                           |  目标的长度一样                                                                                    |
|                           |                                                                                                    |
|                           |b) if :math:`\frac{img_W}{img_H} > \frac{new_W}{new_H}`                                             |
|                           |  - 则需要在H方向填充数据才能与目标的长宽比一致                                                     |
|                           |  - 同比例压缩img_H,img_W,使压缩后的图片 :math:`[W', H']=[new_W, \frac{new_W}{img_W}*img_H]`        |
|                           |  - H方向的上下分别填充 :math:`\frac{new_H-H'}{2}` 个数                                             |
|                           |                                                                                                    |
|                           |c) if :math:`\frac{img_W}{img_H} < {new_W}{new_H}`                                                  |
|                           |  - 则需要在w方向填充数据才能与目标的长宽比一致                                                     |
|                           |  - 同比例压缩img_H，img_W，使压缩后的图片 :math:`[W', H']=[\frac{img_W}{img_H}*new_H,new_H]`       |
|                           |  - W方向的上下分别填充 :math:`\frac{new_W-W'}{2}` 个数                                             |
+---------------------------+----------------------------------------------------------------------------------------------------+


Interp_mode：表示插值时候的模式，含义如下：

  .. table::
     :widths: 50 50

     =================  ==================
     Interp_mode 参数    	与opencv对应关系
     -----------------  ------------------
     LINEAR = 1         cv::INTER_LINEAR
     AREA = 2           cv::INTER_AREA
     NEAREST = 3        cv::INTER_NEAREST
     CUBIC = 4	      cv::INTER_CUBIC
     LANCZOS4 = 5	      cv::INTER_LANCZOS4
     =================  ==================

C) TransformOp

  .. code-block:: c++

     //for tensorflow
     message TransformOp {
     enum Op {
                  RESIZE = 0;
                  CROP = 1;
                  STAND = 2;
                  NONE = 3;
     }
     // For historical reasons, the default normalization for
     // SigmoidCrossEntropyLoss is BATCH_SIZE and *not* VALID.
     optional Op op = 1 [default = NONE];
     //resize parameters
     optional uint32 resize_side = 2 ;
     optional uint32 resize_h = 3 [default = 0];
     optional uint32 resize_w = 4 [default = 0];

     //crop parameters
     optional float  crop_fraction = 5;
     optional uint32 crop_h = 6 [default = 0];
     optional uint32 crop_w = 7 [default = 0];
     optional float  padding = 8 [default = 0];//for resize_with_crop_or_pad

     //mean substraction(stand)
     repeated float mean_value = 9;
     optional string mean_file = 10;
     optional float scale = 11 [default = 1];
     optional float div = 12 [default = 1];
     optional bool   bgr2rgb = 13 [default = false];
     }


当lmdb内的数据是bgr格式的，但是net需要输入为rgb格式时，将bgr2rgb设置为ture。


数据预处理参数的作用流程
''''''''''''''''''''''''

基于以上的TransformationParameter的参数定义，其作用的流程如图 :ref:`ch4-003` 所示。其特点如下：

- 在编译prototxt文件时，transform_op中定义的参数与transform_op外定义的参数只能二选一，如图 :ref:`ch4-004` 所示，左边是包括transform_op参数的例子，右边是不包括transform_op参数的例子。
- transform_op中定义的参数按其在prototxt定义的顺序来执行，适用于灵活的数据预处理组合。
- transform_op外定义的参数其执行顺序是固定的，如图 :ref:`ch4-003` 右半部所示。

.. _ch4-003:

.. figure:: ../_static/ch4_003.png
   :width: 5.76806in
   :height: 4.67015in
   :align: center

   输入预处理的流程

.. _ch4-004:

.. figure:: ../_static/ch4_004.png
   :width: 5.76806in
   :height: 3.00403in
   :align: center

   是否包含transform_op参数对比

对于带Annotated信息的lmdb的处理
```````````````````````````````

对于检测网络来说，其label不仅仅是个数字，它包括类别，检测框的位置等复杂信息。对于这种情况，分两种情况处理：

- 如果lmdb数据尚未生成，请参照章节 :ref:`convert-lmdb` 、:ref:`using-lmdb` 描述的方法，生成lmdb数据集。生成lmdb时，其label随机填充<200的数字即可；读取lmdb时，用“Data layer”来读取该数据lmdb数据集（在量化网络时，那些anntoated信息（包括类别，检测框）不是必须的信息）如图 :ref:`ch4-005` 是fddb数据集基于章节 :ref:`convert-lmdb` 、 :ref:`using-lmdb` 描述的方法生成lmd后，用data layer读取的例子。

.. _ch4-005:

.. figure:: ../_static/ch4_005.png
   :width: 5.76806in
   :height: 2.5357in
   :align: center

   使用Data layer读取fddb数据集

- 如果已经有现成的带anntoated信息的lmdb数据集了，用AnnotatedData layer来读取该lmdb数据集

.. _fig-ch4-006:

.. figure:: ../_static/ch4_006.png
   :width: 5.76806in
   :height: 3.92733in
   :align: center

   使用AnnotatedData Layer来读取该lmdb数据集

生成fp32umodel
--------------

将第三方框架生成的模型文件转换成umodel文件，本阶段生成一个\*.fp32umodel文件以及
一个\*.prototxt文件。

**注意** ：基于精度方面考虑输入Calibration-tools的fp32umodel需要保持Batchnorm层以及
Scale层独立。有时候客户可能会利用第三方工具对网络图做一些等价转换，这个过程中请
确保Batchnorm层以及Scale层不被提前融合到Convolution。

在使用以下转化工具时，需要注意：

a)  如果指定了“-D (-dataset )”参数，那么需要保证
    “-D”参数下的路径正确，同时指定的数据集兼容该网络，否则会有运行错误。

b) 在不能提供合法的数据源时，不应该使用“-D”参数（该参数是可选项，不指定会使用随
   机数据测试网络转化的正确性，可以在转化后的网络中再手动修改数据来源）。

c) 转化模型的时候可以指定参数“--cmp”，使用该参数会比较模型转化的中间格式与原始框
   架下的模型计算结果是否一致，增加了模型转化的正确性验证。


caffe框架下的网络模型生成fp32umodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
本步骤分2步来完成：

- 按照章节 :ref:`using-lmdb` 方法修改prototxt。

  - 使用data layer作为输入
  - 正确设置数据预处理
  - 正确设置lmdb的路径

- 用 \*.caffemodel，\*.prototxt文件作为输入，调用python脚本，完成转换。

python脚本调用
``````````````

a) 参数修改

   以/examples/calibration/examples/caffemodel_to_fp32umodel_demo/
   resnet50_to_umodel.py 为基础，修改其中的-m –w -s 参数：

  .. code-block:: python
     :linenos:
     :emphasize-lines: 4,5,6

     import ufw.tools as tools

     cf_resnet50 = [
         '-m', './models/ResNet-50-test.prototxt',
         '-w', './models/ResNet-50-model.caffemodel',
         '-s', '(1,3,224,224)',
         '-d', 'compilation',
         '--cmp'
     ]

     if __name__ == '__main__':
         tools.cf_to_umodel(cf_resnet50)


  ::

     参数解释
     -m    #指向*.prototxt文件的路径
     -w    #指向*.caffemodel文件的路径
     -s    #输入blob的维度，（N,C,H,W）
     -d    #输出文件夹的名字
     --cmp #可选参数，指定是否测试模型转化的中间文件


b) 运行命令：

  ::

     例如：python3 resnet50_to_umodel.py


c) 输出：

   在当前文件夹下，新生成compilation文件夹，存放新生成的\*.fp32umodel 与 \*.prototxt。

tensorflow框架下的网络模型生成fp32umodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
本步骤分2步来完成：

- 用\*.pb文件作为输入，调用python脚本，完成转换。

- 按照章节 :ref:`using-lmdb` 方法修改prototxt。

  - 使用data layer作为输入
  - 正确设置数据预处理
  - 正确设置lmdb的路径


python脚本调用
``````````````
a) 参数修改

   以/examples/calibration/examples/ tf_to_fp32umodel_demo/
   resnet50_v2_to_umodel.py为基础，修改其中的–m，-i，-s等 参数：

   .. code-block:: python
      :linenos:
      :emphasize-lines: 4,5,7

      import ufw.tools as tools

      tf_resnet50 = [
          '-m', './models/frozen_resnet_v2_50.pb',
          '-i', 'input',
          '-o', 'resnet_v2_50/predictions/Softmax',
          '-s', '(1, 299, 299, 3)',
          '-d', 'compilation',
          '-n', 'resnet50_v2',
          '-p', 'INCEPTION',
          '-D', '../classify_demo/lmdb/imagenet_s/ilsvrc12_val_lmdb',
          '-a',
          '--cmp'
      ]

      if __name__ == '__main__':
          tools.tf_to_umodel(tf_resnet50)


   ::

      参数解释
      -m    #指向*.pb文件的路径
      -i    #输入tensor的名称
      -o    #输出tensor的名称
      -s    #输入tensor的维度，（N,H,W,C）
      -d    #输出文件夹的名字
      -n    #网络的名字
      -p    #数据预处理类型，预先定义了VGG，INCEPTION，SSD_V，SSD_I几种。
            #没有合适的随意选一个，然后在手动编辑prototxt文件的时候，根据实际的预处理来添加
      -D    #lmdb数据集的位置，
            #没有的话，可以暂时随意填个路径，然后在手动编辑prototxt文件的时候，根据实际的路径来添加
      -a    #加上该参数，会在生成的模型中添加top1，top5两个accuracy层
      --cmp #可选参数，指定是否测试模型转化的中间文件

b) 运行命令：

  ::

     例如：python3 resnet50_v2_to_umodel.py

c) 输出：

   在当前文件夹下，新生成compilation文件夹，存放新生成的\*.fp32umodel 与\*.prototxt。


.. _pytorch-to-umodel:

pytorch框架下的网络模型生成fp32umodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

a) 参数修改

   以/examples/calibration/examples/pt_to_fp32umodel_demo/ mobilenet_v2_to_umodel.py为基础，修改其中的–m，-s等 参数。

  .. code-block:: python
     :linenos:
     :emphasize-lines: 4,5

     import ufw.tools as tools

     pt_mobilenet = [
         '-m', './models/mobilenet_v2.pt',
         '-s', '(1,3,224,224)',
         '-d', 'compilation',
         '-p', 'INCEPTION',
         '-D', '../classify_demo/lmdb/imagenet_s/ilsvrc12_val_lmdb',
         '-a',
         '--cmp'
     ]

     if __name__ == '__main__':
         tools.pt_to_umodel(pt_mobilenet)


  ::

     参数解释
     -m    #指向*.pb文件的路径
     -s    #输入tensor的维度，（N,C,H,W）
     -p    #数据预处理类型，预先定义了VGG，INCEPTION，SSD_V，SSD_I几种。
           #没有合适的随意选一个，然后在手动编辑prototxt文件的时候，根据实际的预处理来添加
     -D    #lmdb数据集的位置，
           #没有的话，可以暂时随意填个路径，然后在手动编辑prototxt文件的时候，根据实际的路径来添加
     -a    #加上该参数，会在生成的模型中添加top1，top5两个accuracy层
     --cmp #可选参数，指定是否测试模型转化的中间文件


b) 运行命令：

  ::

     例如：python3 mobilenet_v2_to_umodel.py

c) 输出：

   在当前文件夹下，新生成compilation文件夹，存放新生成的 \*.fp32umodel 与 \*.prototxt


.. _mxnet-to-umodel:

mxnet框架下的网络模型生成fp32umodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

a) 参数修改

   以
   /examples/calibration/examples/mx_to_fp32umodel_demo/mobilenet0.25_to_umodel.py
   为基础，修改其中的–m，-w，-s等 参数：


   .. code-block:: python
      :linenos:
      :emphasize-lines: 4,5,6

      import ufw.tools as tools

      mx_mobilenet = [
          '-m', './models/mobilenet0.25-symbol.json',
          '-w', './models/mobilenet0.25-0000.params',
          '-s', '(1,3,128,128)',
          '-d', 'compilation',
          '-p', 'INCEPTION',
          '-D', '../classify_demo/lmdb/imagenet_s/ilsvrc12_val_lmdb',
          '-a',
          '--cmp'
      ]

      if __name__ == '__main__':
          tools.mx_to_umodel(mx_mobilenet)


   ::

      参数解释
      -m    #指向*.json文件的路径
      -w    #指向*params文件的路径
      -s    #输入tensor的维度，（N,C,H,W）
      -p    #数据预处理类型，预先定义了VGG，INCEPTION，SSD_V，SSD_I几种。
            #没有合适的随意选一个，然后在手动编辑prototxt文件的时候，根据实际的预处理来添加
      -D    #lmdb数据集的位置，
            #没有的话，可以暂时随意填个路径，然后在手动编辑prototxt文件的时候，根据实际的路径来添加
      -a    #加上该参数，会在生成的模型中添加top1，top5两个accuracy层
      --cmp #可选参数，指定是否测试模型转化的中间文件


b) 运行命令：

  ::

     例如：python3 mobilenet0.25_to_umodel.py

c) 输出：

   在当前文件夹下，新生成compilation文件夹，存放新生成的 \*.fp32umodel 与 \*.prototxt。


.. _darknet-to-umodel:

darknet框架下的网络模型生成fp32umodel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

a) 参数修改

   以/examples/calibration/examples/dn_to_fp32umodel_demo/yolov3_to_umodel.py为
   基础，修改其中的–m，-w，-s等 参数：


   .. code-block:: python
      :linenos:
      :emphasize-lines: 4,5,6

      import ufw.tools as tools

      dn_darknet = [
          '-m', 'yolov3/yolov3.cfg',
          '-w', 'yolov3/yolov3.weights',
          '-s', '[[1,3,416,416]]',
          '-d', 'compilation',
          '-cmp'
      ]

      if __name__ == '__main__':
          tools.dn_to_umodel(dn_darknet)


   ::

      参数解释
      -m    #指向*.cfg文件的路径
      -w    #指向*.weights文件的路径
      -s    #输入tensor的维度，（N,C,H,W）
      -d    #生成umodel的文件夹
      -D    #lmdb数据集的位置，
            #没有的话，可以暂时随意填个路径，然后在手动编辑prototxt文件的时候，根据实际的路径来添加
      --cmp #可选参数，指定是否测试模型转化的中间文件


b) 运行命令：

  .. code-block:: bash

     get_model.sh # download model
     python3 yolov3_to_umodel.py


c) 输出：

   在当前文件夹下，新生成compilation文件夹，存放新生成的 \*.fp32umodel 与 \*.prototxt。


量化，生成int8umodel
--------------------

网络量化过程包含下面两个步骤：

- 对输入浮点网络图进行优化。

- 对浮点网络进行量化得到int8网络图及系数文件。

优化网络
~~~~~~~~~

运行命令
````````

  .. code-block:: shell

     $ cd <release dir>
     $ calibration_use_pb  \
                  graph_transform \                   #固定参数
                  -model= PATH_TO/*.prototxt \        #描述网络结构的文件
                  -weights=PATH_TO/*.fp32umodel       #网络系数文件

默认配置下对输入浮点网络进行优化，包括：batchnorm与scale合并，前处理融合到网络，
删除推理过程中不必要的算子等功能。更多对浮点网络图进行优化的选项参见后面
:ref:`quantize_skill` 章节。

命令输入输出
``````````````

Quantization-tools进行网络图优化的输入参数包括3部分：

- graph_transform： 固定参数

- -model= PATH_TO/\*.prototxt：描述网络结构的文件，该prototxt文件的datalayer指向准备好的数据集，如图 4所示。

- -weights=PATH_TO/\*.fp32umodel：保存网络系数的文件。

Quantization-tools进行网络图优化的输出包括2部分：

- PATH_TO/\*.prototxt_optimized
- PATH_TO/\*.fp32umodel_optimized

为了和原始网络模型做区分，新生成的网络模型存储的时候以“optimized”为后缀。以上两
个个文件存放在与通过参数“-weights=PATH_TO/\*.fp32umodel”指定的文件相同的路径下。

graph_transform功能单独列出来是因为在网络量化调优的时候需要对网络进行多次量化，这
时候不需要多次执行网络图优化。可以在网络量化之前先单独用此命令对网络进行处理。

量化网络
~~~~~~~~~

运行命令
````````

  .. code-block:: shell

     $ cd <release dir>
     $ calibration_use_pb  \
                  graph_transform \                   #固定参数
                  -model= PATH_TO/*.prototxt \        #描述网络结构的文件
                  -weights=PATH_TO/*.fp32umodel       #网络系数文件
                  -iterations=200 \                   #迭代次数
                  -winograd=false   \                 #可选参数
                  -graph_transform=false \            #可选参数
                  -save_model=true \                  #可选参数
                  -save_test_proto=false              #可选参数

这里给出了量化网络用到的所有必要参数及部分最常用的可选参数，更多网络量化相关的参
数选项参见后面 :ref:`quantize_skill` 章节。


命令输入输出
``````````````

Quantization-tools进行网络量化的常用输入参数包括6部分：

- graph_transform： 固定参数

- -model= PATH_TO/\*.prototxt：描述网络结构的文件，该prototxt文件的datalayer指向
  准备好的数据集，如图 4所示

- -weights=PATH_TO/\*.fp32umodel：保存网络系数的文件，

- -iteration=200：该参数描述了在定点化的时候需要统计多少张图片的信息，默认200

- -winograd：可选参数，针对3x3 convolution开启winograd功能，默认值为False

- -graph_transform:可选参数，开启网络图优化功能，本参数相当于在量化前先执行上面的graph_transform
  命令，默认值为False

- -save_model:可选参数，存储量化后的系数到int8umodel文件，默认值为True

- -save_test_proto:可选参数，存储测试用的prototxt文件，默认值False


Quantization-tools的输出包括5部分：

- \*.int8umodel:  即量化生成的int8格式的网络系数文件
- \*_test_ fp32_unique_top.prototxt：
- \*_test_ int8_unique_top.prototxt：
  分别为fp32, int8格式的网络结构文件， 该文件包括datalayer
  与原始prototxt文件的差别在于，各layer的输出blob是唯一的，不存在in-place的情况
- \*_ deploy_fp32_unique_top.prototxt：
- \*_ deploy_int8_unique_top.prototxt：分别为fp32，int8格式的网络结构文件,该文件不包括datalayer

以上几个文件存放位置与通过参数“-weights=PATH_TO/\*.fp32umodel”指定的文件位置相同。


精度测试（optional）
--------------------
精度测试是一个可选的操作步骤，用以验证经过int8量化后，网络的精度情况。该步骤可以安排在章节4.5描述的部署之前

量化误差定性分析
~~~~~~~~~~~~~~~~

章节 :ref:`view-demo` 作为示例程序，描述了如何使用calibration可视化分析工具查看网络量化误差。

  .. code-block:: python

     import analysis
     args_ =  [   '-fm', 'path/to/fp32/prototxt',   # float网络模型
                  '-fw',  'path/to/fp32umodel',     # float网络参数
                  '-im', 'path/to/int8/prototxt',   # int8网络模型
                  '-iw',  'path/to/int8umodel']     # int8网络参数
     test_n = analysis.calibration_visual(args_)
     test_n.show_widgets()


该工具使用MAPE（Mean Abusolute Percentage Error）作为误差评价标准，其计算定义为：

  .. math::

     \text{MAPE} = \frac{1}{n}\left( \sum_{i=1}^n \frac{|Actual_i - Forecast_i|}{|Actual_i|} \right)*100


由于int8网络部分层进行了合并计算，例如会将relu与batchnorm合并，所以此时bathcnorm层的MAPE值无效。


分类网络的精度测试
~~~~~~~~~~~~~~~~~~

章节 :ref:`classify-demo` 作为示例程序，描述了分类网络精度测试的方法。

测试原始float32网络的精度
`````````````````````````

  .. code-block:: shell

     $ cd <release dir>
     $ ufw test_fp32 \                                         #固定参数
            -model=PATH_TO/\*_test_fp32_unique_top.prototxt \  #章节4.3.3 输出的文件
            -weights= PATH_TO/\*.fp32umodel \                  #fp32格式的umodel
            -iterations=200                                    #测试的图片个数

测试转换后的int8网络的精度
``````````````````````````

  .. code-block:: shell

     $ cd <release dir>
     $ ufw test_int8 \                                         #固定参数
            -model=PATH_TO/\*test_int8_unique_top.prototxt \   #章节4.3.3 输出的文件
            -weights= PATH_TO/\*.int8umodel \                  #章节4.3.3 输出的文件，量化后int8umodel
            -iterations=200                                    #测试的图片个数

检测网络的精度测试
~~~~~~~~~~~~~~~~~~
本工具提供接口函数供外部程序调用，以方便精度测试程序搜集到网络推理结果，进而得到
最后的精度。本工具提供c、python两种接口形式，供用户调用。完整的c、python接口，见
章节附录 :ref:`c-api` 、:ref:`python-api`。


c接口形式
`````````
章节 :ref:`face-demo` 作为示例程序，描述了C接口的调用方法。 本节是对章节 :ref:`face-demo` 抽象总结。

一个c接口的精度测试程序的框架如图 :ref:`ch4-009`

.. _ch4-009:

.. figure:: ../_static/ch4_009.png
   :height: 9in
   :align: center

   c接口形式精度测试框架

1) 包含必要头文件

   .. code-block:: c++

      #include <ufw/ufw.hpp>
      using namespace ufw;

2) 设置模式

  .. code-block:: c++

      #ifdef INT8_MODE
      Ufw::set_mode(Ufw::INT8);    //运行int8网络的时候，设置为Ufw::INT8
      #else
      Ufw::set_mode(Ufw::FP32);    //运行fp32网络的时候，设置为Ufw::FP32
      #endif


3) 指定网络模型文件

- 运行fp32网络时候，用

  .. code-block:: c++

      String  model_file = **.fp32umodel；
      String  proto_file= **_ deploy_fp32_unique_top.prototxt


- 运行int8网络时候，用

  .. code-block:: c++

     String model_file = **.int8umodel；
     String  proto_file= **_ deploy_int8_unique_top.prototxt


4) 建立网络

  .. code-block:: c++

     net_= new Net<float>(proto_file, TEST);   //proto_file描述网络结构的文件
     net_-> CopyTrainedLayersFrom(model_file); //model_file描述网络系数的文件


5) 读入图片，预处理

   该步骤与待测的检测网络本身特性有关。采用原始网络的处理代码即可。

6) 给网络填充数据

   将经过预处理的图片数据填充给网络：

  .. code-block:: c++

     //根据输入blob的名字（这里是“data”），得到该blob的指针
     Blob<float> *input_blob = (net_-> blob_by_name("data")).get();

     //根据输入图片的信息，对输入blob进行reshape
     input_blob->Reshape(net_b, net_c, net_h, net_w);

     //resized的类型为cv::Mat；其中存储了经过了预处理的数据信息
     // universe_fill_data()函数会将resized中的数据填充给网络的输入blob（这里是input_blob）
     input_blob->universe_fill_data(resized);


7) 网络推理

  .. code-block:: c++

     net_->Forward();


8) 	搜集网络推理结果

- 通过这种方法得到的是网络输出数据的指针，例如const float* m3_scores

  .. code-block:: c++

     //根据输出blob的名字（这里是m3@ssh_cls_prob_reshape_output），net_->blob_by_name得到该blob的指针
     Blob<float>* m3_cls_tensor =
                  net_->blob_by_name("m3@ssh_cls_prob_reshape_output").get();

     // universe_get_data()函数返回float *类型的指针，该指针指向该blob内的数据
     const float* m3_scores = m3_cls_tensor->universe_get_data();


- 网络输出blob的名字，可以通过查看**_ deploy_fp32_unique_top.prototxt文件得到

9) 对推理结果的后处理

   该步骤与待测的检测网络本身特性有关。采用原始网络的处理代码即可。

python接口形式
``````````````

章节 :ref:`object-detection-python-demo` 作为示例程序，描述了python接口的调用方法。 本节是对章节 :ref:`object-detection-python-demo` 抽象总结。

一个python接口的精度测试程序的框架如图 :ref:`ch4-010`

.. _ch4-010:

.. figure:: ../_static/ch4_010.png
   :width: 5.76806in
   :height: 8.28976in
   :align: center

   python接口形式精度测试框架

1) 载入ufw

  .. code-block:: python

     import ufw

2) 设置模式

- fp32模式时：

  .. code-block:: python

     ufw.set_mode_cpu()


- int8模式时：

  .. code-block:: python

     ufw.set_mode_cpu_int8()


3) 指定网络模型文件

- 运行fp32网络时候，用

  .. code-block:: python

     model = './models/ssd_vgg300/ssd_vgg300_deploy_fp32.prototxt'
     weight = './models/ssd_vgg300/ssd_vgg300.fp32umodel'


- 运行int8网络时候，用

  .. code-block:: python

     model = './models/ssd_vgg300/ssd_vgg300_deploy_int8.prototxt'
     weight = './models/ssd_vgg300/ssd_vgg300.int8umodel'


4) 建立网络

  .. code-block:: python

     ssd_net = ufw.Net(model, weight, ufw.TEST)


5) 读入图片，预处理

   该步骤与待测的检测网络本身特性有关。采用原始网络的处理代码即可

6) 给网络填充数据

   将经过预处理的图片数据填充给网络

  .. code-block:: python

     ssd_net.fill_blob_data({blob_name: input_data})


7) 网络推理

  .. code-block:: python

     ssd_net.forward()


8) 搜集网络推理结果

  .. code-block:: python

     ssd_net.get_blob_data(blob_name)


9) 对推理结果的后处理

   该步骤与待测的检测网络本身特性有关。采用原始网络的处理代码即可。


部署
----
部署指的是用int8umodel，生成SOPHON系列AI平台指令集。网络部署时，涉及到以下两个文
件：

  ::

     **.int8umodel,
     **_deploy_ int8_unique_top.prototxt


以上两个文件会送给bmnetu，最终生成可在SOPHON系列AI运算平台上运行的bmodel，具体步骤请参考文档bmnetu的相关文档。


级联网络的量化步骤
------------------

- 生成lmdb

  用章节 :ref:`u_framework` 所述方法，搭建推理环境，调用u_framework接口，一次完成所有网络的lmdb存储。

- 量化，生成int8umodel

  每个网络单独量化，用章节“4.3量化，生成int8umodel”所述方法生成各自的int8umodel。

- 精度测试

  与单个网络的精度测试方法相同。

- 部署

  每个网络单独生成bmodel
