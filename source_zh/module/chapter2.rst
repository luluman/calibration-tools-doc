Quantization-Tools简介
======================

Qantization-Tools是比特大陆自主开发的网络模型量化工具，它解析各种不同框架已训练
好的32bit浮点网络模型，生成8bit的定点网络模型。该8bit定点网络模型，可用于比特大
陆SOPHON系列AI运算平台。在SOPHON运算平台上，网络各层输入、输出、系数都用8bit来表
示，从而在保证网络精度的基础上，大幅减少功耗，内存，传输延迟，大幅提高运算速度。
Quantization-Tools的组成见下图:

.. _ch2-001:

.. figure:: ../_static/ch2_001.png
   :width: 5.76806in
   :height: 2.59056in
   :align: center

   Quantization-Tools结构图

Quantization-Tools由三部分组成：Parse-Tools、Calibration-Tools以及Uframwork。

- Parse-Tools：

  解析各深度学习框架下已训练好的网络模型，生成统一格式的网络模型文件—umodel，
  支持的深度学习框架包括： caffe、tensorflow、 pytorch、 mxnet以及darknet。

- Calibration-Tools：

  分析float32格式的umodel文件，基于熵损失最小原则，将网络系数定点化成8bit，最后
  将网络模型保存成int8格式的umodel文件。

- Uframework：

  自定义的深度学习推理框架，集合了各开源深度学习框架的运算功能，提供的功能包括：

  a) 作为基础运算平台，为定点化时提供基础运算。

  b) 作为验证平台，可以验证fp32，int8格式的网络模型的精度。

  c) 作为接口，通过bmnetu，可以将int8umodel编译成能在SOPHON运算平台上运行的bmodel。
