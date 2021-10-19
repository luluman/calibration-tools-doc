引言
======

术语解释
--------

====================  ================================================================================================
 **术语**              **说明**
--------------------  ------------------------------------------------------------------------------------------------
 BM168X               比特大陆面向深度学习领域推出的张量处理器
 Umodel               Unified Model，比特大陆自主开发的网络模型格式，用于统一存储各深度学习框架下网络的拓扑结构与系数
 Uframework           Unified Framework，比特大陆自主开发的深度学习框架，集成Caffe、TensorFlow、MxNet、PyTorch、Darknet各框架的运算功能
 Calibration-tools    网络模型的float32系数到int8系数的转换工具
 Quantization-tools   Umodel，Uframework，Calibration-tools的统称
 SOPHON               比特大陆AI品牌，本文特指AI系列产品的运算平台
 float32              32 bit 浮点数
 int8                 8 bit 定点数
 bmnetu               针对BM1684的UFramework模型编译器，可将某网络的umodel和 prototxt编译成BMRuntime所需要的文件
 bmodel               面向比特大陆TPU处理器的深度神经网络模型文件格式
====================  ================================================================================================

授权
----

Quantization-Tools是比特大陆自主研发的具有完全知识产权的深度学习开发工具包，未经比特大陆事先书面授权，其它第三方公司或个人不得以任何形式或方式复制、发布和传播。

版本信息
--------

本次Quantization-Tools发布版本号为 v2.5.0。

版本特性
--------

Quantization-Tools主要特性如下：

1) 支持的深度学习框架

   - 支持对Caffe网络模型进行量化

   - 支持对TensorFlow网络模型进行量化

   - 支持对MxNet网络模型进行量化

   - 支持对PyTorch网络模型进行量化

   - 支持对Darknet网络模型进行量化


2) 通过自定义的网络模型格式Umodel，将Caffe、TensorFlow、MxNet、PyTorch、Darknet的网络模型进行统一存储

3) 通过自定义深度学习框架Uframework，使量化过程与Caffe、TensorFlow、MxNet、PyTorch、Darknet等框架完全分离

帮助与支持
----------

在使用过程中，如有关于Quantization-Tools的任何问题或建议，请发邮件至\ `bmai@bitmain.com <mailto:help@bitmain.com>`__\ 。
