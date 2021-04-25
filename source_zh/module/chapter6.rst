常见异常
========

没有正确设置lmdb数据集的路径
----------------------------

.. figure:: ../_static/ch6_001.png
   :width: 5.76806in
   :height: 0.53611in
   :align: center

出现以上信息，说明没有正确设置lmdb数据集的路径，请按照章节 :ref:`prepare-lmdb` 的方法设置lmdb数据集的路径。

有不支持的层
------------

出现以上信息，说明当前正在转换的网络中有Quantization-tools不支持的层，请联系bitmain技术人员，予以解决。

.. figure:: ../_static/ch6_002.png
   :width: 5.76806in
   :height: 0.19674in
   :align: center

当前Quantization-tools支持的layer见本章附录 :ref:`supported-layer` 。

整个网络输入为0
---------------

.. figure:: ../_static/ch6_003.png
   :width: 5.76806in
   :height: 1.18754in
   :align: center

出现以上信息，说明某个层的输出最大值为0，有可能是整个网络输入的问题，请检查是否
正确配置了网络的输入。
