安装与配置
==============

Quantization-Tools作为一套工具包，随BmnnSDK一起发布，在Bmnnsdk2-bm1684 Docker镜像中使用。详细安装过程参见BmnnSDK的安装手册。


安装
----

安装需求
~~~~~~~~

1) 硬件环境：X86主机或服务器

#) 操作系统：Ubuntu/CentOS/Debian

#) Quantization-tools+nntoolchain联合安装包，例如：bmnnsdk2-bm1684\_x.x.x

#) Bmnnsdk2-bm1684 Docker镜像


安装过程
~~~~~~~~

启动docker后进入SDK目录，执行以下脚本以设置路径和为示例程序配置环境。

  .. code-block:: shell

     $ cd bmnnsdk2-bm1684\_x.x.x/scripts/
     $ ./install_lib.sh nntc
     $ source ./envsetup_cmodel.sh

**注意：** 再次启动docker或新的终端窗口时，需要重新运行source ./envsetup_cmodel.sh，否则此环境没有被配置，不能
用来测试或者开发。


目录结构
~~~~~~~~~

安装完成之后，Calibration-tools相关的内容所在目录如下：

 ::

      |-- bin
      |   |-- arm
      |    `- x86
      |       |--- ***
      |        `-- calibration-tools
      |--- examples
      |   |--- ***
      |    `-- calibration
      |        `--- examples
      |           |--- classify_demo
      |           |--- create_lmdb_demo
      |           |--- face_demo
      |           |--- object_detection_python_demo
      |           |--- caffemodel_to_fp32umodel_demo
      |           |--- tf_to_fp32umodel_demo
      |           |--- pt_to_fp32umodel_demo
      |           |--- mx_to_fp32umodel_demo
      |           |--- dn_to_fp32umodel_demo
      |           |--- mtcnn_demo
      |           |--- auto_calib
      |            `-- view_demo
      ***

示例程序
~~~~~~~~~

安装后附带较多的示例程序，涵盖了制作LMDB，转化不同框架的网络和量化过程，以及可视化调试工具。可以参考 :ref:`demos_list`。

