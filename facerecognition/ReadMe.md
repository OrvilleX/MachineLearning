# face_recognition

## 一、安装方式

因为其底层需要依赖dlib，所以针对环境的安装需要具备进行一切前置的准备工作，才能使其
发挥作用，为此我们需要安装对应的底层依赖库，下面将根据不同的系统进行安装，并完成对应库
的引用，以便我们使用。  

### 1.1 编译环境安装

由于笔者的开发电脑采用的是MAC,所以首先我们对应的安装教程主要是围绕MAC系统，首先我们需要
安装编译最低需要的环境，需要先通Apple Store安装XCode开发工具，完成对应的安装工具后，我们
通过[homebrew](https://brew.sh)链接下载brew库管理工具，直接下载对应的pkg文件执行安装即可。  

完成上述的brew库管理包的安装后，我们就可以通过以下的命令安装需要的依赖库，以便安装dlib库的时候
能够顺利编译，对应的指令如下：

* MAC OS
```commandline
brew install cmake
brew install boost
brew install boost-python
```

* Ubuntu
```commandline
sudo apt-get install build-essential cmake
sudo apt-get install libgtk-3-dev
sudo apt-get install libboost-all-dev
```

如果遇到安装总是失败，此时需要通过代理进行安装，只需要在对应的指令前面增加对应的指令即可，具体如下所示，
对应的apt-get则可以通过更换国内源来加速安装：

```commandline
HTTP_PROXY=http://127.0.0.1:7890 HTTPS_PROXY=http://127.0.0.1:7890 brew install 包名
```

至此我们完成了编译所需要依赖的库，接着我们就可以通过pip安装其他需要的库文件了。

### 1.2 pip依赖包安装

```commandline
pip install dlib
pip install face_recognition
```

其中安装dlib库的时候一定要等待，因为其需要进行编译生成。完成上面的pip依赖库的安装后，我们就可以import测试
验证下对应的依赖库是否已经安装成功。  

## 二、实现视频检测

由于针对视频帧经过截取后，本身python采用的是浅复制，但是会导致face_recognition库内部识别的问题，为此需要通过
copy进行深度复制，从而避免这个问题的发生。

