# gestureRecognition with Tetris 



使用手势识别算法玩俄罗斯方块

手势识别分类模型采用resnet18,类别为5类，分别代表左、右、变换、 快速下降、无变化下降.

### How to play?
1. 运行CollectData文件下的collectdata.py收集手势图像作为训练数据集
2. 将训练数据集按类别保存在trainDateset文件下，共5类
3. 调用trainNet.py训练手势识别网络，采用5折交叉验证
4. 调用demo.py进行测试
5. 运行game.py进行愉快的游戏吧

### Viedo
https://www.bilibili.com/video/BV11P4y147bD/

### Requirements
torch >= 1.6 

opencv-python == 3.4.2

### Explain

第一次在github上提交项目，在代码书写和说明文档上不够成熟，请大家多多见谅，如果在使用该代码上有任何问题的邮箱我 mrliu_9936@163.com。另外如果需要项目数据和模型参数文件的也可以邮我发送。
