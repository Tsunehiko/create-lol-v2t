##### 0603~0610
* PCAを使用して特徴量を減らした後K-meansによるClusteringを行った。結果が良くなかったので、アノテーションして教師あり学習を回すことにした <br>
C3Dモデルを通して得た4096次元の特徴量をPCAを使って100次元まで減らした後にK-meansによるClusteringを行ったが、結果はあまり良くなかった。そこで、抽出したい動画に対してアノテーションを行い、C3Dモデルを使った識別モデルを教師あり学習することにした。アノテーションを手作業だけで行うのは大変だったので、抽出したい動画の最初と最後のTimecodeだけを手入力することで自動的にラベルのcsvファイル生成＋動画の分割を行うプログラムを作成した。学習コードは8割りくらい実装した。

##### 0610~0617
* C3Dモデルと2Dモデルのfinetuningを行った <br>
C3Dモデルを使った識別モデルの教師あり学習を行った。validationのaccuracyがepoch0から一定値で動かないため、何かバグがあった可能性があるが、原因は特定できなかった。学習自体もepoch3で既にlossが収束するなど、うまくいっていない様子だった。2Dモデルのfinetuningではalexnetとsqueezenetの2つを使った(どちらもtorchvisionに含まれているものを使用した)。alexnetはlossも順調に下がり、accuracyも上昇したので学習はうまくいったように見える(ただし、それぞれの変化量はそこまで大きくない。最初からaccは0.9を超えており、学習済みであることが活きていたためと考えられる)。squeezenetはloss/acc共に挙動がおかしくなっており、うまくいかなかった。

##### 0618~0624
* 2Dモデルのfinetuningの続きを行った <br>
2Dモデルのfinetuningを行った。resnet18で試すところから始めていたが、accuracyの値が常に一定になってしまい、正確な値をとることができていなかった。accuracyの計算をnn.Sigmoid()を使う手法からtorch.sigmoid()を使う手法に変えることで、正常な値の取得に成功した。resnet18, resnet101, vgg11, squeezenet, resnext50_32x4dで試した。精度はどれも96%前後(squeezenetのみ92%程度)であり、optimizerを変えても精度にほとんど変化はなかった。
* Video Captioningの関連研究を調査した <br>
読んだ方が良さそうな論文をリストアップした。主にCVPR2016-2020, ICCV2018, ECCV2018あたりから。まだ読んでいない。

##### 0625~0701
* 2Dモデルのfinetuningの続きを行った <br>
前回最も性能が高かったresnext50_32x4dについてPresicion/Recall Curveを作成し、thresholdの最適地を探した。Precisionを重視してもthresholdはだいたい0.5~0.6の間くらいが最も良い。また、OptimizerはSGD(lr=0.002, momentum=0.7)あたりが良さそう。
* Video Captioningの関連研究 <br>
Deep Compositional Captioning:Describing Novel Object Categories without Paired Training Dataを読んだ。