##### 0501
* Bottom-Up and Top-Down Attention image captioning modelについて、以下のレポジトリの使用準備を行なった。<br>
　具体的には各種画像データセットのダウンロード、必要なライブラリの導入である。<br>
  まずは論文内で提案されている既存の画像データセットに対しeval.pyによるモデルの適用を試し、その後自分の動画に含まれる画像について適用を行う。<br>
  https://github.com/poojahira/image-captioning-bottom-up-top-down

##### 0502~0504
* Bottom-Up and Top-Down Attention image captioning modelについて、実際に稼働させた。<br>
レポジトリで用意されていたバリデーションデータで試してみた。以下はその結果の一例。<br>
```
references(正解):[['a girl in the car on the phone with something in her hand', 'a child in a vehicle holding some toys', 'a small asian child playing with toys in a car seat', 'a little child sitting in a car seat in the back seat of a car', 'a young child in the back seat of a car pretends to talk on a phone']]                                                   
hypotheses(推測):['a child sitting in a car holding a frisbee']
```

* Bottom-Up Attention Modelを探し、導入する作業を行なっている。現在はCaffeのビルド中であり、protocのインストールが必要になっている。<br>
0501から使用を始めたレポジトリでは、Captioning Modelの入力として既に特定のデータセットに対してBottom-Up Attention Modelを適用し得られた特徴量を使用していることがわかった。つまり、新規の画像(自分で用意した画像)にこのCaptioning Modelを適用するためにはBottom-Up Attention Modelを自分で用意する必要がある。したがって、以下のレポジトリをクローンし、本モデルを適用できるよう準備を行なっている。このレポジトリではPython2下でのビルドやCaffeの導入などが必要であり、不慣れであるため苦戦している。
https://github.com/peteanderson80/bottom-up-attention

##### 0507~0513
* DeepClusterを試した <br>
Facebookが公開しているレポジトリをとりあえず動かせるようにした。試しにLoLの動画1本(約90分)を15fpsで画像にし、1000Clustersに分類するタスクを行った。結果はほとんどランダムに分類されており、あまり良い性能とはいえなかった。

##### 0516
* PySceneDetectの導入(2回目) <br>
PySceneDetectのsourceをダウンロードし、PySceneDetectディレクトリ下で使用できるようにした。また、deepclusterのプログラム中でPython Interfaceとして呼び出せるようにdeepclusterで用いている仮想環境にも導入した。<br>
具体的には、deepclusterで用いているconda仮想環境"deepcluster"をactivateした状態でPySceneDetectのインストールを行っている。これにより、PySceneDetectディレクトリ下でなくてもdeepclusterというconda環境をactivateするだけで使用可能になっている。

##### 0514~0520
* DeepClusterを動画で試した <br>
PySceneDetectを用いて細かく分割した動画から連続する64フレームを抽出し、集めたものを用いて10Clustersに分類した。NNにはAlexNetを用いた。結果はほとんどランダムに分類されており、良い性能ではなかった。

##### 0521~0527
* DeepClusterのモデルをAlexNetからC3Dに変更した <br>
DeepClusterのAlexNetの代わりに3次元データを扱えるC3Dモデルを導入した。PySceneDetectを用いて細かく分割した動画から連続する16フレームを抽出し、それを1つの入力とした。動画1本(約90分)を379分割したものをデータとし、5Clustersに分類するタスクを行ったが、結果はほとんどランダムであった。また、動画を3本に増やし、計1000のデータセットを用いて同様の実験を行なったが、lossがnullになってしまった。

##### 0528~0602
* 学習を行っていないC3Dモデルを使って特徴量を抽出し、それをK-meansでクラスタリングした <br>
パラメータが初期値(heの初期値)のC3Dを使って、各16フレームの動画のデータセット(2500*3*16*112*112)についてそれぞれ4096次元の特徴量を抽出。その特徴量に対し、ScipyのKmeans2を用いてクラスタリングを行った。Cluster数は10, 30と変えて試したが、いずれも大差はなく、ある程度分類できているといった感じであった(確実に分割する事はできていなかった)。完全な分割はできないことを前提に、動画から必要な部分を抽出する手法を考える必要がある。