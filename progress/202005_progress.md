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

##### 0516
* PySceneDetectの導入(2回目) <br>
PySceneDetectのsourceをダウンロードし、PySceneDetectディレクトリ下で使用できるようにした。また、deepclusterのプログラム中でPython Interfaceとして呼び出せるようにdeepclusterで用いている仮想環境にも導入した。<br>
具体的には、deepclusterで用いているconda仮想環境"deepcluster"をactivateした状態でPySceneDetectのインストールを行っている。これにより、PySceneDetectディレクトリ下でなくてもdeepclusterというconda環境をactivateするだけで使用可能になっている。
