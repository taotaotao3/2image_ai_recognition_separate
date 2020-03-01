import pdb
################################### pyocr prepare ################################################################
#
# (sepImTool) C:\Users\user\20190102>
#
# python AI_Image_Separate.py
#
# If featuring up → Put in beforeFolder & 特徴量抽出 push
# If Learning → Put in LearningFolder & 学習 push
# If Inference → Put in targetingFolder & 推論 push
#
#
#
#
################################### pyocr prepare ################################################################



######### to soft important setting ##################################################
import os
import sys
import time
### pyinstaller 使用時はこっち ######################################
# # directory Japanase character
# try:
#     # PyInstaller creates a temp folder and stores path in _MEIPASS
#     currentDirectory = sys._MEIPASS
# except Exception:
#     currentDirectory = os.getcwd()
# time.sleep(3)
# base_path = sys._MEIPASS
### pyinstaller 使用時はこっち ######################################



### pyinstaller 使わない時はこっち ######################################
currentDirectory = os.getcwd()
### pyinstaller 使わない時はこっち ######################################



print("currentDirectory", currentDirectory)

### pyinstaller 使うとき もし他Library等を同フォルダに準備しexe化でまとめたいなら ######################################
# arms_path = os.path.join(currentDirectory, 'arms', 'Tesseract-OCR')
# os.environ["PATH"] += os.pathsep + arms_path
# arms_path2 = os.path.join(currentDirectory, 'arms\\Tesseract-OCR\\tessdata', 'tessdata')
# os.environ["TESSDATA_PREFIX"] += os.pathsep + arms_path2

# TESSERACT_PATH = 'C:\\Users\\user\\20190102\\arms\\Tesseract-OCR'
# TESSDATA_PATH = 'C:\\Users\\user\\20190102\\arms\\Tesseract-OCR\\'

# os.environ["PATH"] += os.pathsep + TESSERACT_PATH
# os.environ["TESSDATA_PREFIX"] = TESSDATA_PATH

### pyinstaller 使うとき もし他Library等を同フォルダに準備しexe化でまとめたいなら ######################################

newCurrentDirectory = currentDirectory + '\\' # ex) C:\Users\user\20190102\\

image1_Directory = newCurrentDirectory + 'image\\before\\1\\'
image2_Directory = newCurrentDirectory + 'image\\before\\2\\'

gakusyu_moto2_dir1 = newCurrentDirectory + 'image\\learning\\1\\'
gakusyu_moto2_dir2 = newCurrentDirectory + 'image\\learning\\2\\'
gakusyu_moto2_dir = newCurrentDirectory + 'image\\learning\\'

target_Directory = newCurrentDirectory + 'image\\target\\'
target_learning_Directory = newCurrentDirectory + 'image\\target_learning\\'

result_Directory1 = newCurrentDirectory + 'image\\result\\1\\'
result_Directory2 = newCurrentDirectory + 'image\\result\\2\\'

# def no_folder_make(filename):
#     file_path = os.path.dirname(filename)
#     if not os.path.exists(file_path):
#         os.makedirs(file_path)
def no_folder_make(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

no_folder_make(image1_Directory)
no_folder_make(image2_Directory)
no_folder_make(gakusyu_moto2_dir1)
no_folder_make(gakusyu_moto2_dir2)
no_folder_make(target_Directory)
no_folder_make(target_learning_Directory)
no_folder_make(result_Directory1)
no_folder_make(result_Directory2)


# 画像名をリスト化
# list_filename = os.listdir(work_folder_dir)


########### ファイルを書くなら必ず必要となる独自メソッド（文字化け防止）##########################
import cv2 # pip install opencv-python
import numpy as np
def imwrite(filename, img, params=None):                        
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)

        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return False


def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8):    # ファイルを書くなら必ず必要となる独自メソッド（文字化け防止）
    try:
        print(filename)
        n = np.fromfile(filename, dtype)
        img = cv2.imdecode(n, flags)
        return img
    except Exception as e:
        print(e)
        return None

########### ファイルを書くなら必ず必要となる独自メソッド（文字化け防止）##########################






################# prepare ############################################################
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
import numpy as np

from keras import backend as K
from keras import initializers
from keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
import scipy.misc


from sklearn.model_selection import train_test_split
from skimage.transform import resize

import keras

from functools import reduce

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    LeakyReLU,
)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
# from keras.layers.advanced_activations import LeakyReLU

def compose(*funcs):
    """複数の層を結合する。
    """
    if funcs:
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError("Composition of empty sequence not supported.")


def ResNetConv2D(*args, **kwargs):
    """conv を作成する。
    """
    conv_kwargs = {
        "strides": (1, 1),
        "padding": "same",
        "kernel_initializer": "he_normal",
        "kernel_regularizer": l2(1.0e-4),
    }
    conv_kwargs.update(kwargs)

    return Conv2D(*args, **conv_kwargs)


def bn_relu_conv(*args, **kwargs):
    """batch mormalization -> ReLU -> conv を作成する。
    """
    return compose(
        # BatchNormalization(), Activation("relu"), ResNetConv2D(*args, **kwargs)
        BatchNormalization(), LeakyReLU(), ResNetConv2D(*args, **kwargs)
    )


def shortcut(x, residual):
    """shortcut connection を作成する。
    """
    x_shape = K.int_shape(x)
    residual_shape = K.int_shape(residual)

    if x_shape == residual_shape:
        # x と residual の形状が同じ場合、なにもしない。
        shortcut = x
    else:
        # x と residual の形状が異なる場合、線形変換を行い、形状を一致させる。
        stride_w = int(round(x_shape[1] / residual_shape[1]))
        stride_h = int(round(x_shape[2] / residual_shape[2]))

        shortcut = Conv2D(
            filters=residual_shape[3],
            kernel_size=(1, 1),
            strides=(stride_w, stride_h),
            kernel_initializer="he_normal",
            kernel_regularizer=l2(1.0e-4),
        )(x)
    return Add()([shortcut, residual])


def basic_block(filters, first_strides, is_first_block_of_first_layer):
    """bulding block を作成する。

        Arguments:
            filters: フィルター数
            first_strides: 最初の畳み込みのストライド
            is_first_block_of_first_layer: max pooling 直後の residual block かどうか
    """

    def f(x):
        if is_first_block_of_first_layer:
            # conv1 で batch normalization -> ReLU はすでに適用済みなので、
            # max pooling の直後の residual block は畳み込みから始める。
            conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)
        else:
            conv1 = bn_relu_conv(
                filters=filters, kernel_size=(3, 3), strides=first_strides
            )(x)

        conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)

        return shortcut(x, conv2)

    return f


def bottleneck_block(filters, first_strides, is_first_block_of_first_layer):
    """bottleneck bulding block を作成する。

        Arguments:
            filters: フィルター数
            first_strides: 最初の畳み込みのストライド
            is_first_block_of_first_layer: max pooling 直後の residual block かどうか
    """

    def f(x):
        if is_first_block_of_first_layer:
            # conv1 で batch normalization -> ReLU はすでに適用済みなので、
            # max pooling の直後の residual block は畳み込みから始める。
            conv1 = ResNetConv2D(filters=filters, kernel_size=(3, 3))(x)
        else:
            conv1 = bn_relu_conv(
                filters=filters, kernel_size=(1, 1), strides=first_strides
            )(x)

        conv2 = bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        conv3 = bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv2)

        return shortcut(x, conv3)

    return f


def residual_blocks(block_function, filters, repetitions, is_first_layer):
    """residual block を反復する構造を作成する。

        Arguments:
            block_function: residual block を作成する関数
            filters: フィルター数
            repetitions: residual block を何個繰り返すか。
            is_first_layer: max pooling 直後かどうか
    """

    def f(x):
        for i in range(repetitions):
            # conv3_x, conv4_x, conv5_x の最初の畳み込みは、
            # プーリング目的の畳み込みなので、strides を (2, 2) にする。
            # ただし、conv2_x の最初の畳み込みは直前の max pooling 層でプーリングしているので
            # strides を (1, 1) にする。
            first_strides = (2, 2) if i == 0 and not is_first_layer else (1, 1)

            x = block_function(
                filters=filters,
                first_strides=first_strides,
                is_first_block_of_first_layer=(i == 0 and is_first_layer),
            )(x)
        return x

    return f


class ResnetBuilder:
    @staticmethod
    def build(input_shape, num_outputs, block_type, repetitions):
        """ResNet モデルを作成する Factory クラス

        Arguments:
            input_shape: 入力の形状
            num_outputs: ネットワークの出力数
            block_type : residual block の種類 ('basic' or 'bottleneck')
            repetitions: 同じ residual block を何個反復させるか
        """
        # block_type に応じて、residual block を生成する関数を選択する。
        if block_type == "basic":
            block_fn = basic_block
        elif block_type == "bottleneck":
            block_fn = bottleneck_block

        # モデルを作成する。
        ##############################################
        input = Input(shape=input_shape)

        # conv1 (batch normalization -> ReLU -> conv)
        conv1 = compose(
            ResNetConv2D(filters=64, kernel_size=(7, 7), strides=(2, 2)),
            BatchNormalization(),
            # Activation("relu"),
            LeakyReLU(),
        )(input)

        # pool
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        # conv2_x, conv3_x, conv4_x, conv5_x
        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = residual_blocks(
                block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0)
            )(block)
            filters *= 2

        # batch normalization -> ReLU
        # block = compose(BatchNormalization(), Activation("relu"))(block)
        block = compose(BatchNormalization(), LeakyReLU())(block)

        # global average pooling
        pool2 = GlobalAveragePooling2D()(block)

        # dense
        fc1 = Dense(
            units=num_outputs, kernel_initializer="he_normal", activation="softmax"
        )(pool2)

        return Model(inputs=input, outputs=fc1)

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, "basic", [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, "basic", [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, "bottleneck", [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, "bottleneck", [3, 4, 23, 3]
        )

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(
            input_shape, num_outputs, "bottleneck", [3, 8, 36, 3]
        )

from tensorflow.python.client import device_lib
device_lib.list_local_devices()



######################################################################################







################# method prepare ############################################################
















############ 画面表示　画像抽出へ ##########################
import os
import wx
choose_text_h5 = ""
class Main(wx.Frame):
    
    def __init__(self, parent, id, title):
        """ レイアウトの作成 """
 
        wx.Frame.__init__(self, parent, id, title, size=(620, 400))
        self.folder = os.path.dirname(os.path.abspath(__file__))
        self.h5file = "～.h5 ファイル"
        panel1 = wx.Panel(self, wx.ID_ANY)
        v_layout = wx.BoxSizer(wx.VERTICAL)

        # 説明書き１
        s_text_1 = wx.StaticText(panel1, wx.ID_ANY, 'はじめに学習させたい画像2パターンをimageフォルダの中のbeforeフォルダへ入れて下さい。', style=wx.TE_CENTER)
        v_layout.Add(s_text_1, proportion=0, flag=wx.EXPAND)

        # 画像1ディレクトリ表示
        # self.choose_text = wx.StaticText(panel1, wx.ID_ANY, self.folder, style=wx.TE_CENTER)
        choose_text = wx.StaticText(panel1, wx.ID_ANY, 'C:\\Users\\user\\20190102\\image\\before\\1', style=wx.TE_CENTER)
        v_layout.Add(choose_text, proportion=0, flag=wx.EXPAND)

        # 画像2ディレクトリ表示
        # self.choose_text2 = wx.StaticText(panel1, wx.ID_ANY, self.folder, style=wx.TE_CENTER)
        choose_text2 = wx.StaticText(panel1, wx.ID_ANY, 'C:\\Users\\user\\20190102\\image\\before\\2', style=wx.TE_CENTER)
        v_layout.Add(choose_text2, proportion=0, flag=wx.EXPAND)

        # 説明書き１と２の間
        s_text_12 = wx.StaticText(panel1, wx.ID_ANY, '↓', style=wx.TE_CENTER)
        v_layout.Add(s_text_12, proportion=0, flag=wx.EXPAND)

        # 説明書き２
        s_text_2 = wx.StaticText(panel1, wx.ID_ANY, '画像を入れたら、それらの特徴抽出を行いましょう。'+ '\r\n' + '（C:\\Users\\user\\20190102\\image\\learning\\に既に' + '\r\n' + '画像が作成されているのであれば省略可能です。)', style=wx.TE_CENTER)
        v_layout.Add(s_text_2, proportion=0, flag=wx.EXPAND)


        # 特徴抽出ボタンButton
        extruct_feature_button = wx.Button(panel1, wx.ID_ANY, "特徴抽出")
        extruct_feature_button.Bind(wx.EVT_BUTTON, self.extruct_feature_method)
        v_layout.Add(extruct_feature_button, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)

        # 説明書き２と３の間
        s_text_23 = wx.StaticText(panel1, wx.ID_ANY, '↓', style=wx.TE_CENTER)
        v_layout.Add(s_text_23, proportion=0, flag=wx.EXPAND)

        # 説明書き３
        s_text_3 = wx.StaticText(panel1, wx.ID_ANY, '抽出完了のダイアログが出たら、学習を行いましょう。' + '\r\n' + '（既に学習済みであれば省略できます。次に進んで下さい。)', style=wx.TE_CENTER)
        v_layout.Add(s_text_3, proportion=0, flag=wx.EXPAND)


        # 学習Button
        learning_button = wx.Button(panel1, wx.ID_ANY, "学習")
        learning_button.Bind(wx.EVT_BUTTON, self.learning_method)
        v_layout.Add(learning_button, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)

        # 説明書き４と５の間
        s_text_45 = wx.StaticText(panel1, wx.ID_ANY, '↓', style=wx.TE_CENTER)
        v_layout.Add(s_text_45, proportion=0, flag=wx.EXPAND)

        # 説明書き４と５の間
        s_text_45 = wx.StaticText(panel1, wx.ID_ANY, 'C:\\Users\\user\\20190102\\image\\target\\ に、' + '\r\n' + '分類したい画像群を入れて下さい。', style=wx.TE_CENTER)
        v_layout.Add(s_text_45, proportion=0, flag=wx.EXPAND)

        # 説明書き４と５の間
        s_text_45 = wx.StaticText(panel1, wx.ID_ANY, '↓', style=wx.TE_CENTER)
        v_layout.Add(s_text_45, proportion=0, flag=wx.EXPAND)

        # 説明書き３.５
        s_text_354 = wx.StaticText(panel1, wx.ID_ANY, '学習モデルのh5ファイルを選択し、分類（推論）を行う。', style=wx.TE_CENTER)
        v_layout.Add(s_text_354, proportion=0, flag=wx.EXPAND)

        # 学習済でモデル選択Button
        choose_button1 = wx.Button(panel1, wx.ID_ANY, "分類作業開始（既学習モデル選択）")
        choose_button1.Bind(wx.EVT_BUTTON, self.choose_h5_inference_method)
        v_layout.Add(choose_button1, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)

        # h5ディレクトリ表示
        # self.choose_text_h5 = wx.StaticText(panel1, wx.ID_ANY, self.folder, style=wx.TE_CENTER)
        #self.choose_text_h5 = wx.StaticText(panel1, wx.ID_ANY, self.folder, style=wx.TE_CENTER)
        #v_layout.Add(choose_text_h5, proportion=0, flag=wx.EXPAND)

        # 説明書き３.５と４の間
        # s_text_34 = wx.StaticText(panel1, wx.ID_ANY, '↓', style=wx.TE_CENTER)
        # v_layout.Add(s_text_34, proportion=0, flag=wx.EXPAND)

        # 説明書き４
        # s_text_4 = wx.StaticText(panel1, wx.ID_ANY, '学習モデルを選択後に、分類（推論）ボタンを押して下さい。' + '\r\n' + 'Resultフォルダに画像が分別されます。', style=wx.TE_CENTER)
        # v_layout.Add(s_text_4, proportion=0, flag=wx.EXPAND)


        # フォルダ選択(分類前)Button
        # choose_button3 = wx.Button(panel1, wx.ID_ANY, "フォルダの選択(分類前)")
        # choose_button3.Bind(wx.EVT_BUTTON, self.choose_h5_method)
        # v_layout.Add(choose_button3, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)

        # 分類前ディレクトリ表示
        # self.choose_text3 = wx.StaticText(panel1, wx.ID_ANY, self.folder, style=wx.TE_CENTER)
        # v_layout.Add(self.choose_text3, proportion=0, flag=wx.EXPAND)


        # 説明書き５
        # s_text_5 = wx.StaticText(panel1, wx.ID_ANY, '分類したい画像群の場所を指定したら、最後に分類(推論)ボタンを押しましょう。Resultフォルダに画像が分かれて出力されます。', style=wx.TE_CENTER)
        # v_layout.Add(s_text_5, proportion=0, flag=wx.EXPAND)


        # 分類実行Button
        # inference_button = wx.Button(panel1, wx.ID_ANY, "分類(推論)")
        # inference_button.Bind(wx.EVT_BUTTON, self.inference_method)
        # v_layout.Add(inference_button, proportion=0, flag=wx.ALIGN_CENTER_HORIZONTAL)

        panel1.SetSizer(v_layout)
 
        self.Centre()
        self.Show(True)
 


    def OnExitApp(self, event):
        # self.Close(True) # ← self.Close(True) & self.Exit(True) → 完全にアプリが閉じられる
        self.Exit(True) # ← self.Exit(True) → 元のトップ画面に戻るだけ

    # D:\ProgramData\Anaconda3\envs\py37gpu_resnet\gazou_bunrui_wake\gakusyu_moto2\1
    # ↑に色塗りされた画像が出来ていればここは実行する必要性なし

    # フォルダ1を実行
    def image1_feature_extruct(self, event):
        import cv2
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 1000000000
        import sys
        import os

        # 以下のディレクトリに画像が配置してある
        # work_folder_dir = newCurrentDirectory + "gazou_bunrui_wake\\1\\"
        # 画像名をリスト化
        list_hensuu = os.listdir(image1_Directory)
        if(list_hensuu == []):
            wx.MessageBox('beforeフォルダの中の1番フォルダに画像が入っていません。')
            self.OnExitApp(event)
        # 初めに画像特徴量抽出メソッド定義
        def gazou_feature_extruct(file_name):
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = 1000000000
            import cv2
            import os
            import numpy as np
            import pdb
            from threshold import apply_threshold
            from matplotlib import pylab as plt
            def square_detect_method(sdm_file):
                def create_if_not_exist(out_dir):
                    try:
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                    except Exception as e:
                        print(e)


                output_path = 'out'
                create_if_not_exist(output_path)

                ######### gaucian etc #####################################
                currentdirectory = os.getcwd()
                # filename = '1.png'
                filename = sdm_file
                # PIL open for pixel up
                img1 = Image.open(image1_Directory + filename)

                width = img1.size[0] * 10
                height = img1.size[1] * 10


                img2 = img1.resize((width, height))  # ex)(39, 374, 3) → (390, 3740, 3)
                img2.save(currentdirectory + '\\1600_' + filename)

                ####### white line making around #########################
                # OpenCV open for after process
                img2 = cv2.imread(currentdirectory + '\\1600_' + filename)

                # load image, change color spaces, and smoothing
                img_HSV = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
                os.remove(currentdirectory + '\\1600_' + filename)
                # cv2.imwrite(filename + '_4HSV.jpg', img_HSV)

                img_HSV = cv2.GaussianBlur(img_HSV, (9, 9), 3)
                # cv2.imwrite(filename + '_5GaussianBlur.jpg', img_HSV)

                # detect tulips
                img_H, img_S, img_V = cv2.split(img_HSV)
                # cv2.imwrite(filename + '_6splitH.jpg', img_H)
                # cv2.imwrite(filename + '_7splitS.jpg', img_S)
                # cv2.imwrite(filename + '_8splitV.jpg', img_V)
                _thre, img_flowers = cv2.threshold(img_V, 140, 255, cv2.THRESH_BINARY)
                
                # cv2.imwrite(filename + '_9mask.jpg', img_flowers)

                # img_flowers_copy = img_flowers.copy()

                # find tulips
                contours, hierarchy = cv2.findContours(
                    img_flowers, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # labels,  delete


                ###### square to black################################################
                # out = cv2.imread(filename + '_9mask.jpg')

                for i in range(0, len(contours)):
                    if len(contours[i]) > 0:
                        print(cv2.contourArea(contours[i]))
                        # remove small objects
                        if cv2.contourArea(contours[i]) < 200:
                            continue
                        if cv2.contourArea(contours[i]) > 1000000:
                            continue
                        rect = contours[i]
                        x, y, w, h = cv2.boundingRect(rect)
                        cv2.rectangle(img_flowers, (x, y), (x + w, y + h), (0, 0, 0), -1)

                # squre in squre is bad
                # this point is black square
                # cv2.imwrite('./out' + '\\' + filename + '_9-2.jpg', img_flowers)

                # 学習用に画像を別途保存（フォルダをあらかじめ作っておかないと作成されない）
                #cv2.imwrite(file_dir + "gakusyu_moto2\\" + "1\\" + file_name, out)
                cv2.imwrite(gakusyu_moto2_dir1 +  filename + '_9-2.jpg', img_flowers)
                # cv2.imshow("img", out)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            square_detect_method(file_name)
        # リスト化された画像名をメソッドに投入
        for list_hensuu_ko in list_hensuu:
            gazou_feature_extruct(list_hensuu_ko)

    ######################################################################################


    #########################################################################################
    # D:\ProgramData\Anaconda3\envs\py37gpu_resnet\gazou_bunrui_wake\gakusyu_moto2\1
    # ↑に色塗りされた画像が出来ていればここは実行する必要性なし

    # フォルダ2に対して実行
    def image2_feature_extruct(self, event):
        import cv2
        from PIL import Image
        import sys
        import os

        # 以下のディレクトリに画像が配置してある
        # work_folder_dir = newCurrentDirectory + "gazou_bunrui_wake\\1\\"
        # 画像名をリスト化
        list_hensuu = os.listdir(image2_Directory)
        if(list_hensuu == []):
            wx.MessageBox('beforeフォルダの中の1番フォルダに画像が入っていません。')
            self.OnExitApp(event)
        # 初めに画像特徴量抽出メソッド定義
        def gazou_feature_extruct(file_name):
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = 1000000000
            import cv2
            import os
            import numpy as np
            import pdb
            from threshold import apply_threshold
            from matplotlib import pylab as plt
            def square_detect_method(sdm_file):
                def create_if_not_exist(out_dir):
                    try:
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                    except Exception as e:
                        print(e)


                output_path = 'out'
                create_if_not_exist(output_path)

                ######### gaucian etc #####################################
                currentdirectory = os.getcwd()
                # filename = '1.png'
                filename = sdm_file
                # PIL open for pixel up
                img1 = Image.open(image2_Directory + filename)

                width = img1.size[0] * 10
                height = img1.size[1] * 10


                img2 = img1.resize((width, height))  # ex)(39, 374, 3) → (390, 3740, 3)
                img2.save(currentdirectory + '\\1600_' + filename)

                ####### white line making around #########################
                # OpenCV open for after process
                img2 = cv2.imread(currentdirectory + '\\1600_' + filename)

                # load image, change color spaces, and smoothing
                img_HSV = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
                os.remove(currentdirectory + '\\1600_' + filename)
                # cv2.imwrite(filename + '_4HSV.jpg', img_HSV)

                img_HSV = cv2.GaussianBlur(img_HSV, (9, 9), 3)
                # cv2.imwrite(filename + '_5GaussianBlur.jpg', img_HSV)

                # detect tulips
                img_H, img_S, img_V = cv2.split(img_HSV)
                # cv2.imwrite(filename + '_6splitH.jpg', img_H)
                # cv2.imwrite(filename + '_7splitS.jpg', img_S)
                # cv2.imwrite(filename + '_8splitV.jpg', img_V)
                _thre, img_flowers = cv2.threshold(img_V, 140, 255, cv2.THRESH_BINARY)
                
                # cv2.imwrite(filename + '_9mask.jpg', img_flowers)

                # img_flowers_copy = img_flowers.copy()

                # find tulips
                contours, hierarchy = cv2.findContours(
                    img_flowers, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # labels,  delete


                ###### square to black################################################
                # out = cv2.imread(filename + '_9mask.jpg')

                for i in range(0, len(contours)):
                    if len(contours[i]) > 0:
                        print(cv2.contourArea(contours[i]))
                        # remove small objects
                        if cv2.contourArea(contours[i]) < 200:
                            continue
                        if cv2.contourArea(contours[i]) > 1000000:
                            continue
                        rect = contours[i]
                        x, y, w, h = cv2.boundingRect(rect)
                        cv2.rectangle(img_flowers, (x, y), (x + w, y + h), (0, 0, 0), -1)

                # squre in squre is bad
                # this point is black square
                # cv2.imwrite('./out' + '\\' + filename + '_9-2.jpg', img_flowers)

                # 学習用に画像を別途保存（フォルダをあらかじめ作っておかないと作成されない）
                #cv2.imwrite(file_dir + "gakusyu_moto2\\" + "1\\" + file_name, out)
                cv2.imwrite(gakusyu_moto2_dir2 +  filename + '_9-2.jpg', img_flowers)
                # cv2.imshow("img", out)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            square_detect_method(file_name)
        # リスト化された画像名をメソッドに投入
        for list_hensuu_ko in list_hensuu:
            gazou_feature_extruct(list_hensuu_ko)

    def extruct_feature_method(self, event):
        self.image1_feature_extruct(event)
        self.image2_feature_extruct(event)
        wx.MessageBox('特徴抽出が完了しました。')
        








    def learning_method(self, event):
        import os
        list_hensuu_lrn1 = os.listdir(gakusyu_moto2_dir1)
        if(list_hensuu_lrn1 == []):
            wx.MessageBox('learningフォルダの中の1番フォルダに画像が入っていません。')
            self.OnExitApp(event)
        list_hensuu_lrn2 = os.listdir(gakusyu_moto2_dir2)
        if(list_hensuu_lrn2 == []):
            wx.MessageBox('learningフォルダの中の2番フォルダに画像が入っていません。')
            self.OnExitApp(event)
        
        # learning method and extruct Learning data

        ############################ 学習 learning #######################################################################################

        ############################# クラスをリストに格納　画像をリストに格納 #####################################################

        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 1000000000
        import os,glob
        import numpy as np
        from sklearn import model_selection
        import glob
        # 画像がフォルダにクラス分けされて格納されているディレクトリ
        # 例：gazou_bunrui_wake/ikki/～.jpg
        # 例：gazou_bunrui_wake/niki/～.jpg
        files = glob.glob(gakusyu_moto2_dir + "*")
        print(files[0][-1:]) # > ikki
        classes = []

        #クラスを配列に格納
        for i in range(0,2,1): # range(スタートの数値,　何個数えるか, 何個ずつ数えるか)(0, 3, 1) → 0,1,2  ※3までいかない。
            classes.append(files[i][-1:])
        num_classes = len(classes)
        print(num_classes)
        # image_size = 128  ※ 1650, 2330に以下で指定
        image_size1 = 165
        image_size2 = 233
        print(classes)

        #画像の読み込み
        #最終的に画像、ラベルはリストに格納される
        import os
        X = []
        Y = []
        #まずfor文で画像のインデックスとクラスを取得(1:りんご,2,ブドウ...)
        #for index,classlabel in enumerate(classes):

        # photos_dir = newCurrentDirectory + "gazou_bunrui_wake\\gakusyu_moto2"
        photos_dir = gakusyu_moto2_dir
        
        #globでそれぞれの漢字一文字ずつのフォルダを取得
        files = glob.glob(photos_dir + "/*") # ['C:\\Users\\user\\20190102\\image\\learning\\1', 'C:\\Users\\user\\20190102\\image\\learning\\2']
        # file = D:/ProgramData/Anaconda3/envs/py37gpu_resnet/gazou_bunrui_wake/ikki
        for index,file in enumerate(files): 

            print(index)
            print("◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆")
            if(index >= 2): # フォルダが２つなので0 , 1 のカウントまでなので2以上でbreak
                break
            files2 = glob.glob(file + "/*") # ['C:\\Users\\user\\20190102\\image\\learning\\1\\out_0_70090154001001financial_statements1_dev_cure.jpg', 'C:\\Users\\user\\20190102\\image\\learning\\1\\out_0_70092582001001financial_statements1_dev_cure.jpg']
            # クラスの画像一覧を取得、それぞれのクラスの200枚までの画像を取得
            # file2 = D:/ProgramData/Anaconda3/envs/py37gpu_resnet/extract/一/ETL9G_01_000079.png
            for t, file2 in enumerate(files2):

                if(t >= 144): # 片方が30個なのでtが29まで来るので30以上でbreakに
                    break
                # print(t)
                image = Image.open(file2)
                image = image.convert("RGB")
                # グレースケールに変換
                gray_image = image.convert('L')
                gray_image_resize = gray_image.resize((image_size1, image_size2)) # 画像の平均ピクセル縦横の値
                #イメージを1枚ずつnumpy配列に変換
                data = np.asarray(gray_image_resize)
                #リストに格納
                X.append(data)
                Y.append(index) # index + 1 とやって無理やり　1と2のクラス分類するとonehot分けで0,1,2の3分けに
                print("aaaa")
                print(Y)
                print(len(Y))
            print("bbbbb")
            print(len(Y))
        print("ccccc")
        print(len(Y))



        print(Y)
        print(len(X))
        print(len(Y))
        print(data.shape)
        #格納したリストをさらにnumpy配列に変換
        X = np.array(X)
        Y = np.array(Y)

        print(X.shape)
        print(Y.shape)


        ############################# クラスをリストに格納　画像をリストに格納 #####################################################
















        ##################### リストになった画像を増幅 ################################################
        # ↓　増幅
        # 想定 入出力shape
        # (4920001, 28, 28)
        # (4920001,)
        #

        import keras
        import numpy as np
        from keras.utils import np_utils
        from matplotlib import pyplot as plt
        from keras.preprocessing import image
        from keras.preprocessing.image import ImageDataGenerator


        '''# サンプル画像を読み込み、4次元テンソルに変換（sample数, row, col, channel)
        img = image.load_img("D:/ProgramData/Anaconda3/envs/py37gpu_resnet/tegaki_math.jpg")
        print(img)
        x = image.img_to_array(img)
        print(x.shape)
        x = np.expand_dims(x, axis=0)
        print(x.shape)

        # 持ち込みデータ を増幅メソッドに入れるため
        '''

        # 画像を表示する
        def show_imgs(imgs, row, col):
            if len(imgs) != (row * col):
                raise ValueError("Invalid imgs len:{} col:{} row:{}".format(len(imgs), row, col))

            for i, img in enumerate(imgs):
                plot_num = i+1
                plt.subplot(row, col, plot_num)
                plt.tick_params(labelbottom="off") # x軸の削除
                plt.tick_params(labelleft="off") # y軸の削除
                plt.imshow(img)
            plt.show()
        # 画像増幅メソッド
        # 引数
        # (増幅対象1枚画像, x軸移動度合い,y軸移動度合い, 今回学習数字, 元学習追加データ、元ラベル追加データ)
        # 
        # （aug_imgs, x, y, now_number, moto_gakusyu, moto_labels)
        # 例：(train_images[23], 0.2, 0.4, 5, train_images, train_labels)  ⇒　返却：train_data, tarain_test
        def amplification_img_super(aug_imgs, x, y, now_number, moto_gakusyu, moto_labels):
            train_images_test = aug_imgs.reshape(image_size1, image_size2)
            train_images_test_dim1 = np.expand_dims(train_images_test, axis=0)
            train_images_test_dim2 = np.expand_dims(train_images_test_dim1, axis=3)
            datagen = ImageDataGenerator(
                        width_shift_range = x,
                        height_shift_range = 0,
                        zoom_range = [1, 1],
                        rotation_range = 0,
                        )
            # 1枚の画像に対して何枚増幅させるか
            num = 9
            max_img_num = num
            imgs_change_i = []
            count = 0
            x_imgs28_append = []

            for d in datagen.flow(train_images_test_dim2, batch_size=1): # 数字の9が1枚ある　⇒　0.7～1.3倍ランダムzoom生成を開始 ★1枚
                imgs_change_i.append(image.array_to_img(d[0], scale=True)) # ほしい画像数9枚になるまでappend　　　　　　　　　　　★9枚
                if (len(imgs_change_i) % max_img_num) == 0:
                    break        
            # show_imgs(imgs_change_i, row=3, col=3)
            # plt.imshow(imgs_change_i3[4])
            # 増幅された9枚の１枚ずつさらに９枚に増幅した１枚ずつをさらに９枚に増幅し、さらにその1枚ずつを９枚に増幅
            for i in range(num):
                x_imgs = image.img_to_array(imgs_change_i[i])
                x_imgs28 = x_imgs.reshape(image_size1, image_size2)
                x_imgs28t_dim1 = np.expand_dims(x_imgs28, axis=0)
                x_imgs28t_dim2 = np.expand_dims(x_imgs28t_dim1, axis=3)
                datagen = ImageDataGenerator(
                        width_shift_range = 0,
                        height_shift_range = y,
                        zoom_range = [1, 1],
                        rotation_range = 0,
                        )
                # 1枚の画像に対して何枚増幅させるか
                max_img_num = num
                imgs_change_i4 = []
                for i_enu, d in enumerate(datagen.flow(x_imgs28t_dim2, batch_size=1)):
                    imgs_change_i4.append(image.array_to_img(d[0], scale=True)) #                                           ★6561枚
                    # count = count + 1
                    # print(count)
                    x_imgs = image.img_to_array(imgs_change_i4[i_enu])
                    x_imgs28 = x_imgs.reshape(image_size1, image_size2) # A reshape
                    x_imgs28_dim1 = np.expand_dims(x_imgs28, axis=0)
                    x_imgs28_append.append(x_imgs28_dim1)
                    if (len(imgs_change_i4) % max_img_num) == 0:
                        break
                #show_imgs(imgs_change_i4, row=3, col=3)
            #return x_imgs28_append
            x_imgs28_zoukekka = np.array(x_imgs28_append)
            # print(x_imgs28_zoukekka.shape)
            # print(x_imgs28_zoukekka.dtype)
            # print(moto_gakusyu.dtype)
            # reshape                                                 # A reshapeとはxとyが逆
            x_imgs28_zoukekka_6561_28_28_1 = x_imgs28_zoukekka.reshape([81, image_size2, image_size1, 1])
            # print(x_imgs28_zoukekka_6561_28_28_1.shape)
            # 次元下げ
            x_imgs28_zoukekka_6561_28_28 = np.squeeze(x_imgs28_zoukekka_6561_28_28_1)
            # print(x_imgs28_zoukekka_6561_28_28.shape)
            # print(x_imgs28_zoukekka_6561_28_28.dtype)
            x_imgs28_zoukekka_6561_28_28 = x_imgs28_zoukekka_6561_28_28.astype(np.uint8)
            # print(x_imgs28_zoukekka_6561_28_28.dtype)
            #print(moto_gakusyu.dtype)
            # print("画像データの要素数", moto_gakusyu.shape)
            # print("ラベルデータの要素数", moto_labels.shape)

            retrain_images = []
            # Goalに元の画像を入れる
            retrain_images.append(moto_gakusyu)                             # ★もともとの学習データ　train_images 引数に必要
            # print("aaaaaaaaaaaaaaaaaaaaaaa")
            # print(moto_gakusyu.shape)
            # print(x_imgs28_zoukekka_6561_28_28.shape)
            # print("bbbbbbbbbbbbbbbbbbbbbbbb")
            retrain_images.append(x_imgs28_zoukekka_6561_28_28)
            all_sum_images = np.concatenate(retrain_images)

            # print(all_sum_images.shape)
            moto_gakusyu = all_sum_images
            #print(moto_gakusyu.shape)

            # labels側にも数字9のラベルをふって加算しよう

            # ①まずは既存のmoto_labelsをnumpy.adarrayからリストに変換（新しいラベルをappendするため）
            #print(moto_labels.shape)                                       # ★もともとの学習データ　train_labels 引数に必要
            # ★ NumPy配列ndarrayをリストに変換: tolist()
            # print(moto_labels)
            # [0 1 2]
            l_moto_labels = moto_labels.tolist()
            #print(l_train_labels)
            # [0, 1, 2]

            # ②リストに変換した既存ラベルに新しいラベルをappendしていく
            ir_labels = 81
            for i_labels in range(ir_labels):
                # appendする中身は数字のnow_numberというラベル
                l_moto_labels.append(now_number)

            # print('aaaaaaaaaaaaaaa')
            # ③完成したリストをnumpyのndarrayへ戻す
            moto_labels = np.array(l_moto_labels)

            #print(moto_labels.shape)
            #print(moto_labels)
            # 9を追加したデータを保存
            train_images_9 = moto_gakusyu
            train_labels_9 = moto_labels
            #print(moto_gakusyu.shape)
            #print(moto_labels.shape)
            # この時点で出来上がったのが train_images, train_labelsになる
            return moto_gakusyu, moto_labels



        print(X.shape)
        print(Y.shape)


        # 指定画像数に対して81倍に増やす (例：range(0,59999))
        for i in range(0,173): # 47
            print(i)
            #print("ラベル", train_labels[i])
            #print(train_labels[i].dtype)
            #plt.imshow(train_images[i].reshape(28, 28), cmap='Greys')
            #plt.show()
            # def amplification_img_super(aug_imgs, x, y, now_number, moto_gakusyu, moto_labels):
            X, Y = amplification_img_super(X[i], 0.001, 0.001, Y[i], X, Y)

        ##################### リストになった画像を増幅 ################################################









        ############################## 学習前準備　データを訓練とテストに分ける ################################

        X_train = X
        X_test = []
        Y_train = Y
        Y_test = []

        # 訓練:テスト = 8:2に分割
        # X_train, X_test, T_train, T_test = train_test_split(digits.data, digits.target, test_size=0.2)


        X_train,X_test,Y_train,Y_test = model_selection.train_test_split(X_train,Y_train, test_size=0.2)

        print(X_train.shape)
        print(X_test.shape)
        print(Y_train.shape)
        print(Y_test.shape)
        print(Y_train)
        print("----------------------------------------")
        #画像の正規化
        X_train = X_train.astype("float")/256
        X_test = X_test.astype("float")/256

        # y_train = []
        # y_test = []

        #ラベルをカテゴリ変数に変換
        Y_train = np_utils.to_categorical(Y_train,num_classes)
        Y_test = np_utils.to_categorical(Y_test,num_classes)

        # ★ dtypeのキャスト（modelを作った際のデータに合わせる）
        print(X_train.dtype)
        X_train_res = X_train.astype(np.float32)
        print(X_train_res.dtype)

        print(X_test.dtype)
        X_test_res = X_test.astype(np.float32)
        print(X_test_res.dtype)

        print("aaaaaaaaaaaaaaaaaa")
        print(Y_train.dtype)
        #Y_train_res = Y_train.astype(np.int32)
        print(Y_train.dtype)

        print(Y_test.dtype)
        #Y_test_res = Y_test.astype(np.int32)
        print(Y_test.dtype)


        #X_train = X_train_res
        print(X_train.shape)
        print(X_train.dtype)

        #Y_train = Y_train_res
        print(Y_train.shape)
        Y_train = Y_train.astype(np.int32)
        print(Y_train.dtype)

        print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")

        #X_test = X_test_res
        print(X_test.shape)
        print(X_test.dtype)

        #Y_test = Y_test_res
        print(Y_test.shape)
        Y_test = Y_test.astype(np.int32)
        print(Y_test.dtype)


        # X_train
        # print("X_train:{0}".format(X_train))
        print("X_train.shape:{0}".format(X_train.shape))
        print("X_train.dtype:{0}".format(X_train.dtype))

        # Y_train
        # print("Y_train:{0}".format(Y_train))
        print("Y_train.shape:{0}".format(Y_train.shape))
        print("Y_train.dtype:{0}".format(Y_train.dtype))


        # X_test
        # print("X_test:{0}".format(X_test))
        print("X_test.shape:{0}".format(X_test.shape))
        print("X_test.dtype:{0}".format(X_test.dtype))

        # Y_test
        # print("Y_test:{0}".format(Y_test))
        print("Y_test.shape:{0}".format(Y_test.shape))
        print("Y_test.dtype:{0}".format(Y_test.dtype))



        # y_train
        #print("y_train.shape:{0}".format(y_train.shape))
        #print("y_train.dtype:{0}".format(y_train.dtype))

        # y_test
        #print("y_test.shape:{0}".format(y_test.shape))
        #print("y_test.dtype:{0}".format(y_test.dtype))

        import matplotlib.pyplot as plt
        import numpy as np

        ############### ラベルと画像データを表示 ####################################
        # for i in range(3700,3800):
        #     print("ラベル", Y_train[i])
        #     #plt.imshow(X_train[i].reshape(128, 128), cmap='Greys')
        #     plt.imshow(X_train[i])
        #     plt.show()
        ############### ラベルと画像データを表示 ####################################


        # (H, W) -> (H, W, 1) にする。
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
        # クラス名の対応の確認
        print(classes)


        print(X_train.shape)
        print(X_train.dtype)
        print(X_train[0])
        print(Y_train[0])
        # float64にしなければresnetエンジンと同じにならないが…？

        print(X_test.shape)
        print(X_test.dtype)
        print(X_test[0])
        print(Y_test[0])

        # (H, W) -> (H, W, 1) にする。
        # X_train_res = X_train_res[..., np.newaxis]
        # X_test_res = X_test_res[..., np.newaxis]
        # https://translate.googleusercontent.com/translate_c?depth=1&hl=ja&prev=search&rurl=translate.google.co.jp&sl=en&sp=nmt4&u=https://stackoverflow.com/questions/49083984/valueerror-can-not-squeeze-dim1-expected-a-dimension-of-1-got-3-for-sparse&xid=17259,15700022,15700186,15700191,15700256,15700259,15700262,15700265&usg=ALkJrhiP80VIGCJouCEhlq42acfud7lDHA
        # Yがonehotベクトルになっていないなどで追加2行↓
        # Y_train = Y_train[..., np.newaxis]
        # Y_test = Y_test[..., np.newaxis]

        # クラス ID とクラス名の対応
        #   "あ",

        ### print(X_train.shape)
        print(Y_train.shape)
        print(len(Y_train))
        print(len(Y_test))

        print(Y_train.shape)
        print(Y_test.shape)


        ############################## 学習前準備　データを訓練とテストに分ける ################################









        ############################# model作成とcompile ####################################################
        # input_shape = (127, 128, 1)  # モデルの入力サイズ
        input_shape = (image_size2, image_size1, 1)  # モデルの入力サイズ #   x  y　逆の場合も学習して結果要確認

        # num_classes = 72  # クラス数   2 宣言済み

        # ResNet18 モデルを作成する。
        model = ResnetBuilder.build_resnet_18(input_shape, num_classes)

        # モデルをコンパイルする。
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        # sparse_categorical_crossentropy
        # categorical_crossentropy
        ############################# model作成とcompile ####################################################







        ############################# 学習 ##################################################################
        # 学習する。
        model.fit(X_train, Y_train, epochs=5)
        ############################# 学習 ##################################################################





        ############################# test data save ########################################################
        print(Y_test.shape)
        # テスト実行直前のところで
        np.save('X_test_ReakyReLU_GazouBunrui_20200105.npy', X_test)
        np.save('Y_test_ReakyReLU_GazouBunrui_20200105.npy', Y_test)
        ############################# test data save ########################################################





        ############################# test data confirm confidence ##########################################
        # テストデータに対する性能を確認する。
        test_loss, test_acc = model.evaluate(X_test, Y_test)

        print(f"test loss: {test_loss:.2f}, test accuracy: {test_acc:.2%}")
        # test loss: 0.36, test accuracy: 86.93%
        ############################# test data confirm confidence ##########################################



        ############################# test data inference example ##########################################
        print(X_test.shape)
        print(Y_test.shape)

        # テストデータを推論する。
        predictions = model.predict(X_test)
        import matplotlib.pyplot as plt
        # test_images[0] の予測結果
        fig, ax = plt.subplots()
        # 次元数を減らす
        import numpy as np
        X_test_res_gen = np.squeeze(X_test)
        print(X_test_res_gen.shape)
        a = 4
        ax.imshow(X_test_res_gen[a], cmap="gray")
        #ax.imshow(X_test_res[50], cmap='gray')
        scoreappend = []
        pred = predictions[a]

        for name, score in zip(classes, pred):
            print("name:{0}: score{1:.2%}".format(name, score))
            scoreappend.append(score)
        ############################# test data inference example ##########################################





        ############################# learning model save ##################################################
        model.save('model_gazouwake_20200105.h5')
        ############################# learning model save ##################################################
        wx.MessageBox('学習が完了しました。学習モデルh5出力完了。')
    ############################ 学習 learning #######################################################################################



    ############################ 分類（推論） learning #######################################################################################
    def choose_h5_inference_method(self, event):
        """ choose_text_h5を選択し、targetフォルダにある画像をまず特徴抽出し、リスト化してから推論を一枚ずつしていき、分類していく """
        
        ######## まずはtarget画像群の特徴抽出 #######################################
        # import pyocr
        # import pyocr.builders
        # import cv2
        # from PIL import Image
        # Image.MAX_IMAGE_PIXELS = 1000000000
        # import sys
        # import os

        # # 以下のディレクトリに画像が配置してある
        # # work_folder_dir = newCurrentDirectory + "gazou_bunrui_wake\\1\\"
        # # 画像名をリスト化
        # list_hensuu = os.listdir(target_Directory)
        # if(list_hensuu == []):
        #     wx.MessageBox('targetフォルダに画像が入っていません。')
        #     self.OnExitApp(event)
        
        # # 初めに画像特徴量抽出メソッド定義
        # def gazou_feature_extruct(file_name):

        #     tools = pyocr.get_available_tools()

        #     if len(tools) == 0:
        #         print("No OCR tool found")
        #         sys.exit(1)

        #     tool = tools[0]

        #     #file_name = file_name_arg
        #     #file_dir = newCurrentDirectory + "gazou_bunrui_wake\\"
        #     #img1 = file_dir + "1\\" + file_name
        #     img1 = target_Directory + file_name
        #     res = tool.image_to_string(Image.open(img1),
        #                             lang="jpn",
        #                             builder=pyocr.builders.WordBoxBuilder(tesseract_layout=6))

        #     out = cv2.imread(img1)
        #     for d in res:
        #         # print(d.content)
        #         # print(d.position)
        #         #cv2.rectangle(out, d.position[0], d.position[1], (0, 0, 255), 2)
        #         # ##### 四角枠に色塗り #####
        #         # 小さすぎる、大きすぎる枠は除外
        #         menseki = (d.position[1][0] - d.position[0][0]) * (d.position[1][1] - d.position[0][1]) 
        #         if((menseki < 100) or ( 125000 < menseki)):
        #             continue
        #         # ##########################
        #         cv2.rectangle(out, (d.position[0][0], d.position[0][1]), (d.position[1][0], d.position[1][1]), (0, 0, 0), thickness=-1)

        #     # 学習用に画像を別途保存（フォルダをあらかじめ作っておかないと作成されない）
        #     #cv2.imwrite(file_dir + "gakusyu_moto2\\" + "1\\" + file_name, out)
        #     cv2.imwrite(target_learning_Directory + file_name, out)
        #     # cv2.imshow("img", out)
        #     cv2.waitKey(0)
        #     cv2.destroyAllWindows()

        # # リスト化された画像名をメソッドに投入
        # for list_hensuu_ko in list_hensuu:
        #     # 1枚ずつ特徴量を抽出していく
        #     gazou_feature_extruct(list_hensuu_ko)
        import cv2
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 1000000000
        import sys
        import os

        # 以下のディレクトリに画像が配置してある
        # work_folder_dir = newCurrentDirectory + "gazou_bunrui_wake\\1\\"
        # 画像名をリスト化
        list_hensuu = os.listdir(target_Directory)
        if(list_hensuu == []):
            wx.MessageBox('targetフォルダの中の1番フォルダに画像が入っていません。')
            self.OnExitApp(event)
        # 初めに画像特徴量抽出メソッド定義
        def gazou_feature_extruct(file_name):
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = 1000000000
            import cv2
            import os
            import numpy as np
            import pdb
            from threshold import apply_threshold
            from matplotlib import pylab as plt
            def square_detect_method(sdm_file):
                def create_if_not_exist(out_dir):
                    try:
                        if not os.path.exists(out_dir):
                            os.makedirs(out_dir)
                    except Exception as e:
                        print(e)


                output_path = 'out'
                create_if_not_exist(output_path)

                ######### gaucian etc #####################################
                currentdirectory = os.getcwd()
                # filename = '1.png'
                filename = sdm_file
                # PIL open for pixel up
                img1 = Image.open(target_Directory + filename)

                width = img1.size[0] * 10
                height = img1.size[1] * 10


                img2 = img1.resize((width, height))  # ex)(39, 374, 3) → (390, 3740, 3)
                img2.save(currentdirectory + '\\1600_' + filename)

                ####### white line making around #########################
                # OpenCV open for after process
                img2 = cv2.imread(currentdirectory + '\\1600_' + filename)

                # load image, change color spaces, and smoothing
                img_HSV = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
                os.remove(currentdirectory + '\\1600_' + filename)
                # cv2.imwrite(filename + '_4HSV.jpg', img_HSV)

                img_HSV = cv2.GaussianBlur(img_HSV, (9, 9), 3)
                # cv2.imwrite(filename + '_5GaussianBlur.jpg', img_HSV)

                # detect tulips
                img_H, img_S, img_V = cv2.split(img_HSV)
                # cv2.imwrite(filename + '_6splitH.jpg', img_H)
                # cv2.imwrite(filename + '_7splitS.jpg', img_S)
                # cv2.imwrite(filename + '_8splitV.jpg', img_V)
                _thre, img_flowers = cv2.threshold(img_V, 140, 255, cv2.THRESH_BINARY)
                
                # cv2.imwrite(filename + '_9mask.jpg', img_flowers)

                # img_flowers_copy = img_flowers.copy()

                # find tulips
                contours, hierarchy = cv2.findContours(
                    img_flowers, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  # labels,  delete


                ###### square to black################################################
                # out = cv2.imread(filename + '_9mask.jpg')

                for i in range(0, len(contours)):
                    if len(contours[i]) > 0:
                        print(cv2.contourArea(contours[i]))
                        # remove small objects
                        if cv2.contourArea(contours[i]) < 200:
                            continue
                        if cv2.contourArea(contours[i]) > 1000000:
                            continue
                        rect = contours[i]
                        x, y, w, h = cv2.boundingRect(rect)
                        cv2.rectangle(img_flowers, (x, y), (x + w, y + h), (0, 0, 0), -1)

                # squre in squre is bad
                # this point is black square
                # cv2.imwrite('./out' + '\\' + filename + '_9-2.jpg', img_flowers)

                # 学習用に画像を別途保存（フォルダをあらかじめ作っておかないと作成されない）
                #cv2.imwrite(file_dir + "gakusyu_moto2\\" + "1\\" + file_name, out)
                cv2.imwrite(target_learning_Directory +  filename + '_9-2.jpg', img_flowers)
                # cv2.imshow("img", out)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            square_detect_method(file_name)
        # リスト化された画像名をメソッドに投入
        for list_hensuu_ko in list_hensuu:
            gazou_feature_extruct(list_hensuu_ko)
        
        ######## まずはtarget画像群の特徴抽出 これでtarget_learningフォルダに画像が出力された#######################################


        ############################# target_learningフォルダにある特徴抽出後画像をリストに格納 #####################################################

        from PIL import Image
        Image.MAX_IMAGE_PIXELS = 1000000000
        import os,glob
        import numpy as np
        from sklearn import model_selection
        import glob
        import shutil
        # image_size = 128  ※ 1650, 2330に以下で指定
        image_size1 = 165
        image_size2 = 233

        #画像の読み込み
        #最終的に画像、ラベルはリストに格納される
        import os
        X = []
        Y = []
        files2 = glob.glob(target_learning_Directory + "/*") # ['C:\\Users\\user\\20190102\\image\\learning\\1\\out_0_70090154001001financial_statements1_dev_cure.jpg', 'C:\\Users\\user\\20190102\\image\\learning\\1\\out_0_70092582001001financial_statements1_dev_cure.jpg']
        # クラスの画像一覧を取得、それぞれのクラスの200枚までの画像を取得
        # file2 = D:/ProgramData/Anaconda3/envs/py37gpu_resnet/extract/一/ETL9G_01_000079.png
        for t, file2 in enumerate(files2):
            if(t >= 144): # 片方が30個なのでtが29まで来るので30以上でbreakに
                break
            # print(t)
            image = Image.open(file2)
            image = image.convert("RGB")
            # グレースケールに変換
            gray_image = image.convert('L')
            gray_image_resize = gray_image.resize((image_size1, image_size2)) # 画像の平均ピクセル縦横の値
            #イメージを1枚ずつnumpy配列に変換
            data = np.asarray(gray_image_resize)
            #リストに格納
            X.append(data)
        #格納したリストをさらにnumpy配列に変換
        X = np.array(X)
        print(X.shape)

        ############################# target_learningフォルダにある特徴抽出後画像をリストに格納 #####################################################




        filer = wx.FileDialog(self,
                            style=wx.DD_CHANGE_DIR,
                            message="学習モデルh5ファイル")
        if filer.ShowModal() == wx.ID_OK:
            self.filer = filer.GetPath()
        file_name_h5 = os.path.basename(filer.GetPath())
        filer.Destroy()
        #self.choose_text_h5.SetLabel(self.filer)
        #self.choose_text_h5.SetLabel(file_name_h5)
        choose_text_h5 = file_name_h5

        import numpy as np
        print("3")

        image_size1 = 165
        image_size2 = 233

        # input_shape = (127, 128, 1)  # モデルの入力サイズ
        input_shape = (image_size2, image_size1, 1)  # モデルの入力サイズ #   x  y　逆の場合も学習して結果要確認

        # num_classes = 72  # クラス数   2 宣言済み

        # ResNet18 モデルを作成する。
        model = ResnetBuilder.build_resnet_18(input_shape, 2)

        from keras.models import model_from_json

        #model.load_weights('model_gazouwake_20200104.h5')
        model.load_weights(str(choose_text_h5))

        # モデルをコンパイルする。
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        # sparse_categorical_crossentropy
        # categorical_crossentropy

        # X_test = np.load('X_test_ReakyReLU_GazouBunrui_20190906.npy')
        # Y_test = np.load('Y_test_ReakyReLU_GazouBunrui_20190906.npy')
        # # テストデータに対する性能を確認する。
        # test_loss, test_acc = model.evaluate(X_test, Y_test)

        # print(f"test loss: {test_loss:.2f}, test accuracy: {test_acc:.2%}")
        # test loss: 0.36, test accuracy: 86.93%

        # 次元数を増やす
        # ②2番目の位置に次元数を上げる
        X_dim = [np.expand_dims(x, axis=2) for x in X]
        # ③shapeみるためにnumpyへ戻す
        X_dim_np = np.asarray(X_dim)
        print(X_dim_np.shape)



        # テストデータを推論する。
        predictions = model.predict(X_dim_np)

        import matplotlib.pyplot as plt
        # test_images[0] の予測結果
        fig, ax = plt.subplots()
        # 次元数を減らす
        import numpy as np
        X_test_res_gen = np.squeeze(X_dim_np)

        # a = 0
        # # ax.imshow(X_test_res_gen[a], cmap="gray")
        # #ax.imshow(X_test_res[50], cmap='gray')
        # scoreappend = []
        # pred = predictions[a]
        # classes = []
        # classes.append(1)
        # classes.append(2)

        # for name, score in zip(classes, pred):
        #     print("name:{0}: score{1:.2%}".format(name, score))
        #     scoreappend.append(score)

        for list_hensuu_num, list_hensuu_child in enumerate(list_hensuu):

            a = int(list_hensuu_num)
            scoreappend = []
            pred = predictions[a]
            classes = []
            classes.append(1)
            classes.append(2)

            for name, score in zip(classes, pred):
                print("name:{0}: score{1:.2%}".format(name, score))
                scoreappend.append(score)

            if(pred[0] >= pred[1]):
                shutil.move(target_Directory + list_hensuu_child, result_Directory1)
            else:
                shutil.move(target_Directory + list_hensuu_child, result_Directory2)
        target_dir = target_learning_Directory
        shutil.rmtree(target_dir)
        # os.mkdir(target_dir)
        no_folder_make(target_learning_Directory)

    ############################ 分類（推論） learning #######################################################################################









    #######################################################################################



def main():
    app = wx.App(False)
    Main(None, wx.ID_ANY, "AI画像特徴抽出 分類器")
    app.MainLoop()
 
if __name__ == "__main__":
    main()