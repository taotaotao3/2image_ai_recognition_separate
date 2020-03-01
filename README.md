# Image feature extraction classifier
![キャプチャ4](https://user-images.githubusercontent.com/20910951/75622794-8549ca00-5be7-11ea-8393-d9f0472951bc.PNG)

When there are two types of documents, each feature is first extracted, and then the features are learned and inferred, This classifies documents with over 98% accuracy.

## Description

Each document has its own characteristics. Text position, numeric position, line location, overall positional relationship.

In order to make them stand out, the characters etc. are painted black once. Then you can find the weight of the position of the whole character.

Subsequent training of the image itself allows for great separation.

## Constuction
Keras==2.2.0
tensorflow-gpu==1.6.0
termcolor==1.1.0
wxPython==4.0.7.post2

## Requirement
Please use requirements.txt

## Usage
1:$ pip install -r requirements.txt

2:Please put A-Pattern image in "before\1" Folder.

3:Please put B-Pattern image in "before\2" Folder.

4:python AI_Image_Separate.py

5:Click "Feature extraction" Button
(This results is making changing image in "learning\1" Folder & "learning\2" Folder)

6:Click "Learning" Button
(This results is making h5 File.

7:Click "Classification work started(learned model selection)

8:Please select your h5 File

## Licence

[MIT](https://github.com/tcnksm/tool/blob/master/LICENCE)

## A word
Please request any improvements that are better.
Now your document organization is left to artificial intelligence.
Now, get started!

