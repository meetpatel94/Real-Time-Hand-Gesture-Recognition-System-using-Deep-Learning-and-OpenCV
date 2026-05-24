**create virtual environment**
py -3.10 -m venv venv
venv\Scripts\activate

**packages**
pip install numpy==1.23.5
pip install opencv-python
pip install tensorflow==2.10.0
pip install matplotlib

**Verify**
pip list

**create dataset**
python create_dataset.py

**check images**
python check_images.py

**Model train**
python train_model.py

**Run code**
python predict_live.py

