# conda create --name pytorch3d python=3.8
# conda activate pytorch3d
conda install pytorch=1.10.0 torchvision torchaudio cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c pytorch3d pytorch3d
conda install numpy pillow matplotlib
pip install opencv-python

pip install pytorch-lightning==1.5.0
pip install -r requirements_template.txt

pip install setuptools==59.5.0
pip install pandas
pip install imageio
pip install trimesh
pip install h5py
pip install einops
pip install seaborn
pip install scikit-learn
