#!/bin/bash
set -e
VENV="tfg_env"
echo ""
echo "══════════════════════════════════════════════════"
echo "  TFG — M2 Pro + Metal GPU + numpy 2.0"
echo "══════════════════════════════════════════════════"

# Detectar Python
PY=$(command -v python3.11 || command -v python3)
echo "  Python: $($PY --version) | arch: $(uname -m)"

# Limpiar y crear venv
[ -d "$VENV" ] && rm -rf "$VENV"
$PY -m venv "$VENV"
source "$VENV/bin/activate"
pip install -U pip setuptools wheel -q

# Instalar en orden correcto
echo "  [1/3] numpy 2.0.2..."
pip install numpy==2.0.2 -q
echo "  [2/3] TensorFlow 2.18.1 + Metal 1.2.0..."
pip install tensorflow==2.18.1 tensorflow-metal==1.2.0 -q
echo "  [3/3] resto de dependencias..."
pip install pandas==2.2.3 scipy==1.14.1 scikit-learn==1.5.2 -q
pip install matplotlib==3.9.3 seaborn==0.13.2 tqdm==4.67.1 -q
pip install jupyter==1.1.1 ipykernel==6.29.5 notebook==7.3.2 -q

# Kernel Jupyter
python -m ipykernel install --user --name=tfg --display-name="TFG M2 Pro"

# Verificar
echo ""
python -c "
import numpy, tensorflow as tf, sklearn, pandas
print(f'  numpy:      {numpy.__version__}')
print(f'  tensorflow: {tf.__version__}')
print(f'  sklearn:    {sklearn.__version__}')
print(f'  pandas:     {pandas.__version__}')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'  Metal GPU:  {gpus[0].name}')
    with tf.device('/GPU:0'):
        tf.matmul(tf.random.normal([1000,1000]), tf.random.normal([1000,1000]))
    print(f'  Test Metal: OK')
else:
    print('  Metal GPU:  no detectada (CPU mode)')
"
echo ""
echo "  LISTO — Kernel: 'TFG M2 Pro'"
echo ""
