# Proyecto Final APBF - VAE 3D para CMR

Clasificacion de patologias cardiacas (NOR / DCM / HCM / RV) a partir de imagenes de resonancia magnetica cardiaca (CMR) en formato NIfTI (`.nii.gz`), usando un Variational Autoencoder 3D.

---

## Pipeline general

```
NIfTI (.nii.gz)
      |
      v
preprocessing.py          resampleo -> [0,1] -> crop/pad -> z-score
      |
      v
dataset.py                CardiacFrameDataset  (indice de frames, splits)
      |
      |--> train.py       entrena el VAE sin etiquetas (reconstruccion)
      |         |
      |         v
      |    checkpoints/best.pt
      |
      |--> encode_all.py  codifica todos los frames -> Z.npy + metadata.csv
      |         |
      |         v
      |    analysis.py    PCA / UMAP / silhouette sobre el espacio latente
      |
      |--> train_classifier.py   clasificador sobre encoder congelado
      |--> train_enc_clf.py      baseline: encoder entrenable sin decoder
      +--> train_joint.py        entrenamiento conjunto semi-supervisado
```

---

## Estructura del proyecto

```
Proyecto_Final_APBF/
|-- src/
|   |-- preprocessing.py
|   |-- preprocess_dataset.py
|   |-- dataset.py
|   |-- model.py
|   |-- train.py
|   |-- encode_all.py
|   |-- analysis.py
|   |-- plot_recon.py
|   |-- train_classifier.py
|   |-- train_enc_clf.py
|   +-- train_joint.py
|-- notebooks/
|   |-- preprocesamiento.ipynb
|   +-- arquitecturas.ipynb
+-- presentacion_final/
    +-- APBF.pdf
```

---

## `src/`

### `preprocessing.py`
Pipeline de preprocesamiento para un unico volumen NIfTI. Se aplica frame a frame:

1. **Resampleo** al spacing fisico fijo (1.5 mm x 1.5 mm x 10 mm) con `scipy.ndimage.zoom`.
2. **Clip p1-p99 -> [0, 1]** antes del padding, de modo que el fondo real quede en aprox. 0.
3. **Center crop / pad** a `(224, 224, 10)` con ceros (el padding con ceros coincide con el fondo).
4. **Z-score** sobre el volumen resultante.

Devuelve una lista de tensores `(1, 224, 224, 10)`, uno por frame temporal. 

---

### `preprocess_dataset.py`
Preprocesa todos los pacientes del dataset una sola vez y guarda cada frame como tensor `.pt`.

---

### `dataset.py`
Define el `Dataset` de PyTorch y las funciones de split.

- **`list_patients()`** - lista todos los `.nii.gz` del dataset.
- **`split_patients_stratified()`** - split 60-20-20 estratificado por `(dataset, patologia)` usando un CSV de datos.
- **`CardiacFrameDataset`** - dataset plano de frames individuales. Soporta carga desde tensores `.pt` preprocesados.

---

### `model.py`
Arquitecturas del VAE y funciones de loss.

#### Arquitecturas

| Clase | Encoder | Decoder | Latente |
|---|---|---|---|
| `VAE3D` | 2 bloques Conv3d stride-4, kernel-5 | 2 bloques ConvTranspose3d | `(lat_ch, 14, 14, 1)` |
| `VAE3DDeep` | 4 bloques Conv3d stride-2, kernel-3 | 4 bloques ConvTranspose3d | `(lat_ch, 14, 14, 1)` |

#### Funciones de loss

- **`vae_loss()`** - ELBO: MSE + beta*KL, con terminos opcionales SSIM.

---

### `train.py`
Entrenamiento del VAE (solo reconstruccion, sin etiquetas de patologia).

Caracteristicas:
- Soporta las arquitecturas: `v1` (VAE3D), `deep` (VAE3DDeep).
- beta-warmup lineal.
- Logging con TensorBoard / CSV.
- Checkpointing atomico (`latest.pt` y `best.pt` por minimo de val loss).
- Resume desde checkpoint con `--resume`.

---

### `encode_all.py`
Carga el VAE entrenado y codifica todos los frames del dataset (o un split especifico). Guarda:

- `Z.npy` - array `(N_frames, latent_dim)`.
- `metadata.csv` - columnas `patient, frame_idx, dataset, path`.

---

### `analysis.py`
Analisis del espacio latente a partir de `Z.npy` + `metadata.csv` producidos por `encode_all.py`.

Genera:
- Scatter PCA 2D coloreado por patologia y por dataset.
- Scatter UMAP 2D.
- Trayectorias temporales `z(t)` por paciente en el embedding 2D.
- Silhouette scores por frame y por paciente (media).

---

### `plot_recon.py`
Visualizacion cualitativa de reconstrucciones. Para cada patologia (NOR / DCM / HCM / RV) elige un paciente y muestra 4 filas: original ED, reconstruido ED, original ES, reconstruido ES (slice central z=5).

---

### `train_classifier.py`
Entrena una cabeza clasificadora sobre el encoder **congelado** del VAE.

**Clases:** NOR, DCM, HCM, RV.  
**Entrada:** concatenacion de los latentes ED y ES de cada paciente -> `(lat_chx2, 14, 14, 1)`.

Tres cabezas disponibles (argumento `--classifier-arch`):

| Clase | Descripcion |
|---|---|
| `CardiacClassifierConvMLP` | Conv3d + AdaptiveAvgPool + MLP |
| `CardiacClassifierMLP` | Solo pool global + MLP |
| `CardiacClassifierFlatMLP` | Flatten completo del latente + MLP |

Guarda `best_classifier.pt`.

```bash
python src/train_classifier.py --checkpoint checkpoints/best.pt \
    --split-csv checkpoints/split.csv \
    --pathology-csv <ruta_csv> \
    --preprocessed-dir <ruta_pt> \
    --output-dir checkpoints_classifier
```

---

### `train_enc_clf.py`
**Baseline supervisado puro.** Entrena Encoder + Clasificador con CrossEntropy, sin decoder ni loss VAE. El encoder es entrenable (no congelado). Sirve para comparar si el preentrenamiento VAE aporta valor.

Guarda `best.pt`.

---

### `train_joint.py`
**Entrenamiento conjunto semi-supervisado.** Combina loss VAE y clasificacion:

```
Loss total = L_VAE(ED) + L_VAE(ES) + lambda*CrossEntropy    [frames etiquetados]
           + mu*L_VAE(frame)                              [frames sin etiqueta]
```

Puede partir de un VAE preentrenado (`--checkpoint`) o entrenar desde cero.  
Guarda `best.pt` con `{"vae": ..., "clf": ...}`.

---

## `notebooks/`

### `preprocesamiento.ipynb`
Muestra el pipeline de preprocesamiento paso a paso sobre un paciente real:
paso 0 (original) -> paso 1 (resampleo) -> paso 2 ([0,1]) -> paso 3 (crop/pad) -> paso 4 (z-score).
Incluye visualizacion de todos los slices en cada paso. Esto es lo que hace `preprocessing.py`.

### `arquitecturas.ipynb`
Muestra las arquitecturas del VAE y los clasificadores.

---

## `presentacion_final/`

### `APBF.pdf`
Presentacion final del proyecto.

---

## Clases de patologia

| Codigo | Patologia |
|---|---|
| NOR | Normal |
| DCM | Miocardiopatia dilatada |
| HCM | Miocardiopatia hipertrofica |
| RV | Ventriculo derecho anormal |

