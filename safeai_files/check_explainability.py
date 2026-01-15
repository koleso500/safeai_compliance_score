import pandas as pd
from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from typing import Union
from safeai_files.utils import check_nan, convert_to_dataframe, find_yhat, manipulate_testdata, validate_variables
from xgboost import XGBClassifier, XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import auc
import random

from safeai_files.core import rga
from safeai_files.check_robustness import align_proba_to_class_order
from safeai_files.cramer import wrga_cramer, wrgr_cramer

def compute_rge_values(xtrain: pd.DataFrame, 
                xtest: pd.DataFrame,
                yhat: list,
                model: Union[XGBClassifier, XGBRegressor, BaseEstimator,
                torch.nn.Module],
                variables: list, 
                group: bool = False,
                metric: str = 'original'):
    """
    Helper function to compute the RGE values for given variables or groups of variables.

    Parameters
    ----------
    xtrain : pd.DataFrame
            A dataframe including train data.
    xtest : pd.DataFrame
            A dataframe including test data.
    yhat : list
            A list of predicted values.
    model : Union[XGBClassifier, XGBRegressor, BaseEstimator, torch.nn.Module]
            A trained model, which could be a classifier or regressor. 
    variables : list
            A list of variables.
    group : bool
            If True, calculate RGE for the group of variables as a whole; otherwise, calculate for each variable.
    metric: str
            'original': uses RGE
            'cramer': uses WRGE

    Returns
    -------
    pd.DataFrame
            The RGE values for each variable or for the group.
    """
    # Convert inputs to DataFrames and concatenate them
    xtrain, xtest, yhat = convert_to_dataframe(xtrain, xtest, yhat)
    # check for missing values
    check_nan(xtrain, xtest, yhat)
    # variables should be a list
    validate_variables(variables, xtrain)

    if metric not in ['original', 'cramer']:
        raise ValueError("Metric must be 'original' or 'cramer'")

    # find RGEs
    if group:
        # Apply manipulate_testdata iteratively for each variable in the group
        for variable in variables:
            xtest = manipulate_testdata(xtrain, xtest, variable)
        
        # Calculate yhat after manipulating all variables in the group
        yhat_rm = find_yhat(model, xtest)

        # Calculate a single RGE for the entire group except these variables
        if metric == "original":
            rge = rga(yhat, yhat_rm)
        else: # 'cramer'
            rge = wrga_cramer(yhat, yhat_rm)

        return pd.DataFrame([rge], index=[str(variables)], columns=['RGE'])

    else:
        # Calculate RGE for each variable individually
        rge_list = []
        for variable in variables:
            xtest_rm = manipulate_testdata(xtrain, xtest, variable)
            yhat_rm = find_yhat(model, xtest_rm)

            if metric == "original":
                rge_val = 1 - rga(yhat, yhat_rm)
            else:  # "cramer"
                rge_val = 1 - wrga_cramer(yhat, yhat_rm)

            rge_list.append(rge_val)
        
        return pd.DataFrame(rge_list, index=variables, columns=['RGE']).sort_values(by='RGE', ascending=False)


def wrge_cramer_multiclass(prob_full, prob_reduced, class_weights=None, verbose=False):
    """
    Calculate WRGE for multiclass classification.
    Measures impact of feature removal/occlusion on predictions.
    Use align_proba_to_class_order() before calling this function.

    Parameters
    ----------
    prob_full : array-like, shape (n_samples, n_classes)
        Predictions from original model
    prob_reduced : array-like, shape (n_samples, n_classes)
        Predictions from occluded model
    class_weights : array-like, optional
        Custom weights for each class. If None, uses uniform weighting.
    verbose : bool, optional
        Print detailed information

    Returns
    -------
    tuple
        (wrge_weighted, wrge_per_class, weights_used)
        - wrge_weighted: Overall weighted WRGE score
        - wrge_per_class: WRGE score for each class
        - weights_used: Weights used for each class
    """
    prob_full = np.asarray(prob_full)
    prob_reduced = np.asarray(prob_reduced)

    n_samples, n_classes = prob_full.shape

    if prob_reduced.shape != prob_full.shape:
        raise ValueError(
            f'Shape mismatch: prob_full {prob_full.shape} and prob_reduced {prob_reduced.shape}'
        )

    # Set up class weights
    if class_weights is None:
        class_weights = np.ones(n_classes) / n_classes
    else:
        class_weights = np.asarray(class_weights)
        if len(class_weights) != n_classes:
            raise ValueError(
                f'class_weights length {len(class_weights)} does not match n_classes {n_classes}'
            )

    wrges = []

    for k in range(n_classes):
        pred_full = prob_full[:, k]
        pred_reduced = prob_reduced[:, k]

        # WRGE uses same computation as WRGR
        wrge_k = wrgr_cramer(pred_full, pred_reduced)
        wrges.append(wrge_k)

        if verbose:
            print(f'Class {k}: WRGE = {wrge_k:.4f}')

    wrges = np.array(wrges)

    # Weighted average
    wrge_weighted = np.nansum(wrges * class_weights) / np.nansum(class_weights)

    return wrge_weighted, wrges, class_weights


# Linear Head
class ScaledLinearHead(nn.Module):
    """
    Linear head that optionally applies the same scaler as sklearn models.
    Scaling is in the forward pass so Grad-CAM path matches sklearn preprocessing.
    """
    def __init__(self, in_dim, n_classes, scaler=None, eps=1e-12):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)
        self.has_scaler = scaler is not None

        if self.has_scaler:
            mean = torch.tensor(scaler.mean_, dtype=torch.float32)
            scale = torch.tensor(scaler.scale_, dtype=torch.float32)
            # avoid division by zero
            scale = torch.clamp(scale, min=eps)
            self.register_buffer('mean', mean)
            self.register_buffer('scale', scale)

    def forward(self, feats):
        if self.has_scaler:
            feats = (feats - self.mean) / self.scale
        return self.linear(feats)


# CAM Model
class CAMModel(nn.Module):
    """
    Simple wrapper
    """
    def __init__(self, feature_extractor, head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, x):
        feats = self.feature_extractor(x)
        return self.head(feats)


# Grad-CAM
class GradCAM:
    """
    Grad-CAM for a CAMModel.

    By default, attempts to hook into `feature_extractor.layer4[-1].conv2` for ResNet18/34.
    Provide `target_layer` if different.
    """
    def __init__(self, cam_model, target_layer=None):
        self.model = cam_model

        if target_layer is None:
            fe = cam_model.feature_extractor
            if hasattr(fe, "layer4"):
                target_layer = fe.layer4[-1].conv2
            else:
                raise ValueError('Cannot auto-detect target layer. Provide target_layer')

        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._save_gradient)

    def close(self):
        if getattr(self, '_fwd_handle', None) is not None:
            self._fwd_handle.remove()
            self._fwd_handle = None

        if getattr(self, '_bwd_handle', None) is not None:
            self._bwd_handle.remove()
            self._bwd_handle = None

    def _save_activation(self, _module, _inp, out):
        self.activations = out

    def _save_gradient(self, _module, _grad_inp, grad_out):
        self.gradients = grad_out[0]

    @torch.no_grad()
    def predict_classes(self, images, device, batch_size=64):
        """
        Predict argmax class for each image.

        Parameters
        ----------
        images :
            Input tensor (N, C, H, W)
        device :
            Torch device
        batch_size :
            Batch size for forward passes

        Returns
        -------
        np.ndarray
            Predicted class
        """
        self.model.eval()
        preds = []
        for i in range(0, len(images), batch_size):
            x = images[i : i + batch_size].to(device, non_blocking=True)
            logits = self.model(x)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
        return np.concatenate(preds, axis=0)

    def cam_single(self, image, target_class=None, device=None):
        """
        Compute Grad-CAM for a single image.

        Parameters
        ----------
        image :
            Tensor of shape (C, H, W) or (1, C, H, W)
        target_class :
            Class to explain. If None, uses model argmax
        device :
            Device to run on. If None, inferred from model parameters

        Returns
        -------
        np.ndarray
            Normalized heatmap (H, W) in [0, 1]
        """
        if device is None:
            device = next(self.model.parameters()).device

        if image.dim() == 3:
            image = image.unsqueeze(0)

        image = image.to(device, non_blocking=True)
        image.requires_grad_(True)

        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.activations, self.gradients = None, None

        logits = self.model(image)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError('GradCAM hooks did not capture activations or gradients.')

        # Weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = nnf.relu(cam)

        cam = nnf.interpolate(cam, size=image.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        mn, mx = float(cam.min()), float(cam.max())
        if mx > mn:
            cam = (cam - mn) / (mx - mn)
        else:
            cam = np.zeros_like(cam)

        return cam


def train_cam_model(feature_extractor, images, labels, scaler=None,
                    n_classes=None, device=None,
                    epochs=15, lr=1e-3, batch_size=64, verbose=True):
    """
    Train a linear head (with optional scaler) on top of a frozen feature extractor.
    Uses true labels.

    Parameters
    ----------
    feature_extractor :
        Torch feature extractor (e.g., ResNet with removed classifier)
    images :
        Image tensor (N, C, H, W)
    labels :
        Class labels, length N
    scaler :
        sklearn-like StandardScaler to embed into the head forward pass
    n_classes :
        Number of classes. If None, inferred from unique labels
    device :
        Torch device. If None, inferred from feature_extractor parameters
    epochs, lr, batch_size :
        Training hyperparameters
    verbose :
        Print progress

    Returns
    -------
    CAMModel
        Frozen feature extractor and trained head
    """

    if device is None:
        device = next(feature_extractor.parameters()).device

    feature_extractor.eval().to(device)
    for p in feature_extractor.parameters():
        p.requires_grad_(False)

    labels = np.asarray(labels)
    if n_classes is None:
        n_classes = int(len(np.unique(labels)))

    # Extract features once
    if verbose:
        print('Extracting raw features for CAM training...')

    feats_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            x = images[i:i+batch_size].to(device, non_blocking=True)
            feats_list.append(feature_extractor(x).cpu().numpy())
    feats = np.vstack(feats_list)

    in_dim = feats.shape[1]
    head = ScaledLinearHead(in_dim, n_classes, scaler=scaler).to(device)
    cam_model = CAMModel(feature_extractor, head).to(device)

    x = torch.tensor(feats, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x, y),
        batch_size=batch_size, shuffle=True
    )

    opt = torch.optim.Adam(cam_model.head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    if verbose:
        print(f'Training CAM head for {epochs} epochs...')

    for ep in range(epochs):
        cam_model.head.train()
        tot_loss, correct, total = 0.0, 0, 0

        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = cam_model.head(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

            tot_loss += float(loss.item())
            pred = torch.argmax(logits, dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.size(0))

        if verbose and ((ep + 1) % 5 == 0 or ep == epochs - 1):
            print(f'Epoch {ep+1:02d}/{epochs}: loss={tot_loss/len(loader):.4f}, acc={100*correct/total:.2f}%')

    cam_model.eval()
    return cam_model


# Image Blur
def blur_images_gaussian(images, ksize=31, sigma=7.0):
    """
    Applies Gaussian blur to a batch of images (N, C, H, W) using separable conv.

    Parameters
    ----------
    images :
        Input images tensor (N, C, H, W)
    ksize :
        Kernel size. If even, it will be incremented by 1 to keep it odd
    sigma :
        Gaussian sigma

    Returns
    -------
    torch.Tensor
        Blurred images, same shape as input
    """
    if ksize % 2 == 0:
        ksize += 1

    device = images.device
    dtype = images.dtype

    # Gaussian kernel
    x = torch.arange(ksize, device=device, dtype=dtype) - (ksize - 1) / 2.0
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()

    # Separable kernels
    g_x = g.view(1, 1, 1, ksize).repeat(images.shape[1], 1, 1, 1)
    g_y = g.view(1, 1, ksize, 1).repeat(images.shape[1], 1, 1, 1)

    pad = ksize // 2
    out = nnf.conv2d(images, g_x, padding=(0, pad), groups=images.shape[1])
    out = nnf.conv2d(out, g_y, padding=(pad, 0), groups=images.shape[1])
    return out


def compute_gradcam_maps(images, cam_model, device=None, batch_pred=64, verbose=True):
    """
    Compute Grad-CAM importance maps for a batch of images.

    Parameters
    ----------
    images :
        Image tensor (N, C, H, W)
    cam_model :
        CAMModel used for Grad-CAM
    device :
        Torch device. If None, inferred from cam_model parameters
    batch_pred :
        Batch size used for predicting target classes
    verbose :
        Print progress

    Returns
    -------
    np.ndarray
        Importance maps, shape (N, H, W), dtype float32.
    """
    if device is None:
        device = next(cam_model.parameters()).device

    gradcam = GradCAM(cam_model)

    if verbose:
        print('Predicting target classes for Grad-CAM...')
    targets = gradcam.predict_classes(images, device=device, batch_size=batch_pred)

    if verbose:
        print('Computing Grad-CAM maps...')
    maps = []
    for i in range(len(images)):
        maps.append(gradcam.cam_single(images[i:i+1], target_class=int(targets[i]), device=device))
        if verbose and (i + 1) % 100 == 0:
            print(f'{i+1}/{len(images)} maps')

    gradcam.close()
    return np.asarray(maps, dtype=np.float32)


# Patches rankings
def precompute_patch_rankings(importance_maps, patch_size=32):
    """
    Convert per-pixel importance maps into per-image patch rankings.

    Parameters
    ----------
    importance_maps :
        Array of shape (N, H, W)
    patch_size:
        Size of square patches

    Returns
    -------
    rankings :
        List of length N with arrays of patch indices sorted by descending importance
    meta :
        PatchMeta describing the grid
    """
    n, h, w = importance_maps.shape
    n_ph = h // patch_size
    n_pw = w // patch_size

    patch_coords = []
    for ph in range(n_ph):
        for pw in range(n_pw):
            y0 = ph * patch_size
            x0 = pw * patch_size
            y1 = min(y0 + patch_size, h)
            x1 = min(x0 + patch_size, w)
            patch_coords.append((y0, y1, x0, x1))

    rankings = []
    for i in range(n):
        imp = importance_maps[i]
        scores = np.array([imp[y0:y1, x0:x1].mean() for (y0, y1, x0, x1) in patch_coords], dtype=np.float32)
        rankings.append(np.argsort(scores)[::-1])

    meta = {
        "patch_size": patch_size,
        "patch_coords": patch_coords,
        "total_patches": len(patch_coords),
        "n_patches_h": n_ph,
        "n_patches_w": n_pw,
    }
    return rankings, meta


# Wise masking
def apply_importance_masking(images, patch_rankings, patch_meta, fraction_to_mask,
                             mask_strategy="most_important",
                             mask_value=0.0,
                             baseline="constant",
                             blur_ksize=31, blur_sigma=7.0):
    """
    Mask a fraction of the image area using patch importance rankings.

    Parameters
    ----------
    images :
        Tensor (N, C, H, W)
    patch_rankings :
        Per-image arrays of patch indices sorted by descending patch importance
    patch_meta :
        Produced by `precompute_patch_rankings`
    fraction_to_mask :
        Fraction of total pixels to mask in [0, 1]
    mask_strategy :
        'most_important' masks top-ranked patches, supports adding new strategies later
    mask_value :
        Constant value used if baseline="constant"
    baseline :
        - 'constant': fill masked area with mask_value
        - 'blur': replace masked area with blurred content
    blur_ksize, blur_sigma :
        Parameters for Gaussian blur baseline

    Returns
    -------
    torch.Tensor
        Masked images, same shape as input
    """
    out = images.clone()
    n, c, h, w = out.shape

    if baseline == 'blur':
        blurred = blur_images_gaussian(images, ksize=blur_ksize, sigma=blur_sigma)
    else:
        blurred = None

    patch_size = patch_meta['patch_size']
    patch_pixels = patch_size * patch_size
    total_pixels = h * w

    pixels_to_mask = int(fraction_to_mask * total_pixels)
    k = pixels_to_mask // patch_pixels
    k = min(k, patch_meta['total_patches'])
    if k <= 0:
        return out

    coords = patch_meta['patch_coords']

    for i in range(n):
        order = patch_rankings[i]
        if mask_strategy == 'most_important':
            chosen = order[:k]
        else:
            raise ValueError(f'Unknown mask_strategy: {mask_strategy}')

        for idx in chosen:
            y0, y1, x0, x1 = coords[int(idx)]
            if baseline == 'blur':
                out[i, :, y0:y1, x0:x1] = blurred[i, :, y0:y1, x0:x1]
            else:
                out[i, :, y0:y1, x0:x1] = mask_value

    return out


# Second option with random masking
def apply_patch_occlusion(images, num_patches, patch_size=32, random_seed=None,
                          mask_value=0.0,
                          baseline="constant",
                          blur_ksize=31, blur_sigma=7.0):
    """
    Random patch masking.

    Parameters
    ----------
    images :
        Tensor (N, C, H, W)
    num_patches :
        Number of random patches to mask per image
    patch_size :
        Square patch size
    random_seed :
        If provided, seeds torch and numpy for reproducibility
    mask_value :
        Constant fill value used when baseline='constant'
    baseline :
        - 'constant': fill masked area with mask_value
        - 'blur': replace masked area with blurred content
    blur_ksize, blur_sigma:
        Parameters for Gaussian blur baseline

    Returns
    -------
    torch.Tensor
        Masked images, same shape as input
    """
    if random_seed is not None:
        random.seed(random_seed)
        torch.manual_seed(random_seed)

    out = images.clone()
    n, c, h, w = out.shape
    if num_patches <= 0:
        return out

    if baseline == 'blur':
        blurred = blur_images_gaussian(images, ksize=blur_ksize, sigma=blur_sigma)
    else:
        blurred = None

    for i in range(n):
        for _ in range(num_patches):
            y0 = random.randint(0, h - patch_size)
            x0 = random.randint(0, w - patch_size)
            if baseline == 'blur':
                out[i, :, y0:y0+patch_size, x0:x0+patch_size] = blurred[i, :, y0:y0+patch_size, x0:x0+patch_size]
            else:
                out[i, :, y0:y0+patch_size, x0:x0+patch_size] = mask_value
    return out


# Features extraction
def extract_features_from_images(images, feature_extractor, pca=None, scaler=None,
                                 device=None, batch_size=64):
    """
    Extract features from images using a torch feature extractor, optionally apply PCA and scaling.

    Parameters
    ----------
    images :
        Tensor (N, C, H, W)
    feature_extractor :
        Torch module mapping images -> features (N, D)
    pca :
        sklearn-like object with `.transform(x)` or None
    scaler :
        sklearn-like object with `.transform(x)` or None
    device :
        Device for feature extraction. If None, inferred from feature_extractor
    batch_size :
        Batch size for extraction

    Returns
    -------
    np.ndarray
        Feature matrix after optional PCA and scaling
    """

    feature_extractor.eval()
    if device is None:
        device = next(feature_extractor.parameters()).device

    feats_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size].to(device, non_blocking=True)
            feats_list.append(feature_extractor(batch).cpu().numpy())
    x = np.vstack(feats_list)

    if pca is not None:
        x = pca.transform(x)
    if scaler is not None:
        x = scaler.transform(x)

    return x


# Predictions from features
def get_predictions_from_features(features, model, model_class_order, class_order,
                                  model_type="sklearn", device=None, batch_size=64):
    """
    Get class probabilities from a model given feature vectors and align them to `class_order`.

    Parameters
    ----------
    features :
        Feature matrix (N, D)
    model :
        sklearn model with predict_proba or torch module producing logits
    model_class_order :
        Class labels order as produced by the model (e.g., sklearn `model.classes_`)
    class_order :
        Desired canonical class order
    model_type :
        'sklearn' or 'pytorch'
    device :
        Torch device required when model_type='pytorch'
    batch_size :
        Batch size for torch inference

    Returns
    -------
    np.ndarray
        Probabilities aligned to `class_order`, shape (N, n_classes)
    """
    if model_type == 'sklearn':
        probs = model.predict_proba(features)

    elif model_type == 'pytorch':
        model.eval()
        probs_list = []
        with torch.no_grad():
            for i in range(0, len(features), batch_size):
                batch = torch.tensor(features[i:i+batch_size], dtype=torch.float32).to(device)
                logits = model(batch)
                probs_list.append(torch.softmax(logits, dim=1).cpu().numpy())
        probs = np.vstack(probs_list)

    else:
        raise ValueError(f"model_type must be 'sklearn' or 'pytorch', got {model_type}")

    return align_proba_to_class_order(probs, model_class_order, class_order)


# RGE evaluation
def evaluate_wrge_multiclass_occlusion(
    model, preprocess_fn, images_dataset, removal_fractions,
    model_class_order, class_order,
    model_type='sklearn', device=None,
    patch_size=32, batch_size=64,
    class_weights=None, model_name='Model', wrga_full=None,
    occlusion_method='random', patch_rankings=None, patch_meta=None,
    plot=True, fig_size=(10, 6), verbose=True,
    random_seed=None, mask_value=0.0
):
    """
    Evaluate WRGE across increasing occlusion levels and compute AURGE.

    Parameters
    ----------
    model :
        The classifier to evaluate (sklearn or torch)
    preprocess_fn :
        Callable mapping images tensor (N,C,H,W) -> features ndarray (N,D) ready for `model`.
        This is where you typically call: feature extractor -> PCA -> scaler.
    images_dataset :
        Torch dataset yielding images and possibly labels
    removal_fractions :
        Fractions of image area to occlude in [0,1]
    model_class_order :
        Model's class order (e.g. sklearn model.classes_)
    class_order :
        Canonical class order
    model_type :
        'sklearn' or 'pytorch'
    device :
        Torch device
    patch_size :
        Patch size for random occlusion or importance patching
    batch_size :
        Batch size for loading dataset and for prediction
    class_weights :
        Optional weights for WRGE aggregation
    model_name :
        Name used for logging and plots
    wrga_full :
        If provided, WRGE curve is rescaled by this value (required by SAFE)
    occlusion_method :
        'random' or 'gradcam_most'
    patch_rankings, patch_meta :
        Required when occlusion_method is gradcam_
    plot :
        Whether to plot the WRGE curve
    fig_size :
        Figure size for plotting
    verbose :
        Verbose logging
    random_seed :
        Seed used for random occlusion
    mask_value :
        Fill value for masked pixels when using constant baseline

    Returns
    -------
    dict
        Contains raw/rescaled WRGE values, AURGE, per-class WRGE, and metadata.
    """
    removal_fractions = np.asarray(removal_fractions, dtype=float)

    if occlusion_method in ('gradcam_most', 'gradcam_least'):
        if patch_rankings is None or patch_meta is None:
            raise ValueError('For Grad-CAM masking you must pass patch_rankings and patch_meta')

    if verbose:
        print(f'RGE Evaluation: {model_name}')
        print(f'Occlusion: {occlusion_method}')
        print(f'Testing {len(removal_fractions)} removal fractions')

    # Load all images once
    loader = DataLoader(images_dataset, batch_size=batch_size, shuffle=False)
    images_all = []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        images_all.append(x)
    images_all = torch.cat(images_all, dim=0)

    _, _, h, w = images_all.shape
    total_pixels = h * w
    patch_pixels = patch_size * patch_size

    # Baseline predictions
    if verbose:
        print('Extracting features from original images...')
    feat_full = preprocess_fn(images_all)
    prob_full = get_predictions_from_features(
        feat_full, model, model_class_order, class_order,
        model_type=model_type, device=device, batch_size=batch_size
    )

    wrge_scores = []
    per_class_wrge_list = []

    for frac in removal_fractions:
        if verbose:
            print(f'\nOcclusion level: {frac*100:.0f}%')

        if occlusion_method == 'random':
            pixels_to_remove = int(frac * total_pixels)
            num_patches = pixels_to_remove // patch_pixels
            images_occ = apply_patch_occlusion(
                images_all, num_patches, patch_size,
                random_seed=random_seed, mask_value=mask_value
            )

        elif occlusion_method == 'gradcam_most':
            images_occ = apply_importance_masking(
                images_all, patch_rankings, patch_meta, frac,
                mask_strategy='most_important', mask_value=mask_value
            )

        else:
            raise ValueError(f'Unknown occlusion_method: {occlusion_method}')

        feat_occ = preprocess_fn(images_occ)
        prob_occ = get_predictions_from_features(
            feat_occ, model, model_class_order, class_order,
            model_type=model_type, device=device, batch_size=batch_size
        )

        wrge_val, wrge_per_class, _ = wrge_cramer_multiclass(prob_full, prob_occ, class_weights=class_weights)
        wrge_val = 0.0 if np.isnan(wrge_val) else float(wrge_val)

        wrge_scores.append(wrge_val)
        per_class_wrge_list.append(wrge_per_class)

        if verbose:
            print(f'RGE = {wrge_val:.4f}')

    wrge_scores = np.asarray(wrge_scores, dtype=float)
    per_class_wrge_list = np.asarray(per_class_wrge_list)

    # Rescale by WRGA
    wrge_rescaled = wrge_scores * float(wrga_full) if (wrga_full is not None and np.isfinite(wrga_full)) else wrge_scores

    # AUC on normalized x-axis
    max_frac = float(np.max(removal_fractions)) if len(removal_fractions) else 1.0
    x = removal_fractions / max_frac if max_frac > 0 else removal_fractions
    aurge = auc(x, wrge_rescaled)

    if verbose:
        print(f'AURGE: {aurge:.4f}')

    if plot:
        plt.figure(figsize=fig_size)
        plt.plot(removal_fractions * 100, wrge_rescaled, '-o', linewidth=2.5, markersize=6)
        plt.fill_between(removal_fractions * 100, 0, wrge_rescaled, alpha=0.2)
        plt.xlabel('Occluded Image Area (%)', fontsize=11, fontweight='bold')
        plt.ylabel('RGE Score', fontsize=11, fontweight='bold')
        plt.title(f'RGE Curve: {model_name} ({occlusion_method})', fontsize=12, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.show()

    return {
        'wrge_scores': wrge_scores,
        'wrge_rescaled': wrge_rescaled,
        'aurge': aurge,
        'removal_fractions': removal_fractions,
        'per_class_wrge': per_class_wrge_list,
        'class_order': class_order,
        'occlusion_method': occlusion_method,
    }


def compare_models_wrge(
    models_dict, images_dataset, removal_fractions, class_order,
    occlusion_method='random',
    patch_size=32, batch_size=64, class_weights=None,
    wrga_dict=None, device=None, fig_size=(12, 6), verbose=True,
    random_seed=None, patch_rankings=None, patch_meta=None
):
    """
    Evaluate and plot WRGE curves for multiple models.

    Parameters
    ----------
    models_dict :
        Mapping model_name -> (model, preprocess_fn, model_class_order, model_type)
    images_dataset :
        Dataset images
    removal_fractions :
        Occlusion fractions in [0,1]
    class_order :
        Canonical class order.
    occlusion_method :
        Single method for all models OR per-model dict
    patch_size :
        Patch size for random occlusion or importance patching
    batch_size :
        Batch size for loading dataset and for prediction
    class_weights :
        Optional weights for WRGE aggregation
    wrga_dict :
        Needed for rescaling
    device :
        Torch device
    fig_size :
        Figure size for plotting
    verbose :
        Verbose logging
    random_seed :
        Seed used for random occlusion
    patch_rankings, patch_meta :
        Shared Grad-CAM patch ranking info (compute once) for gradcam_ methods

    Returns
    -------
    dict
        Results per model name
    """
    if isinstance(occlusion_method, str):
        methods = {name: occlusion_method for name in models_dict}
    elif isinstance(occlusion_method, dict):
        methods = occlusion_method
    else:
        raise TypeError(
            'occlusion_method must be a string (single method) or a dict {model_name: method}.'
        )

    results = {}

    for name, (model, preprocess_fn, model_class_order, model_type) in models_dict.items():
        if verbose:
            print(f'\nEvaluating {name}')

        res = evaluate_wrge_multiclass_occlusion(
            model=model,
            preprocess_fn=preprocess_fn,
            images_dataset=images_dataset,
            removal_fractions=removal_fractions,
            model_class_order=model_class_order,
            class_order=class_order,
            model_type=model_type,
            device=device,
            patch_size=patch_size,
            batch_size=batch_size,
            class_weights=class_weights,
            model_name=name,
            wrga_full=(wrga_dict.get(name) if wrga_dict else None),
            occlusion_method=methods.get(name, 'random'),
            patch_rankings=patch_rankings,
            patch_meta=patch_meta,
            plot=False,
            verbose=verbose,
            random_seed=random_seed,
        )
        results[name] = res

    # Plot comparison.
    plt.figure(figsize=fig_size)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, len(results)))

    for (name, res), col in zip(results.items(), colors):
        plt.plot(
            res['removal_fractions'] * 100,
            res['wrge_rescaled'],
            '-o',
            linewidth=2.2,
            markersize=5,
            color=col,
            label=f"{name} [{res['occlusion_method']}] (AURGE={res['aurge']:.3f})",
        )

    plt.xlabel('Occluded Image Area (%)', fontsize=11, fontweight='bold')
    plt.ylabel('RGE Score', fontsize=11, fontweight='bold')
    plt.title('RGE Curves Comparison', fontsize=12, fontweight='bold')
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

    if verbose:
        print('\nExplainability Comparison Summary (AURGE)')
        for name in results:
            print(f"{name:15s}: AURGE={results[name]['aurge']:.4f}")

    return results

