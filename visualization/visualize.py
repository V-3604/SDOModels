import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import torch
from captum.attr import GradientShap, IntegratedGradients, Occlusion
from captum.attr import visualization as viz
import cv2

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def create_attention_heatmap(attention_weights, title, save_path):
    """
    Create heatmap visualization of attention weights
    
    Args:
        attention_weights: Attention weights tensor
        title: Plot title
        save_path: Path to save visualization
    """
    # Convert to numpy
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, cmap='viridis', cbar=True)
    plt.title(title)
    plt.savefig(save_path)
    plt.close()


def visualize_attributions(magnetogram, euv, attributions, timestep_idx, save_dir):
    """
    Visualize attributions from model interpretability methods
    
    Args:
        magnetogram: Magnetogram input tensor (B, T, C, H, W)
        euv: EUV input tensor (B, T, C, H, W)
        attributions: Dictionary of attributions for different inputs
        timestep_idx: Index of timestep to visualize
        save_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get batch size, timesteps, channels, height, width
    batch_size, time_steps, mag_channels, height, width = magnetogram.shape
    _, _, euv_channels, _, _ = euv.shape
    
    # Select sample from batch (assuming first sample)
    sample_idx = 0
    
    # Visualize magnetogram attributions
    if 'magnetogram' in attributions:
        mag_attr = attributions['magnetogram'][sample_idx, timestep_idx].sum(dim=0)
        mag_input = magnetogram[sample_idx, timestep_idx, 0].cpu().numpy()
        
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(mag_input, cmap='gray')
        plt.colorbar()
        plt.title(f'Magnetogram Input (t={timestep_idx})')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mag_attr.cpu().numpy(), cmap='coolwarm', vmin=-abs(mag_attr).max().item(), vmax=abs(mag_attr).max().item())
        plt.colorbar()
        plt.title(f'Magnetogram Attribution (t={timestep_idx})')
        
        plt.subplot(1, 3, 3)
        plt.imshow(mag_input, cmap='gray')
        plt.imshow(mag_attr.cpu().numpy(), cmap='coolwarm', alpha=0.5, vmin=-abs(mag_attr).max().item(), vmax=abs(mag_attr).max().item())
        plt.colorbar()
        plt.title(f'Magnetogram Overlay (t={timestep_idx})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'magnetogram_attribution_t{timestep_idx}.png'))
        plt.close()
    
    # Visualize EUV attributions
    if 'euv' in attributions:
        # Use a selection of important EUV channels (94Å and 211Å)
        euv_channels_to_visualize = [0, 4]  # Assuming indices for 94Å and 211Å channels
        wavelengths = [94, 211]  # Corresponding wavelengths
        
        for i, (ch_idx, wavelength) in enumerate(zip(euv_channels_to_visualize, wavelengths)):
            euv_attr = attributions['euv'][sample_idx, timestep_idx, ch_idx].cpu().numpy()
            euv_input = euv[sample_idx, timestep_idx, ch_idx].cpu().numpy()
            
            plt.figure(figsize=(18, 6))
            plt.subplot(1, 3, 1)
            plt.imshow(euv_input, cmap='viridis')
            plt.colorbar()
            plt.title(f'EUV {wavelength}Å Input (t={timestep_idx})')
            
            plt.subplot(1, 3, 2)
            plt.imshow(euv_attr, cmap='coolwarm', vmin=-abs(euv_attr).max(), vmax=abs(euv_attr).max())
            plt.colorbar()
            plt.title(f'EUV {wavelength}Å Attribution (t={timestep_idx})')
            
            plt.subplot(1, 3, 3)
            plt.imshow(euv_input, cmap='viridis')
            plt.imshow(euv_attr, cmap='coolwarm', alpha=0.5, vmin=-abs(euv_attr).max(), vmax=abs(euv_attr).max())
            plt.colorbar()
            plt.title(f'EUV {wavelength}Å Overlay (t={timestep_idx})')
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'euv_{wavelength}A_attribution_t{timestep_idx}.png'))
            plt.close()


def visualize_multimodal_comparison(magnetogram, euv, predictions, targets, timestep_idx, save_path):
    """
    Create a comprehensive visualization comparing multimodal inputs and model predictions
    
    Args:
        magnetogram: Magnetogram input tensor (B, T, C, H, W)
        euv: EUV input tensor (B, T, C, H, W)
        predictions: Model predictions dictionary
        targets: Target values dictionary
        timestep_idx: Index of timestep to visualize
        save_path: Path to save visualization
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Select sample from batch (assuming first sample)
    sample_idx = 0
    
    # Get important EUV channels
    euv_94 = euv[sample_idx, timestep_idx, 0].cpu().numpy()  # Assuming 94Å is first channel
    euv_211 = euv[sample_idx, timestep_idx, 4].cpu().numpy()  # Assuming 211Å is fifth channel
    
    # Get magnetogram
    mag = magnetogram[sample_idx, timestep_idx, 0].cpu().numpy()
    
    # Get predictions and targets
    peak_flux_pred = predictions['peak_flux_mean'][sample_idx].item() if 'peak_flux_mean' in predictions else predictions['peak_flux'][sample_idx].item()
    peak_flux_true = targets['peak_flux'][sample_idx].item()
    
    # Create figure with gridspec
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # Magnetogram
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(mag, cmap='gray')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Magnetogram')
    ax1.axis('off')
    
    # EUV 94Å
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(euv_94, cmap='viridis')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('EUV 94Å')
    ax2.axis('off')
    
    # EUV 211Å
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(euv_211, cmap='plasma')
    plt.colorbar(im3, ax=ax3)
    ax3.set_title('EUV 211Å')
    ax3.axis('off')
    
    # Overlay magnetogram contours on EUV images
    ax4 = fig.add_subplot(gs[1, 0:3])
    
    # Create RGB image
    rgb = np.zeros((mag.shape[0], mag.shape[1], 3))
    
    # Normalize each channel to [0, 1]
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    euv_94_norm = (euv_94 - euv_94.min()) / (euv_94.max() - euv_94.min() + 1e-8)
    euv_211_norm = (euv_211 - euv_211.min()) / (euv_211.max() - euv_211.min() + 1e-8)
    
    # Assign channels (red: magnetogram, green: EUV 94Å, blue: EUV 211Å)
    rgb[:, :, 0] = mag_norm
    rgb[:, :, 1] = euv_94_norm
    rgb[:, :, 2] = euv_211_norm
    
    # Display RGB image
    im4, = [ax4.imshow(rgb)]
    ax4.set_title('RGB Composite (R: Magnetogram, G: 94Å, B: 211Å)')
    ax4.axis('off')
    
    # Metrics and predictions
    ax5 = fig.add_subplot(gs[2, 0:3])
    ax5.axis('off')
    
    # Create text for predictions
    prediction_text = f"True Peak Flux: {peak_flux_true:.6f}\nPredicted Peak Flux: {peak_flux_pred:.6f}\nError: {abs(peak_flux_true - peak_flux_pred):.6f}"
    
    # Add classification results if available
    if 'c_vs_0' in predictions:
        c_vs_0_pred = predictions['c_vs_0'][sample_idx].item()
        c_vs_0_true = targets['is_c_flare'][sample_idx].item()
        prediction_text += f"\n\nC-class or Above:\n  True: {c_vs_0_true:.1f}\n  Predicted: {c_vs_0_pred:.4f}"
    
    if 'm_vs_0' in predictions:
        m_vs_0_pred = predictions['m_vs_0'][sample_idx].item()
        m_vs_0_true = targets['is_m_flare'][sample_idx].item()
        prediction_text += f"\n\nM-class or Above:\n  True: {m_vs_0_true:.1f}\n  Predicted: {m_vs_0_pred:.4f}"
    
    # Add prediction text
    ax5.text(0.5, 0.5, prediction_text, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    # Add title with timestamp
    time_labels = ['12h before', '5h before', '1.5h before', '10min before']
    fig.suptitle(f"Solar Flare Prediction Visualization - {time_labels[timestep_idx]}", fontsize=16)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def create_temporal_evolution_video(magnetogram, euv, predictions, targets, save_path, fps=1):
    """
    Create video showing temporal evolution of inputs and predictions
    
    Args:
        magnetogram: Magnetogram input tensor (B, T, C, H, W)
        euv: EUV input tensor (B, T, C, H, W)
        predictions: Model predictions dictionary
        targets: Target values dictionary
        save_path: Path to save video
        fps: Frames per second
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get timesteps
    time_steps = magnetogram.shape[1]
    time_labels = ['12h before', '5h before', '1.5h before', '10min before']
    
    # Create temporary directory for frames
    temp_dir = os.path.join(os.path.dirname(save_path), 'temp_frames')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Create frames for each timestep
    for t in range(time_steps):
        frame_path = os.path.join(temp_dir, f'frame_{t:03d}.png')
        visualize_multimodal_comparison(
            magnetogram, euv, predictions, targets, t, frame_path
        )
    
    # Create video using OpenCV
    frame_files = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
    if frame_files:
        # Read first frame to get dimensions
        first_frame = cv2.imread(os.path.join(temp_dir, frame_files[0]))
        height, width, layers = first_frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # Add frames to video
        for frame_file in frame_files:
            video.write(cv2.imread(os.path.join(temp_dir, frame_file)))
        
        # Release video writer
        video.release()
        
        print(f"Video saved to {save_path}")
    else:
        print("No frames found for video creation")
    
    # Clean up temporary frames
    for frame_file in frame_files:
        os.remove(os.path.join(temp_dir, frame_file))
    os.rmdir(temp_dir)


def plot_uncertainty_estimates(predictions, targets, save_path):
    """
    Plot uncertainty estimates
    
    Args:
        predictions: Model predictions dictionary
        targets: Target values dictionary
        save_path: Path to save visualization
    """
    # Check if uncertainty estimates are available
    if 'peak_flux_mean' not in predictions or 'peak_flux_logvar' not in predictions:
        print("Uncertainty estimates not available")
        return
    
    # Get predictions
    mean = predictions['peak_flux_mean'].cpu().numpy().flatten()
    logvar = predictions['peak_flux_logvar'].cpu().numpy().flatten()
    std = np.exp(logvar / 2)
    
    # Get targets
    true_values = targets['peak_flux'].cpu().numpy().flatten()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Sort points by true value for better visualization
    sort_idx = np.argsort(true_values)
    true_values = true_values[sort_idx]
    mean = mean[sort_idx]
    std = std[sort_idx]
    
    # Plot with error bars
    plt.errorbar(np.arange(len(true_values)), mean, yerr=2*std, fmt='o', capsize=5, label='Prediction with 95% CI')
    plt.plot(np.arange(len(true_values)), true_values, 'rx', label='True Value')
    
    # Add labels and legend
    plt.xlabel('Sample Index (sorted by true value)')
    plt.ylabel('Peak Flux')
    plt.title('Prediction with Uncertainty Estimates')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # Create calibration plot: fraction of true values within prediction interval vs. confidence level
    confidence_levels = np.linspace(0.01, 0.99, 50)
    within_interval = []
    
    for conf in confidence_levels:
        # Calculate interval half-width for this confidence level
        z = np.sqrt(2) * np.erfinv(conf)
        interval = z * std
        
        # Check how many true values are within interval
        within = np.abs(true_values - mean) <= interval
        fraction_within = np.mean(within)
        within_interval.append(fraction_within)
    
    # Create calibration plot
    plt.figure(figsize=(8, 8))
    plt.plot(confidence_levels, within_interval, 'b-', label='Model Calibration')
    plt.plot([0, 1], [0, 1], 'r--', label='Perfect Calibration')
    plt.xlabel('Confidence Level')
    plt.ylabel('Fraction of True Values within Interval')
    plt.title('Uncertainty Calibration Plot')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save calibration plot
    calibration_path = os.path.join(os.path.dirname(save_path), 'uncertainty_calibration.png')
    plt.tight_layout()
    plt.savefig(calibration_path)
    plt.close()


def create_gradient_shap_interpretation(model, magnetogram, euv, save_dir, device='cuda'):
    """
    Create GradientSHAP interpretations for model
    
    Args:
        model: Model to interpret
        magnetogram: Magnetogram input tensor (B, T, C, H, W)
        euv: EUV input tensor (B, T, C, H, W)
        save_dir: Directory to save interpretations
        device: Device to run interpretation on
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Clone inputs to detach from computation graph
    magnetogram = magnetogram.clone().detach().to(device)
    euv = euv.clone().detach().to(device)
    
    # Create baseline inputs (zeros)
    mag_baseline = torch.zeros_like(magnetogram).to(device)
    euv_baseline = torch.zeros_like(euv).to(device)
    
    # Set requires_grad
    magnetogram.requires_grad = True
    euv.requires_grad = True
    
    # Initialize GradientShap
    gradient_shap = GradientShap(model)
    
    # Compute attributions for each timestep
    for t in range(magnetogram.shape[1]):
        # Generate attributions for magnetogram
        mag_attr = gradient_shap.attribute(
            (magnetogram, euv),
            baselines=(mag_baseline, euv_baseline),
            target=0,  # Assuming 0 is the peak flux output
            n_samples=10
        )
        
        # Visualize attributions
        attributions = {
            'magnetogram': mag_attr[0],
            'euv': mag_attr[1]
        }
        
        # Save visualizations
        visualize_attributions(
            magnetogram, 
            euv, 
            attributions, 
            timestep_idx=t, 
            save_dir=os.path.join(save_dir, f'timestep_{t}')
        )
    
    print(f"GradientSHAP interpretations saved to {save_dir}")


def create_integrated_gradients_interpretation(model, magnetogram, euv, save_dir, device='cuda'):
    """
    Create Integrated Gradients interpretations for model
    
    Args:
        model: Model to interpret
        magnetogram: Magnetogram input tensor (B, T, C, H, W)
        euv: EUV input tensor (B, T, C, H, W)
        save_dir: Directory to save interpretations
        device: Device to run interpretation on
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Clone inputs to detach from computation graph
    magnetogram = magnetogram.clone().detach().to(device)
    euv = euv.clone().detach().to(device)
    
    # Create baseline inputs (zeros)
    mag_baseline = torch.zeros_like(magnetogram).to(device)
    euv_baseline = torch.zeros_like(euv).to(device)
    
    # Set requires_grad
    magnetogram.requires_grad = True
    euv.requires_grad = True
    
    # Initialize Integrated Gradients
    integrated_gradients = IntegratedGradients(model)
    
    # Compute attributions for each timestep
    for t in range(magnetogram.shape[1]):
        # Generate attributions for magnetogram
        mag_attr, euv_attr = integrated_gradients.attribute(
            (magnetogram, euv),
            baselines=(mag_baseline, euv_baseline),
            target=0,  # Assuming 0 is the peak flux output
            n_steps=50
        )
        
        # Visualize attributions
        attributions = {
            'magnetogram': mag_attr,
            'euv': euv_attr
        }
        
        # Save visualizations
        visualize_attributions(
            magnetogram, 
            euv, 
            attributions, 
            timestep_idx=t, 
            save_dir=os.path.join(save_dir, f'timestep_{t}')
        )
    
    print(f"Integrated Gradients interpretations saved to {save_dir}") 