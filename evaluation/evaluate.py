import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, 
    roc_curve, auc, precision_recall_curve, 
    confusion_matrix, classification_report
)
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import get_data_loaders
from models.model import SolarFlareModel
from training.train import SolarFlareModule
from captum.attr import GradientShap, IntegratedGradients, Occlusion
import shap


def calculate_tss(y_true, y_pred):
    """
    Calculate True Skill Statistic (TSS)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        TSS value
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return sensitivity + specificity - 1


def calculate_hss(y_true, y_pred):
    """
    Calculate Heidke Skill Score (HSS)
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        HSS value
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    a = tp + tn  # Correct predictions
    b = ((tp + fp) * (tp + fn) + (tn + fp) * (tn + fn)) / (tp + tn + fp + fn)  # Expected correct predictions by chance
    
    return (a - b) / (tp + tn + fp + fn - b)


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """
    Plot confusion matrix
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_score, title, save_path):
    """
    Plot ROC curve
    
    Args:
        y_true: True labels
        y_score: Predicted scores
        title: Plot title
        save_path: Path to save the plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(model, data_loader, output_dir, device):
    """
    Evaluate model on a dataset
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        output_dir: Output directory for plots and results
        device: Device to run evaluation on
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize lists to store predictions and targets
    peak_flux_pred = []
    peak_flux_true = []
    c_vs_0_pred = []
    c_vs_0_true = []
    m_vs_0_pred = []
    m_vs_0_true = []
    
    # Disable gradient calculation
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            magnetogram = batch['magnetogram'].to(device)
            euv = batch['euv'].to(device)
            targets = batch['target']
            
            # Forward pass
            predictions = model(magnetogram, euv)
            
            # Get predictions
            peak_flux = predictions['peak_flux_mean'] if 'peak_flux_mean' in predictions else predictions['peak_flux']
            peak_flux = peak_flux.cpu().numpy()
            
            # Append predictions and targets
            peak_flux_pred.append(peak_flux)
            peak_flux_true.append(targets['peak_flux'].numpy())
            
            # Classification predictions
            if 'c_vs_0' in predictions:
                c_vs_0_pred.append(predictions['c_vs_0'].cpu().numpy())
                c_vs_0_true.append(targets['is_c_flare'].numpy())
            
            if 'm_vs_0' in predictions:
                m_vs_0_pred.append(predictions['m_vs_0'].cpu().numpy())
                m_vs_0_true.append(targets['is_m_flare'].numpy())
    
    # Concatenate predictions and targets
    peak_flux_pred = np.concatenate(peak_flux_pred).flatten()
    peak_flux_true = np.concatenate(peak_flux_true).flatten()
    
    # Calculate regression metrics
    mae = mean_absolute_error(peak_flux_true, peak_flux_pred)
    mse = mean_squared_error(peak_flux_true, peak_flux_pred)
    rmse = np.sqrt(mse)
    
    # Print regression metrics
    print(f"Regression Metrics:")
    print(f"  MAE: {mae:.6f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    
    # Plot regression results
    plt.figure(figsize=(8, 6))
    plt.scatter(peak_flux_true, peak_flux_pred, alpha=0.5)
    plt.plot([min(peak_flux_true), max(peak_flux_true)], [min(peak_flux_true), max(peak_flux_true)], 'k--')
    plt.xlabel('True Peak Flux')
    plt.ylabel('Predicted Peak Flux')
    plt.title('Regression Results')
    plt.savefig(os.path.join(output_dir, 'regression_results.png'))
    plt.close()
    
    # Calculate classification metrics if available
    metrics = {'mae': mae, 'mse': mse, 'rmse': rmse}
    
    if c_vs_0_pred:
        # Concatenate predictions and targets
        c_vs_0_pred = np.concatenate(c_vs_0_pred).flatten()
        c_vs_0_true = np.concatenate(c_vs_0_true).flatten()
        
        # Get binary predictions
        c_vs_0_pred_binary = (c_vs_0_pred > 0.5).astype(int)
        
        # Calculate classification metrics
        tss_c_vs_0 = calculate_tss(c_vs_0_true, c_vs_0_pred_binary)
        hss_c_vs_0 = calculate_hss(c_vs_0_true, c_vs_0_pred_binary)
        
        # Print classification metrics
        print(f"\nC vs 0 Classification Metrics:")
        print(f"  TSS: {tss_c_vs_0:.4f}")
        print(f"  HSS: {hss_c_vs_0:.4f}")
        print("\nClassification Report:")
        print(classification_report(c_vs_0_true, c_vs_0_pred_binary, target_names=['Quiet', 'C-class or above']))
        
        # Plot confusion matrix
        plot_confusion_matrix(
            c_vs_0_true, 
            c_vs_0_pred_binary, 
            ['Quiet', 'C-class or above'], 
            'C vs 0 Confusion Matrix',
            os.path.join(output_dir, 'c_vs_0_confusion_matrix.png')
        )
        
        # Plot ROC curve
        plot_roc_curve(
            c_vs_0_true,
            c_vs_0_pred,
            'C vs 0 ROC Curve',
            os.path.join(output_dir, 'c_vs_0_roc_curve.png')
        )
        
        # Add metrics to dictionary
        metrics['tss_c_vs_0'] = tss_c_vs_0
        metrics['hss_c_vs_0'] = hss_c_vs_0
    
    if m_vs_0_pred:
        # Concatenate predictions and targets
        m_vs_0_pred = np.concatenate(m_vs_0_pred).flatten()
        m_vs_0_true = np.concatenate(m_vs_0_true).flatten()
        
        # Get binary predictions
        m_vs_0_pred_binary = (m_vs_0_pred > 0.5).astype(int)
        
        # Calculate classification metrics
        tss_m_vs_0 = calculate_tss(m_vs_0_true, m_vs_0_pred_binary)
        hss_m_vs_0 = calculate_hss(m_vs_0_true, m_vs_0_pred_binary)
        
        # Print classification metrics
        print(f"\nM vs 0 Classification Metrics:")
        print(f"  TSS: {tss_m_vs_0:.4f}")
        print(f"  HSS: {hss_m_vs_0:.4f}")
        print("\nClassification Report:")
        print(classification_report(m_vs_0_true, m_vs_0_pred_binary, target_names=['Non-M', 'M-class or above']))
        
        # Plot confusion matrix
        plot_confusion_matrix(
            m_vs_0_true, 
            m_vs_0_pred_binary, 
            ['Non-M', 'M-class or above'], 
            'M vs 0 Confusion Matrix',
            os.path.join(output_dir, 'm_vs_0_confusion_matrix.png')
        )
        
        # Plot ROC curve
        plot_roc_curve(
            m_vs_0_true,
            m_vs_0_pred,
            'M vs 0 ROC Curve',
            os.path.join(output_dir, 'm_vs_0_roc_curve.png')
        )
        
        # Add metrics to dictionary
        metrics['tss_m_vs_0'] = tss_m_vs_0
        metrics['hss_m_vs_0'] = hss_m_vs_0
    
    return metrics


def interpret_model(model, data_loader, output_dir, device, num_samples=10):
    """
    Generate model interpretations
    
    Args:
        model: Model to interpret
        data_loader: Data loader
        output_dir: Output directory for plots and results
        device: Device to run interpretation on
        num_samples: Number of samples to interpret
    """
    # Create output directory if it doesn't exist
    interpretation_dir = os.path.join(output_dir, 'interpretations')
    os.makedirs(interpretation_dir, exist_ok=True)
    
    # Set model to evaluation mode
    model.eval()
    
    # Select a few samples for interpretation
    samples = []
    for batch in data_loader:
        # Add samples from batch
        for i in range(min(batch['magnetogram'].size(0), num_samples - len(samples))):
            samples.append({
                'magnetogram': batch['magnetogram'][i:i+1].to(device),
                'euv': batch['euv'][i:i+1].to(device),
                'target': {k: v[i:i+1] for k, v in batch['target'].items()}
            })
        
        # Break if we have enough samples
        if len(samples) >= num_samples:
            break
    
    # Initialize interpretation methods
    gradient_shap = GradientShap(model)
    integrated_gradients = IntegratedGradients(model)
    
    # Interpret each sample
    for i, sample in enumerate(samples):
        # Create sample-specific directory
        sample_dir = os.path.join(interpretation_dir, f'sample_{i}')
        os.makedirs(sample_dir, exist_ok=True)
        
        # Get inputs
        magnetogram = sample['magnetogram']
        euv = sample['euv']
        
        # Forward pass
        predictions = model(magnetogram, euv)
        
        # Get prediction
        peak_flux = predictions['peak_flux_mean'] if 'peak_flux_mean' in predictions else predictions['peak_flux']
        peak_flux = peak_flux.item()
        
        # Save true and predicted values
        true_peak_flux = sample['target']['peak_flux'].item()
        with open(os.path.join(sample_dir, 'predictions.txt'), 'w') as f:
            f.write(f"True Peak Flux: {true_peak_flux:.6f}\n")
            f.write(f"Predicted Peak Flux: {peak_flux:.6f}\n")
            
            if 'c_vs_0' in predictions:
                c_vs_0 = predictions['c_vs_0'].item()
                true_c_vs_0 = sample['target']['is_c_flare'].item()
                f.write(f"True C vs 0: {true_c_vs_0:.1f}\n")
                f.write(f"Predicted C vs 0: {c_vs_0:.4f}\n")
            
            if 'm_vs_0' in predictions:
                m_vs_0 = predictions['m_vs_0'].item()
                true_m_vs_0 = sample['target']['is_m_flare'].item()
                f.write(f"True M vs 0: {true_m_vs_0:.1f}\n")
                f.write(f"Predicted M vs 0: {m_vs_0:.4f}\n")
        
        # Visualization and interpretation code would go here
        # This would use captum and SHAP libraries to visualize important features
        # Example: Gradient SHAP for magnetogram
        # attributions = gradient_shap.attribute(magnetogram, n_samples=20, target=0)
        # Plot attributions
        
        # Save sample images (simplified visualization)
        # Magnetogram (last timestep, single channel)
        mag_img = magnetogram[0, -1, 0].cpu().numpy()
        plt.figure(figsize=(6, 6))
        plt.imshow(mag_img, cmap='gray')
        plt.colorbar()
        plt.title('Magnetogram (Last Timestep)')
        plt.savefig(os.path.join(sample_dir, 'magnetogram.png'))
        plt.close()
        
        # EUV images (selected channels)
        for ch_idx, wavelength in enumerate([94, 211]):
            euv_img = euv[0, -1, ch_idx].cpu().numpy()
            plt.figure(figsize=(6, 6))
            plt.imshow(euv_img, cmap='viridis')
            plt.colorbar()
            plt.title(f'EUV {wavelength}Å (Last Timestep)')
            plt.savefig(os.path.join(sample_dir, f'euv_{wavelength}.png'))
            plt.close()


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description='Evaluate solar flare prediction model')
    
    # Model configuration
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='data/sdo',
                        help='Path to dataset')
    parser.add_argument('--output_dir', type=str, default='evaluation/results',
                        help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--sample_type', type=str, default='all',
                        choices=['all', 'balanced', 'oversampled'],
                        help='Sampling strategy for test data')
    parser.add_argument('--do_interpretation', action='store_true',
                        help='Whether to perform model interpretation')
    parser.add_argument('--num_interpretation_samples', type=int, default=10,
                        help='Number of samples to use for interpretation')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SolarFlareModule.load_from_checkpoint(args.checkpoint_path)
    model.to(device)
    model.eval()
    
    # Get test data loader
    data_loaders = get_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_type=args.sample_type
    )
    test_loader = data_loaders['test']
    
    # Evaluate model
    metrics = evaluate_model(model.model, test_loader, args.output_dir, device)
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name}: {metric_value}\n")
    
    # Interpret model if requested
    if args.do_interpretation:
        interpret_model(
            model.model, 
            test_loader, 
            args.output_dir, 
            device, 
            num_samples=args.num_interpretation_samples
        )
    
    print(f"Evaluation complete. Results saved to {args.output_dir}")


if __name__ == '__main__':
    main() 