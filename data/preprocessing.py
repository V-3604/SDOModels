import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import random
from sklearn.model_selection import train_test_split


class SDODataAugmentation:
    """
    Data augmentation for SDO magnetogram and EUV images
    """
    def __init__(self, p=0.5, rotation_degrees=15, scale_range=(0.9, 1.1),
                 brightness_range=(0.8, 1.2), contrast_range=(0.8, 1.2),
                 noise_level=0.05, flip_p=0.5):
        """
        Initialize the data augmentation with various transformations.
        
        Args:
            p: Probability of applying an augmentation
            rotation_degrees: Maximum rotation in degrees
            scale_range: Range for random scaling
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            noise_level: Standard deviation for Gaussian noise
            flip_p: Probability of horizontal flip
        """
        self.p = p
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_level = noise_level
        self.flip_p = flip_p
    
    def __call__(self, x):
        """
        Apply augmentation to input tensor
        
        Args:
            x: Input tensor of shape [T, C, H, W]
            
        Returns:
            Augmented tensor
        """
        # Don't augment with probability 1-p
        if random.random() > self.p:
            return x
        
        # Get tensor shape
        t, c, h, w = x.shape
        device = x.device
        
        # Convert to numpy for OpenCV operations
        x_np = x.cpu().numpy()
        
        # Generate random augmentation parameters (consistent across timesteps)
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        scale = random.uniform(*self.scale_range)
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        do_flip = random.random() < self.flip_p
        
        # Apply transformations to each timestep
        for time_idx in range(t):
            for channel_idx in range(c):
                img = x_np[time_idx, channel_idx]
                
                # Skip empty channels (all zeros)
                if np.max(img) == 0:
                    continue
                
                # Handle NaN and Inf values
                img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
                
                # Convert to uint8 for OpenCV
                img_uint8 = (img * 255).astype(np.uint8)
                
                # Horizontal flip
                if do_flip:
                    img_uint8 = cv2.flip(img_uint8, 1)
                
                # Rotation and scaling
                center = (w//2, h//2)
                M = cv2.getRotationMatrix2D(center, angle, scale)
                img_uint8 = cv2.warpAffine(img_uint8, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                
                # Brightness and contrast adjustment
                img_uint8 = cv2.convertScaleAbs(img_uint8, alpha=contrast, beta=(brightness-1)*255)
                
                # Convert back to float
                img = img_uint8.astype(np.float32) / 255.0
                
                # Add Gaussian noise
                if self.noise_level > 0:
                    noise = np.random.normal(0, self.noise_level, img.shape)
                    img = np.clip(img + noise, 0, 1)
                
                # Update tensor
                x_np[time_idx, channel_idx] = img
        
        # Convert back to tensor
        x_aug = torch.from_numpy(x_np).to(device)
        
        return x_aug


class SDOBenchmarkDataset(Dataset):
    def __init__(self, root_dir, metadata_df, split='train', img_size=128, transform=None):
        """
        Initialize the SDO Benchmark dataset.

        Args:
            root_dir (str): Root directory of the dataset (e.g., 'data/SDOBenchmark-data-full').
            metadata_df (pd.DataFrame): DataFrame containing metadata (e.g., sample_id, peak_flux).
            split (str): Dataset split ('train', 'val', 'test').
            img_size (int): Size to resize images to (e.g., 128x128).
            transform (callable, optional): Optional transform to apply to images.
        """
        self.root_dir = root_dir
        self.metadata_df = metadata_df
        self.split = split
        self.img_size = img_size
        self.transform = transform

        # Determine base directory based on split
        self.data_dir = os.path.join(root_dir, 'training' if split in ['train', 'val'] else 'test')
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")

        # Filter metadata for this split
        if split in ['train', 'val']:
            all_samples = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
            train_samples, val_samples = train_test_split(
                all_samples, test_size=0.2, random_state=42
            )
            self.sample_ids = train_samples if split == 'train' else val_samples
        else:  # test
            self.sample_ids = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

        self.sample_folders = [os.path.join(self.data_dir, sid) for sid in self.sample_ids]
        print(f"Initialized {split} dataset with {len(self.sample_folders)} samples")

    def __len__(self):
        return len(self.sample_folders)

    def __getitem__(self, idx):
        sample_folder = self.sample_folders[idx]
        sample_id = self.sample_ids[idx]

        # Get metadata for this sample
        meta_row = self.metadata_df[self.metadata_df['sample_id'] == sample_id].iloc[0] if sample_id in self.metadata_df['sample_id'].values else None

        # Load timestamp subfolders (assuming 4 timesteps)
        timestamp_subfolders = [os.path.join(sample_folder, d) for d in os.listdir(sample_folder) if os.path.isdir(os.path.join(sample_folder, d))]
        timestamp_subfolders.sort()

        magnetogram = torch.zeros(4, 1, self.img_size, self.img_size)
        euv = torch.zeros(4, 8, self.img_size, self.img_size)

        for i, ts_folder in enumerate(timestamp_subfolders[:4]):
            files = os.listdir(ts_folder)
            # Load magnetogram
            mag_files = [f for f in files if 'magnetogram' in f.lower() and f.endswith(('.jpg', '.jpeg', '.png'))]
            if mag_files:
                img_path = os.path.join(ts_folder, mag_files[0])
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    # Normalize magnetogram to [-1, 1] range instead of [0, 1]
                    magnetogram[i, 0] = torch.from_numpy(img).float() / 255.0 * 2.0 - 1.0

            # Load EUV wavelengths
            wavelengths = ['94', '131', '171', '193', '211', '304', '335', '1700']  # Restored '1700' for 8 channels
            for wl_idx, wl in enumerate(wavelengths):
                euv_files = [f for f in files if f.endswith(f'_{wl}.jpg') or f.endswith(f'_{wl}.jpeg') or f.endswith(f'_{wl}.png')]
                if euv_files:
                    img_path = os.path.join(ts_folder, euv_files[0])
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        # Apply log scaling to EUV data to enhance contrast of solar features
                        img_array = np.array(img).astype(np.float32) / 255.0
                        # Avoid log(0)
                        img_array = np.clip(img_array, 0.001, 1.0)
                        img_log = np.log1p(img_array * 10) / np.log(11)
                        euv[i, wl_idx] = torch.from_numpy(img_log).float()

        # Handle NaN and Inf values
        magnetogram = torch.nan_to_num(magnetogram, nan=0.0, posinf=1.0, neginf=-1.0)
        euv = torch.nan_to_num(euv, nan=0.0, posinf=1.0, neginf=0.0)

        sample = {
            'magnetogram': magnetogram,
            'euv': euv,
            'sample_id': sample_id
        }

        # Assign labels from metadata or defaults
        if meta_row is not None:
            peak_flux = meta_row.get('peak_flux', 1e-6)
            # Clip and normalize peak flux to handle outliers
            peak_flux = np.clip(peak_flux, 1e-8, 1e-3)
            # Log transform for better numerical stability
            log_peak_flux = np.log10(peak_flux) + 8  # Shift for positive values
            sample['peak_flux'] = torch.tensor([log_peak_flux], dtype=torch.float32)
            
            flare_class = meta_row.get('flare_class', 'N')
            sample['is_c_flare'] = torch.tensor([1 if flare_class in ['C', 'M', 'X'] else 0], dtype=torch.float32)
            sample['is_m_flare'] = torch.tensor([1 if flare_class in ['M', 'X'] else 0], dtype=torch.float32)
            sample['flare_class'] = flare_class
        else:
            # Default transformed value for 1e-6
            sample['peak_flux'] = torch.tensor([np.log10(1e-6) + 8], dtype=torch.float32)
            sample['is_c_flare'] = torch.tensor([0], dtype=torch.float32)
            sample['is_m_flare'] = torch.tensor([0], dtype=torch.float32)
            sample['flare_class'] = 'N'

        if self.transform:
            sample['magnetogram'] = self.transform(sample['magnetogram'])
            sample['euv'] = self.transform(sample['euv'])

        return sample


class SDODataset(SDOBenchmarkDataset):
    """Alias for SDOBenchmarkDataset for backward compatibility"""
    pass


class SDODataPreprocessor:
    """
    Preprocessor for SDO data, handling normalization and standardization
    """
    def __init__(self):
        # Statistics pre-computed on the training set
        self.magnetogram_mean = 0.0
        self.magnetogram_std = 1.0
        self.euv_means = [0.2, 0.15, 0.3, 0.25, 0.2, 0.35, 0.15, 0.4]  # Restored value for 1700
        self.euv_stds = [0.2, 0.15, 0.2, 0.15, 0.15, 0.25, 0.1, 0.3]   # Restored value for 1700
    
    def __call__(self, sample):
        """
        Normalize and standardize the data
        
        Args:
            sample: Sample dictionary with 'magnetogram' and 'euv' keys
            
        Returns:
            Preprocessed sample
        """
        # Standardize magnetogram
        mag = sample['magnetogram']
        mag = (mag - self.magnetogram_mean) / self.magnetogram_std
        sample['magnetogram'] = mag
        
        # Standardize EUV channels
        euv = sample['euv']
        for i in range(min(euv.shape[1], len(self.euv_means))):
            euv[:, i] = (euv[:, i] - self.euv_means[i]) / self.euv_stds[i]
        sample['euv'] = euv
        
        return sample


def get_data_loaders(data_path, metadata_path=None, batch_size=8, img_size=128, num_workers=2, sample_type='all'):
    """
    Create data loaders for train, val, and test splits.

    Args:
        data_path (str): Root directory of the dataset.
        metadata_path (str): Path to metadata CSV file.
        batch_size (int): Batch size for data loaders.
        img_size (int): Image size for resizing.
        num_workers (int): Number of workers for data loading.
        sample_type (str): Sampling strategy for class imbalance.
    """
    # If metadata_path is not specified, look for metadata.csv in the data directory
    if metadata_path is None:
        metadata_path = os.path.join(data_path, 'metadata.csv')
        if not os.path.exists(metadata_path):
            # Create a dummy metadata DataFrame if no file exists
            print(f"Warning: No metadata file found at {metadata_path}. Using dummy metadata.")
            metadata_df = pd.DataFrame(columns=['sample_id', 'peak_flux', 'flare_class'])
        else:
            metadata_df = pd.read_csv(metadata_path)
    else:
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        metadata_df = pd.read_csv(metadata_path)

    # Create transforms
    train_transform = SDODataAugmentation(p=0.7)
    preprocessor = SDODataPreprocessor()

    # Create datasets
    train_dataset = SDOBenchmarkDataset(data_path, metadata_df, split='train', img_size=img_size, transform=train_transform)
    val_dataset = SDOBenchmarkDataset(data_path, metadata_df, split='val', img_size=img_size, transform=None)
    test_dataset = SDOBenchmarkDataset(data_path, metadata_df, split='test', img_size=img_size, transform=None)

    # Handle class imbalance
    if sample_type != 'all':
        # Identify flare samples (C and M class)
        c_flare_indices = [i for i, sample in enumerate(train_dataset) if sample['is_c_flare'].item() == 1]
        m_flare_indices = [i for i, sample in enumerate(train_dataset) if sample['is_m_flare'].item() == 1]
        non_flare_indices = [i for i, sample in enumerate(train_dataset) 
                             if sample['is_c_flare'].item() == 0 and sample['is_m_flare'].item() == 0]
        
        if sample_type == 'balanced':
            # Undersample non-flare class
            n_flare = len(c_flare_indices) + len(m_flare_indices)
            sampled_non_flare = random.sample(non_flare_indices, min(n_flare, len(non_flare_indices)))
            balanced_indices = c_flare_indices + m_flare_indices + sampled_non_flare
            
            # Create a subset with balanced samples
            from torch.utils.data import Subset
            train_dataset = Subset(train_dataset, balanced_indices)
            
        elif sample_type == 'oversampled':
            # Create a sampler that oversamples the minority classes
            weights = torch.ones(len(train_dataset))
            
            # Set higher weights for flare samples
            for idx in c_flare_indices:
                weights[idx] = 5.0  # Oversample C-class flares
            for idx in m_flare_indices:
                weights[idx] = 10.0  # Oversample M-class flares even more
                
            # Create a weighted random sampler
            from torch.utils.data import WeightedRandomSampler
            sampler = WeightedRandomSampler(weights, num_samples=len(train_dataset), replacement=True)
            
            # Use the sampler in the DataLoader
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                sampler=sampler,
                num_workers=num_workers,
                drop_last=True
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            
            return {'train': train_loader, 'val': val_loader, 'test': test_loader}

    # Default DataLoaders if no specialized sampling is used
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}