"""
Configuration constants for QModel Trainer

Centralizes all magic numbers and configuration values used across the plugin.
Makes it easy to adjust behavior without hunting through code.
"""

# =============================================================================
# QUALITATIVE EXAMPLES CONFIGURATION
# =============================================================================

# IoU thresholds for example categorization
QUALITATIVE_HIGH_IOU_THRESHOLD = 0.90  # Examples with IoU >= 0.90
QUALITATIVE_LOW_IOU_THRESHOLD = 0.60   # Examples with IoU <= 0.60
QUALITATIVE_MEDIUM_IOU_MARGIN = 0.10   # Margin around median IoU (±0.10)

# Number of examples to generate per category
QUALITATIVE_NUM_EXAMPLES_PER_CATEGORY = 4

# Maximum validation images to process for qualitative examples
QUALITATIVE_MAX_IMAGES_FOR_INFERENCE = 30

# Display size for example images (resized from original)
QUALITATIVE_EXAMPLE_TILE_SIZE = 256  # pixels


# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

# Default hyperparameters
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 4
DEFAULT_IMAGE_SIZE = 512
DEFAULT_VAL_SPLIT = 0.2
DEFAULT_EPOCHS = 50
DEFAULT_EARLY_STOPPING_PATIENCE = 10

# Optimizer defaults
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_MOMENTUM = 0.9  # For SGD

# Scheduler defaults (ReduceLROnPlateau)
DEFAULT_SCHEDULER_PATIENCE = 5
DEFAULT_SCHEDULER_FACTOR = 0.5


# =============================================================================
# LR FINDER CONFIGURATION
# =============================================================================

# Learning rate range to test
LR_FINDER_START_LR = 1e-7
LR_FINDER_END_LR = 1e-1

# Number of steps for LR range test
LR_FINDER_NUM_STEPS = 100

# Exponential moving average smoothing factor (0-1, lower = more smooth)
LR_FINDER_SMOOTH_FACTOR = 0.05

# Safety factor: suggested_lr = gradient_max_lr / SAFETY_FACTOR
LR_FINDER_SAFETY_FACTOR = 10


# =============================================================================
# REPORT GENERATION
# =============================================================================

# Plot DPI settings
PLOT_DPI_STANDARD = 150
PLOT_DPI_HIGH = 180  # For confusion matrix

# Confusion matrix settings
CONFUSION_MATRIX_MIN_CLASSES = 2  # Skip if fewer classes present

# Report quality assessment thresholds
REPORT_QUALITY_EXCELLENT_THRESHOLD = 0.85
REPORT_QUALITY_GOOD_THRESHOLD = 0.70
REPORT_QUALITY_MODERATE_THRESHOLD = 0.50

# Class imbalance detection thresholds
CLASS_IMBALANCE_SEVERE_RATIO = 50  # max_pct / min_pct > 50
CLASS_IMBALANCE_MODERATE_RATIO = 10  # max_pct / min_pct > 10
CLASS_IMBALANCE_MIN_PERCENTAGE = 5.0  # Warn if class < 5%


# =============================================================================
# MODEL EXPORT (QMTP)
# =============================================================================

# QMTP format version
QMTP_FORMAT_VERSION = "1.0"

# Default tile stride (used if not specified in metadata)
DEFAULT_TILE_STRIDE_FACTOR = 0.5  # stride = tile_size * 0.5

# Default padding for tile inference
DEFAULT_TILE_PADDING = 64  # pixels


# =============================================================================
# DATASET VALIDATION
# =============================================================================

# Minimum number of images per split
MIN_TRAIN_IMAGES = 1
MIN_VAL_IMAGES = 1

# Recommended minimum for production
RECOMMENDED_MIN_IMAGES_PER_CLASS = 100

# File extensions
SUPPORTED_IMAGE_EXTENSIONS = ['.tif', '.tiff', '.png', '.jpg', '.jpeg']
SUPPORTED_MASK_EXTENSIONS = ['.tif', '.tiff', '.png']


# =============================================================================
# PLUGIN METADATA INUTILE POUR L'INSTANT SI PAS UTILISE A DEGAGER
# =============================================================================

PLUGIN_NAME = "QModel Trainer"
PLUGIN_VERSION = "1.0.0"
PLUGIN_AUTHOR = "Nextelia"
PLUGIN_WEBSITE = "https://qgeoai.nextelia.fr/"
PLUGIN_REPO = "https://github.com/nextelia/qmodel-trainer"