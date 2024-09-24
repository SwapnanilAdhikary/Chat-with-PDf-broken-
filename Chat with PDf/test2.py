import tensorflow as tf

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPUs available:")
    for gpu in gpus:
        print(f" - {gpu}")
else:
    print("No GPU available.")
