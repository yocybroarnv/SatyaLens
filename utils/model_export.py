import os
import tensorflow as tf

def export_to_tflite(model_path="satyalens_v6_efficientnetb0.keras", export_path="satyalens_v6_efficientnetb0.tflite"):
    """
    Converts the Keras model to TFLite format with basic optimizations.
    Returns: (success_bool, message_str)
    """
    if not os.path.exists(model_path):
        return False, f"Source model weights {model_path} not found."
    
    try:
        # Load keras model
        model = tf.keras.models.load_model(model_path)
        
        # Convert model
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Write TFLite file
        with open(export_path, 'wb') as f:
            f.write(tflite_model)
            
        return True, f"Successfully converted and saved TFLite model to '{export_path}' (~7MB)."
    except Exception as e:
        return False, f"TFLite conversion failed: {str(e)}"

def export_to_onnx(model_path="satyalens_v6_efficientnetb0.keras", export_path="satyalens_v6_efficientnetb0.onnx"):
    """
    Attempts to convert the Keras model to ONNX format using tf2onnx.
    Returns: (success_bool, message_str)
    """
    if not os.path.exists(model_path):
        return False, f"Source model weights {model_path} not found."
        
    try:
        import tf2onnx
        import onnx
        
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Set input tensor signature (224x224 RGB image batches)
        spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
        
        # Convert
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
        
        # Save ONNX file
        with open(export_path, "wb") as f:
            f.write(model_proto.SerializeToString())
            
        return True, f"Successfully converted and saved ONNX model to '{export_path}'."
    except ImportError:
        return False, "ONNX conversion requires 'tf2onnx' and 'onnx' Python packages. Install them using: pip install tf2onnx onnx"
    except Exception as e:
        return False, f"ONNX conversion failed: {str(e)}"
