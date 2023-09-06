from tensorflow.python.compiler.tensorrt import trt_convert as trt

from options import load_trt_convert_options

def convert_model(input_name,output_name):
    # Conversion Parameters
    conversion_params = trt.TrtConversionParams(
        precision_mode=trt.TrtPrecisionMode.FP32)
    
    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=input_name,
        conversion_params=conversion_params)
    
    # Converter method used to partition and optimize TensorRT compatible segments
    converter.convert()
    
    # Save the model to the disk
    converter.save(output_name)


if __name__ == "__main__":
    parser = load_trt_convert_options()
    args = parser.parse_args()

    input_name = args.input_name
    output_name = args.output_name
    
    if input_name and output_name:
        print("Converting Model to TensorRT...")
        convert_model(input_name, output_name)
        print("Success!")