from generate import (generate_dsp_constants, generate_feature_config,
                      generate_inference)

if __name__ == "__main__":
    generate_dsp_constants()
    generate_feature_config()
    generate_inference()
    print("Firmware generation complete. All necessary C++/H files have been created in the firmware directory.")