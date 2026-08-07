from generate import (generate_dsp_constants, generate_feature_config,
                      generate_inference)

if __name__ == "__main__":
    print("\n")
    generate_dsp_constants()
    print("\n")
    generate_feature_config()
    print("\n")
    generate_inference()
    print("\n")
    print("Firmware generation complete. All necessary C++/H files have been created in the firmware directory.")