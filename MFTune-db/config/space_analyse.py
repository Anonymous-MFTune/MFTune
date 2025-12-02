import os
import json

def analyze_knobs(file_path):
    with open(file_path, 'r') as f:
        knobs = json.load(f)

    first_20_keys = list(knobs.keys())[:20]
    categorical_20 = 0
    numerical_20 = 0
    total_space = 1

    for i, (k, v) in enumerate(knobs.items()):
        knob_type = v.get("type", "").lower()
        if i < 20:
            if knob_type in ["enum", "categorical"]:
                categorical_20 += 1
            elif knob_type in ["integer", "float"]:
                numerical_20 += 1

        # Estimate search space
        if knob_type in ["enum", "categorical"]:
            values = v.get("enum_values") or v.get("choices") or []
            if values:
                total_space *= len(values)
        elif knob_type == "integer" and "min" in v and "max" in v:
            total_space *= (int(v["max"]) - int(v["min"]) + 1)
        elif knob_type == "float" and "min" in v and "max" in v:
            total_space *= 100  # assume 100 discretized float levels

    return categorical_20, numerical_20, total_space

def main():
    config_dir = os.path.dirname(os.path.abspath(__file__))
    files = {
        "MySQL": os.path.join(config_dir, "mysql_knobs.json"),
        "PostgreSQL": os.path.join(config_dir, "postgresql_knobs.json"),
        "Tomcat:": os.path.join(config_dir, "tomcat_knobs.json"),
        "Httpd": os.path.join(config_dir, "httpd_knobs.json"),
        "GCC": os.path.join(config_dir, "gcc_knobs.json"),
        "Clang": os.path.join(config_dir, "clang_knobs.json"),
        "tomcat": os.path.join(config_dir, "tomcat_knobs2.json")
    }

    for system, path in files.items():
        if not os.path.isfile(path):
            print(f"[Error] File not found: {path}")
            continue

        cat20, num20, space = analyze_knobs(path)
        print(f"--- {system} ---")
        print(f"Categorical (first 20): {cat20}")
        print(f"Numerical   (first 20): {num20}")
        from decimal import Decimal
        print(f"Total search space: {Decimal(space):.2e}")

        print()

if __name__ == "__main__":
    main()
