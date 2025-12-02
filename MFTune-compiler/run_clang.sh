#!/bin/bash

#declare -a methods=("ga" "smac" "bestconfig"  "flash" "ga" "hyperband" "hebo" "bohb" "dehb")
#declare -a services=("clang_gamf" "clang_smac" "clang_bestconfig"  "clang_flash"  "clang_gasf" "clang_hyperband" "clang_hebo" "clang_bohb" "clang_dehb")
#declare -a fidelity_types=("multi_fidelity" "single_fidelity" "single_fidelity" "single_fidelity"  "single_fidelity" "multi_fidelity" "single_fidelity" "multi_fidelity" "multi_fidelity")

declare -a methods=("ga")
declare -a services=("clang_gamf")
declare -a fidelity_types=("multi_fidelity")


# SYSTEM setup
SYSTEM="clang"

mkdir -p logs
chmod -R 777 logs

for i in "${!methods[@]}"; do
  method="${methods[$i]}"
  compiler_service="${services[$i]}"
  fidelity="${fidelity_types[$i]}"

  for run in {0..9}; do
    echo "========= RUN ${run} START | METHOD: ${method} ========="
    run_log="logs/run${run}_${method}_${compiler_service}_container.log"

    echo ">>> RUN $run | METHOD: $method | FIDELITY: $fidelity | DOCKER-SERVICE: $compiler_service" | tee -a "$run_log"

    # Start containers
    COMPILER_HOST=$compiler_service \
    TUNING_METHOD=$method \
    FIDELITY_TYPE=$fidelity \
    SYSTEM=$SYSTEM \
    RUN=$run \
    docker-compose up -d app_tuning $compiler_service >> "$run_log" 2>&1

    if [ $? -ne 0 ]; then
      echo "❌ Failed to start containers: app_tuning + $compiler_service" | tee -a "$run_log"
      continue
    fi

    # Wait for container to stabilize
    sleep 10

    # Check if container is running
    status=$(docker inspect -f '{{.State.Status}}' app_tuning_container 2>/dev/null)
    if [ "$status" != "running" ]; then
      echo "❌ app_tuning_container is not running. Status: $status" | tee -a "$run_log"
      docker logs app_tuning_container >> "$run_log" 2>&1
      docker-compose down -v >> "$run_log" 2>&1
      continue
    fi

    echo "✅ app_tuning_container is running. Waiting for completion..." | tee -a "$run_log"
    docker wait app_tuning_container >> "$run_log" 2>&1

    echo "🔍 app_tuning_container logs:" | tee -a "$run_log"
    docker logs app_tuning_container >> "$run_log" 2>&1

    echo "🧹 Cleaning up containers and volumes..." | tee -a "$run_log"
    docker-compose down -v >> "$run_log" 2>&1

    echo "🧼 Dropping caches..." | tee -a "$run_log"
    sudo sync
    sudo sysctl -w vm.drop_caches=3

    echo "😴 Cooling down for 30 minutes..." | tee -a "$run_log"
    sleep 1800

    echo "✅ Finished $method on $compiler_service for run $run" | tee -a "$run_log"
    echo "----------------------------------------"
  done

done
