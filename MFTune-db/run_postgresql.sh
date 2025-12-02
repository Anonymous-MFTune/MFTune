#!/bin/bash

declare -a methods=("ga" "ga" "flash" "smac" "bestconfig" "hebo" "promise" "hyperband" "priorband" "bohb" "dehb")
declare -a services=("postgres_ga_mf" "postgres_ga_sf" "postgres_flash" "postgres_smac" "postgres_bestconfig" "postgres_hebo" "postgres_promise" "postgres_hyperband" "postgres_priorband" "postgres_bohb" "postgres_dehb")
declare -a fidelity_types=("multi_fidelity" "single_fidelity" "single_fidelity" "single_fidelity" "single_fidelity" "single_fidelity" "single_fidelity" "multi_fidelity" "multi_fidelity" "multi_fidelity" "multi_fidelity")

#declare -a methods=("ga")
#declare -a services=("postgres_ga_mf")
#declare -a fidelity_types=("multi_fidelity")

SYSTEM="postgresql"

mkdir -p logs
chmod -R 777 logs

for i in "${!methods[@]}"; do
  method="${methods[$i]}"
  db_service="${services[$i]}"
  fidelity="${fidelity_types[$i]}"

  for run in {0..2}; do
    echo "========= RUN ${run} START | METHOD: ${method} ========="
    run_log="logs/run${run}_${method}_${db_service}_container.log"

    echo ">>> RUN $run | METHOD: $method | FIDELITY: $fidelity | DB: $db_service" | tee -a "$run_log"

    # Start containers
    DB_HOST=$db_service \
    TUNING_METHOD=$method \
    FIDELITY_TYPE=$fidelity \
    SYSTEM=$SYSTEM \
    RUN=$run \
    docker-compose up -d app_tuning $db_service >> "$run_log" 2>&1

    if [ $? -ne 0 ]; then
      echo "❌ Failed to start containers: app_tuning + $db_service" | tee -a "$run_log"
      continue
    fi

    # Wait a bit to let services stabilize
    sleep 10

    # Check if app_tuning container is running
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

    echo "✅ Finished $method on $db_service for run $run" | tee -a "$run_log"
    echo "----------------------------------------"
  done
done
