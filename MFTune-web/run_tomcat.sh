#!/bin/bash

#declare -a methods=("bestconfig" "smac" "ga" "flash" "hyperband")
#declare -a services=("tomcatbestconfig" "tomcatsmac" "tomcatgasf" "tomcatflash" "tomcathyperband")
#declare -a ports=("8081" "8082" "8084" "8085" "8086")
#declare -a fidelity_types=("single_fidelity" "single_fidelity" "single_fidelity" "single_fidelity" "multi_fidelity")

#declare -a methods=("priorband")
#declare -a services=("tomcatpriorband")
#declare -a ports=("8092")
#declare -a fidelity_types=("multi_fidelity")

declare -a methods=("priorband" "promise")
declare -a services=("tomcatpriorband" "tomcatpromise")
declare -a ports=("8092" "8091")
declare -a fidelity_types=("multi_fidelity" "single_fidelity")

SYSTEM="tomcat"

mkdir -p logs
chmod -R 777 logs

for i in "${!methods[@]}"; do
  method="${methods[$i]}"
  server_service="${services[$i]}"
  port="${ports[$i]}"
  fidelity="${fidelity_types[$i]}"

  for run in {0..9}; do
    echo "========= RUN ${run} START | METHOD: ${method} ========="
    run_log="logs/run${run}_${method}_${server_service}_container.log"

    echo ">>> RUN $run | METHOD: $method | FIDELITY: $fidelity | SERVER: $server_service | PORT: $port" | tee -a "$run_log"

    # Start containers
    SERVER_HOST=$server_service \
    TUNING_METHOD=$method \
    FIDELITY_TYPE=$fidelity \
    SYSTEM=$SYSTEM \
    SERVER_PORT=$port \
    RUN=$run \
    docker-compose up --build -d app_tuning $server_service >> "$run_log" 2>&1

    if [ $? -ne 0 ]; then
      echo "❌ Failed to start containers: app_tuning + $server_service" | tee -a "$run_log"
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

    echo "✅ Finished $method on $server_service for run $run" | tee -a "$run_log"
    echo "----------------------------------------"
  done
done
