# MySQL 5.7.19 Configuration Parameters

This table summarizes selected MySQL 5.7.19 configuration parameters, showing the configured defaults, allowed ranges, types, and official descriptions with documentation links for verification.

| Parameter | Default Values | Valid Range / Enum Values | Type | Description |
|-----------|----------------|--------|------|-------------|
| [`innodb_thread_concurrency`](https://dev.mysql.com/doc/refman/5.7/en/innodb-performance-thread_concurrency.html) | 0              | 0 – 1000 | integer | Limits the number of threads that can enter InnoDB concurrently. A value of 0 means no limit. |
| [`max_allowed_packet`](https://dev.mysql.com/doc/refman/5.7/en/replication-features-max-allowed-packet.html) | 4194304        | 1024 – 1073741824 | integer | Sets the maximum size of a packet that the server can handle. |
| [`innodb_io_capacity_max`](https://dev.mysql.com/doc/refman/5.7/en/innodb-parameters.html#sysvar_innodb_io_capacity_max) | 400            | 100 – 40000 | integer | Specifies the maximum number of I/O operations per second InnoDB can perform. |
| [`tmp_table_size`](https://dev.mysql.com/doc/refman/5.7/en/server-system-variables.html#sysvar_tmp_table_size) | 16777216       | 1024 – 1073741824 | integer | Sets the maximum size for internal in-memory temporary tables. |
| [`query_prealloc_size`](https://dev.mysql.com/doc/refman/5.7/en/server-system-variables.html#sysvar_query_prealloc_size) | 8192           | 8192 – 134217728 | integer | Size of the persistent buffer used for query parsing and execution. |
| [`max_heap_table_size`](https://dev.mysql.com/doc/refman/5.7/en/memory-storage-engine.html#sysvar_max_heap_table_size) | 16777216       | 16384 – 1073741824 | integer | Limits the maximum size of user-created MEMORY tables. |
| [`transaction_alloc_block_size`](https://dev.mysql.com/doc/refman/5.7/en/server-system-variables.html#sysvar_transaction_alloc_block_size) | 8192           | 1024 – 131072 | integer | Size of memory blocks allocated for transactions. |
| [`join_buffer_size`](https://dev.mysql.com/doc/refman/5.7/en/nested-loop-joins.html) | 262144         | 128 – 1073741824 | integer | Size of the buffer used for joins that do not use indexes. |
| [`innodb_flush_log_at_trx_commit`](https://dev.mysql.com/doc/refman/5.7/en/innodb-parameters.html#sysvar_innodb_flush_log_at_trx_commit) | 1              | 0, 1, 2 | enum | Controls the balance between strict ACID compliance and performance. |
| [`innodb_max_dirty_pages_pct_lwm`](https://dev.mysql.com/doc/refman/5.7/en/innodb-parameters.html#sysvar_innodb_max_dirty_pages_pct_lwm) | 0              | 0 – 99 | integer | Sets a low water mark for dirty pages in the buffer pool to trigger flushing. |
| [`innodb_buffer_pool_size`](https://dev.mysql.com/doc/refman/5.7/en/innodb-parameters.html#sysvar_innodb_buffer_pool_size) | 13958643712    | 10307921510 – 15618062894 | integer | Size of the memory buffer InnoDB uses to cache data and indexes. |
| [`key_cache_age_threshold`](https://dev.mysql.com/doc/refman/5.7/en/server-system-variables.html#sysvar_key_cache_age_threshold) | 300            | 100 – 30000 | integer | Determines the number of hits a block must have before it is considered hot. |
| [`binlog_cache_size`](https://dev.mysql.com/doc/refman/5.7/en/replication-options-binary-log.html#sysvar_binlog_cache_size) | 32768          | 4096 – 4294967296 | integer | Size of the cache to hold changes to the binary log during a transaction. |
| [`innodb_purge_rseg_truncate_frequency`](https://dev.mysql.com/doc/refman/5.7/en/innodb-parameters.html#sysvar_innodb_purge_rseg_truncate_frequency) | 128            | 1 – 128 | integer | Frequency at which the purge system looks for undo tablespaces to truncate. |
| [`query_cache_limit`](https://dev.mysql.com/doc/refman/5.7/en/query-cache-configuration.html#sysvar_query_cache_limit) | 1048576        | 0 – 134217728 | integer | Maximum size of individual query results that can be cached. |
| [`binlog_row_image`](https://dev.mysql.com/doc/refman/5.7/en/replication-options-binary-log.html#sysvar_binlog_row_image) | full           | full, minimal, noblob | enum | Controls how much information is written to the binary log when using row-based logging. |
| [`innodb_lru_scan_depth`](https://dev.mysql.com/doc/refman/5.7/en/innodb-parameters.html#sysvar_innodb_lru_scan_depth) | 1024           | 100 – 10240 | integer | Determines the number of pages scanned by the LRU manager thread. |
| [`sort_buffer_size`](https://dev.mysql.com/doc/refman/5.7/en/server-system-variables.html#sysvar_sort_buffer_size) | 262144         | 32768 – 134217728 | integer | Size of the buffer used for sorts. |
| [`innodb_stats_transient_sample_pages`](https://dev.mysql.com/doc/refman/5.7/en/innodb-parameters.html#sysvar_innodb_stats_transient_sample_pages) | 8              | 1 – 100 | integer | Number of sample pages used for transient statistics. |
| [`innodb_adaptive_max_sleep_delay`](https://dev.mysql.com/doc/refman/5.7/en/innodb-parameters.html#sysvar_innodb_adaptive_max_sleep_delay) | 150000         | 0 – 1000000 | integer | Maximum sleep delay in microseconds for the adaptive flushing thread. |
| [`innodb_adaptive_flushing_lwm`](https://dev.mysql.com/doc/refman/5.7/en/innodb-parameters.html#sysvar_innodb_adaptive_flushing_lwm) | 10             | 0 – 70 | integer | Low water mark percentage for adaptive flushing to start. |
| [`innodb_sync_array_size`](https://dev.mysql.com/doc/refman/5.7/en/innodb-parameters.html#sysvar_innodb_sync_array_size) | 1              | 1 – 1024 | integer | Number of slots in the InnoDB sync array. |
 
