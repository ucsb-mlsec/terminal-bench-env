package com.example.kafka;

import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.Topology;
import org.apache.kafka.streams.kstream.KStream;
import org.apache.kafka.streams.processor.api.Processor;
import org.apache.kafka.streams.processor.api.ProcessorContext;
import org.apache.kafka.streams.processor.api.ProcessorSupplier;
import org.apache.kafka.streams.processor.api.Record;
import org.apache.kafka.streams.state.KeyValueStore;
import org.apache.kafka.streams.state.StoreBuilder;
import org.apache.kafka.streams.state.Stores;

public class StateStoreTopology {

    private static final String STORE_NAME = "user-sessions-store";
    private static final String INPUT_TOPIC = "user-sessions-input";

    public Topology buildTopology() {
        StreamsBuilder builder = new StreamsBuilder();

        // Create persistent key-value state store
        StoreBuilder<KeyValueStore<String, String>> storeBuilder = Stores
                .keyValueStoreBuilder(
                        Stores.persistentKeyValueStore(STORE_NAME),
                        Serdes.String(),
                        Serdes.String()
                )
                .withLoggingEnabled();

        // Add the state store to the topology
        builder.addStateStore(storeBuilder);

        // Process records from input topic
        KStream<String, String> inputStream = builder.stream(INPUT_TOPIC);

        inputStream.process(new UserSessionProcessorSupplier(), STORE_NAME);

        return builder.build();
    }

    private static class UserSessionProcessorSupplier implements ProcessorSupplier<String, String, Void, Void> {
        @Override
        public Processor<String, String, Void, Void> get() {
            return new UserSessionProcessor();
        }
    }

    private static class UserSessionProcessor implements Processor<String, String, Void, Void> {

        private ProcessorContext<Void, Void> context;
        private KeyValueStore<String, String> stateStore;

        @Override
        public void init(ProcessorContext<Void, Void> context) {
            this.context = context;
            this.stateStore = context.getStateStore(STORE_NAME);
        }

        @Override
        public void process(Record<String, String> record) {
            String key = record.key();
            String value = record.value();

            if (key != null && value != null) {
                // Read existing session data
                String existingSession = stateStore.get(key);

                // Update or create session
                if (existingSession != null) {
                    // Merge or update session data
                    String updatedSession = mergeSessionData(existingSession, value);
                    stateStore.put(key, updatedSession);
                } else {
                    // Create new session
                    stateStore.put(key, value);
                }
            }
        }

        private String mergeSessionData(String existing, String incoming) {
            // Simple merge logic - in production this would be more sophisticated
            return existing + "," + incoming;
        }

        @Override
        public void close() {
            // Cleanup if needed
        }
    }
}