CC = gcc
CFLAGS = -Wall -Wextra
SRC_DIR = src
PLUGIN_DIR = src/plugins

all: logger_app

logger_app: logger_app.o json_plugin.o xml_plugin.o
	$(CC) $(CFLAGS) -o logger_app logger_app.o json_plugin.o xml_plugin.o

logger_app.o: $(SRC_DIR)/logger_app.c
	$(CC) $(CFLAGS) -c $(SRC_DIR)/logger_app.c

json_plugin.o: $(PLUGIN_DIR)/json_plugin.c
	$(CC) $(CFLAGS) -c $(PLUGIN_DIR)/json_plugin.c

xml_plugin.o: $(PLUGIN_DIR)/xml_plugin.c
	$(CC) $(CFLAGS) -c $(PLUGIN_DIR)/xml_plugin.c

clean:
	rm -f *.o logger_app