FRAMAC_SHARE := $(shell frama-c -print-share-path)
FRAMAC_LIBDIR := $(shell frama-c -print-libpath)

PLUGIN_NAME := var_access_counter
PLUGIN_CMO := $(PLUGIN_NAME).cmo
PLUGIN_CMX := $(PLUGIN_NAME).cmx

.PHONY: all clean install test

all: $(PLUGIN_CMO)

$(PLUGIN_CMO): $(PLUGIN_NAME).ml
	ocamlc -I $(FRAMAC_LIBDIR) -package frama-c -c $<

$(PLUGIN_CMX): $(PLUGIN_NAME).ml
	ocamlopt -I $(FRAMAC_LIBDIR) -package frama-c -c $<

install: $(PLUGIN_CMO)
	cp $(PLUGIN_CMO) $(FRAMAC_SHARE)/plugins/

test: $(PLUGIN_CMO)
	frama-c -load-module ./$(PLUGIN_CMO) -debug-parser test.c

clean:
	rm -f *.cmo *.cmx *.cmi *.o